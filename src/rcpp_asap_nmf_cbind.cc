#include "mmutil.hh"
#include "rcpp_asap.hh"
#include "rcpp_model_common.hh"
#include "rcpp_model_factorization.hh"
#include "gamma_parameter.hh"

//' A quick NMF estimation based on alternating Poisson regressions
//' while sharing a dictionary/factors matrix
//'
//' @param y_dn_vec a list of non-negative data matrices (gene x sample)
//' @param maxK maximum number of factors
//' @param max_iter max number of optimization steps
//' @param min_iter min number of optimization steps
//' @param burnin number of initiation steps (default: 50)
//' @param verbose verbosity
//' @param a0 gamma(a0, b0) default: a0 = 1
//' @param b0 gamma(a0, b0) default: b0 = 1
//' @param do_scale scale each column by standard deviation (default: TRUE)
//' @param do_log1p do log(1+y) transformation
//' @param rseed random seed (default: 1337)
//' @param svd_init initialize by SVD (default: FALSE)
//' @param EPS (default: 1e-8)
//'
//' @return a list that contains:
//'  \itemize{
//'   \item log.likelihood log-likelihood trace
//'   \item beta dictionary (gene x factor)
//'   \item log.beta log-dictionary (gene x factor)
//'   \item log.beta.sd sd(log-dictionary) (gene x factor)
//'   \item theta a list of loading matrices (sample x factor)
//'   \item log.theta a list of log loadings (sample x factor)
//'   \item log.theta.sd a list of standard deviations (sample x factor)
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_nmf_cbind(const std::vector<Eigen::MatrixXf> &y_dn_vec,
                   const std::size_t maxK,
                   const std::size_t max_iter = 100,
                   const std::size_t burnin = 0,
                   const bool verbose = true,
                   const double a0 = 1,
                   const double b0 = 1,
                   const bool do_log1p = false,
                   const std::size_t rseed = 1337,
                   const bool svd_init = false,
                   const double EPS = 1e-8,
                   const std::size_t NUM_THREADS = 1)
{
    Eigen::setNbThreads(NUM_THREADS);
    TLOG_(verbose, Eigen::nbThreads() << " threads");

    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Eigen::MatrixXf, RNG>;

    const Index M = y_dn_vec.size();
    ASSERT_RETL(M >= 1, "at least one data matrix is required");

    TLOG_(verbose, "Found " << M << " data matrices.");

    std::size_t K = maxK;
    Index D = 0;
    for (const Eigen::MatrixXf &y_dn : y_dn_vec) {

        TLOG_(verbose, "Y: " << y_dn.rows() << " x " << y_dn.cols());

        if (y_dn.cols() < K)
            K = y_dn.cols();
        if (D == 0) {
            D = y_dn.rows();
        } else {
            ASSERT_RETL(y_dn.rows() == D,
                        "Found inconsistent # rows: "
                            << "the previous data: " << D << " vs. "
                            << "this data:" << y_dn.rows());
        }
    }

    using model_t = factorization_t<gamma_t, gamma_t, RNG>;

    RNG rng(rseed);

    ////////////////////////////////////////////
    // We have:				      //
    // 					      //
    //   Each Y(t)_dn ~ beta_dk * theta(t)_nk //
    // 					      //
    ////////////////////////////////////////////

    TLOG_(verbose, "Building " << M << " NMF models");

    ////////////////////////////////////////
    // Create parameters and theta models //
    ////////////////////////////////////////

    gamma_t beta_dk(D, K, a0, b0, rng);

    std::vector<std::shared_ptr<gamma_t>> theta_nk_ptr_vec;
    for (const Eigen::MatrixXf &y : y_dn_vec) {
        const std::size_t n = y.cols();
        theta_nk_ptr_vec.emplace_back(
            std::make_shared<gamma_t>(n, K, a0, b0, rng));
    }

    std::vector<std::shared_ptr<model_t>> model_dn_ptr_vec;
    for (auto theta_nk_ptr : theta_nk_ptr_vec) {
        gamma_t &theta_nk = *theta_nk_ptr.get();
        model_dn_ptr_vec.emplace_back(
            std::make_shared<model_t>(beta_dk, theta_nk, RSEED(rseed)));
    }

    using norm_dist_t = boost::random::normal_distribution<Scalar>;
    norm_dist_t norm_dist(0., 1.);
    auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };

    for (std::size_t m = 0; m < M; ++m) {
        model_t &model_dn = *model_dn_ptr_vec[m].get();
        gamma_t &theta_nk = *theta_nk_ptr_vec[m].get();
        const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);
        const std::size_t n = y_dn.cols();
        const Mat temp_nk = Mat::NullaryExpr(n, K, rnorm);
        exp_op<Mat> exp;
        theta_nk.update(temp_nk.unaryExpr(exp), Mat::Ones(n, K));
        theta_nk.calibrate();
    }

    Scalar llik = 0;
    for (std::size_t m = 0; m < M; ++m) {
        model_t &model_dn = *model_dn_ptr_vec[m].get();
        const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);
        const Scalar _llik = log_likelihood(model_dn, y_dn);
        TLOG_(verbose, "llik [" << m << "] " << _llik);
        llik += _llik;
    }

    TLOG_(verbose, "Finished initialization: " << llik);

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter + burnin + 1);
    llik_trace.emplace_back(llik);

    for (std::size_t tt = 0; tt < (burnin + max_iter); ++tt) {

        ///////////////////////////////////////////
        // Add matrix data to each model's stats //
        ///////////////////////////////////////////

        beta_dk.reset_stat_only();

        for (std::size_t m = 0; m < M; ++m) {
            model_t &model_dn = *model_dn_ptr_vec[m].get();
            gamma_t &theta_nk = *theta_nk_ptr_vec[m].get();
            const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);
            // a. Update theta based on the current beta
            theta_nk.reset_stat_only();
            add_stat_by_col(model_dn, y_dn, STOCH(tt < burnin), STD(true));
            theta_nk.calibrate();
            // b. Update beta factors based on the new theta
            theta_nk.reset_stat_only();
            add_stat_by_row(model_dn, y_dn, STOCH(tt < burnin), STD(false));
            theta_nk.calibrate();
        }

        beta_dk.calibrate();

        llik = 0;

        for (std::size_t m = 0; m < M; ++m) {
            model_t &model_dn = *model_dn_ptr_vec[m].get();
            const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);
            const Scalar _llik = log_likelihood(model_dn, y_dn);
            llik += _llik;
        }

        const Scalar diff =
            (llik_trace.size() > 0 ?
                 (std::abs(llik - llik_trace.at(llik_trace.size() - 1)) /
                  std::abs(llik + EPS)) :
                 llik);

        TLOG_(verbose, "NMF [ " << tt << " ] " << llik << ", " << diff);

        llik_trace.emplace_back(llik);

        if (tt > burnin && diff < EPS) {
            TLOG("Converged at " << tt << ", " << diff);
            break;
        }

        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            WLOG("Interruption by the user at t=" << tt);
            break;
        }
    }

    Rcpp::List theta_log_mean_list(M);
    Rcpp::List theta_log_sd_list(M);
    Rcpp::List theta_mean_list(M);

    for (std::size_t m = 0; m < M; ++m) {
        theta_mean_list[m] = theta_nk_ptr_vec[m].get()->mean();
        theta_log_mean_list[m] = theta_nk_ptr_vec[m].get()->log_mean();
        theta_log_sd_list[m] = theta_nk_ptr_vec[m].get()->log_sd();
    }

    TLOG_(verbose, "Done");

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["beta"] = beta_dk.mean(),
                              Rcpp::_["log.beta"] = beta_dk.log_mean(),
                              Rcpp::_["log.beta.sd"] = beta_dk.log_sd(),
                              Rcpp::_["theta.list"] = theta_mean_list,
                              Rcpp::_["log.theta.sd.list"] = theta_log_sd_list,
                              Rcpp::_["log.theta.list"] = theta_log_mean_list);
}
