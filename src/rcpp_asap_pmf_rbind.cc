#include "rcpp_asap_pmf_rbind.hh"

//' A quick PMF estimation based on alternating Poisson regressions
//' while sharing a factor loading/topic proportion matrix
//'
//' @param y_dn_vec a list of non-negative data matrices (gene x sample)
//' @param maxK maximum number of factors
//' @param max_iter max number of optimization steps
//' @param min_iter min number of optimization steps
//' @param verbose verbosity
//' @param a0 gamma(a0, b0) default: a0 = 1
//' @param b0 gamma(a0, b0) default: b0 = 1
//' @param do_scale scale each column by standard deviation (default: TRUE)
//' @param do_log1p do log(1+y) transformation
//' @param rseed random seed (default: 1337)
//' @param EPS (default: 1e-8)
//'
//' @return a list that contains:
//'  \itemize{
//'   \item log.likelihood log-likelihood trace
//'   \item theta loading (sample x factor)
//'   \item log.theta log-loading (sample x factor)
//'   \item log.theta.sd sd(log-loading) (sample x factor)
//'   \item beta a list of dictionary matrices (gene x factor)
//'   \item log.beta a list of log dictionary (gene x factor)
//'   \item log.beta.sd a list of standard deviations (gene x factor)
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_pmf_rbind(const std::vector<Eigen::MatrixXf> y_dn_vec,
                   const std::size_t maxK,
                   const std::size_t max_iter = 100,
                   const bool verbose = true,
                   const double a0 = 1,
                   const double b0 = 1,
                   const bool do_log1p = false,
                   const std::size_t rseed = 1337,
                   const double EPS = 1e-8,
                   const std::size_t NUM_THREADS = 0)
{
    const std::size_t nthreads =
        (NUM_THREADS > 0 ? NUM_THREADS : omp_get_max_threads());

    Eigen::setNbThreads(nthreads);
    TLOG_(verbose, Eigen::nbThreads() << " threads");

    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Eigen::MatrixXf, RNG>;

    const Index M = y_dn_vec.size();
    ASSERT_RETL(M >= 1, "at least one data matrix is required");

    TLOG_(verbose, "Found " << M << " data matrices.");

    std::size_t K = maxK;

    Index N = 0;
    for (const Eigen::MatrixXf &y_dn : y_dn_vec) {
        TLOG_(verbose, "Y: " << y_dn.rows() << " x " << y_dn.cols());

        if (N == 0) {
            N = y_dn.cols();
        } else {
            ASSERT_RETL(y_dn.cols() == N,
                        "Found inconsistent # columns: "
                            << "the previous data: " << N << " vs. "
                            << "this data:" << y_dn.cols());
        }
    }

    using model_t = factorization_t<gamma_t, gamma_t, RNG>;

    RNG rng(rseed);

    ////////////////////////////////////////////
    // We have:				      //
    // 					      //
    //   Each Y(t)_dn ~ beta_dk(t) * theta_nk //
    // 					      //
    ////////////////////////////////////////////

    TLOG_(verbose, "Building " << M << " PMF models");

    ///////////////////////////////////////
    // Create parameters and beta models //
    ///////////////////////////////////////

    gamma_t theta_nk(N, K, a0, b0, rng);

    {
        Mat temp_nk = theta_nk.sample();
        theta_nk.update(temp_nk, Mat::Ones(N, K));
        theta_nk.calibrate();
    }

    std::vector<std::shared_ptr<gamma_t>> beta_dk_ptr_vec;

    for (const Eigen::MatrixXf &y : y_dn_vec) {
        const std::size_t d = y.rows();
        beta_dk_ptr_vec.emplace_back(
            std::make_shared<gamma_t>(d, K, a0, b0, rng));
    }

    std::vector<std::shared_ptr<model_t>> model_dn_ptr_vec;
    for (auto beta_dk_ptr : beta_dk_ptr_vec) {
        gamma_t &beta_dk = *beta_dk_ptr.get();
        model_dn_ptr_vec.emplace_back(
            std::make_shared<model_t>(beta_dk,
                                      theta_nk,
                                      RSEED(rseed),
                                      NThreads(nthreads)));
    }

    using norm_dist_t = boost::random::normal_distribution<Scalar>;
    norm_dist_t norm_dist(0., 1.);
    auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };

    for (std::size_t m = 0; m < M; ++m) {
        gamma_t &beta_dk = *beta_dk_ptr_vec[m].get();
        const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);
        const std::size_t d = y_dn.rows();
        const Mat temp_dk = Mat::NullaryExpr(d, K, rnorm);
        exp_op<Mat> exp;
        beta_dk.update(temp_dk.unaryExpr(exp), Mat::Ones(d, K));
        beta_dk.calibrate();
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
    llik_trace.reserve(max_iter + 1);
    llik_trace.emplace_back(llik);

    for (std::size_t tt = 0; tt < (max_iter); ++tt) {

        ///////////////////////////////////////////
        // Add matrix data to each model's stats //
        ///////////////////////////////////////////

        theta_nk.reset_stat_only();

        for (std::size_t m = 0; m < M; ++m) {
            model_t &model_dn = *model_dn_ptr_vec[m].get();
            gamma_t &beta_dk = *beta_dk_ptr_vec[m].get();
            const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);

            // a. Update beta factors based on the new theta
            beta_dk.reset_stat_only();
            add_stat_to_row(model_dn, y_dn, STD(false));
            beta_dk.calibrate();

            // b. Update theta based on the current beta
            add_stat_to_col(model_dn, y_dn, STD(true));
        }

        theta_nk.calibrate();

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

        TLOG_(verbose, "PMF [ " << tt << " ] " << llik << ", " << diff);

        llik_trace.emplace_back(llik);

        if (tt > 1 && diff < EPS) {
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

    Rcpp::List beta_log_mean_list(M);
    Rcpp::List beta_log_sd_list(M);
    Rcpp::List beta_mean_list(M);

    for (std::size_t m = 0; m < M; ++m) {
        beta_mean_list[m] = beta_dk_ptr_vec[m].get()->mean();
        beta_log_mean_list[m] = beta_dk_ptr_vec[m].get()->log_mean();
        beta_log_sd_list[m] = beta_dk_ptr_vec[m].get()->log_sd();
    }

    TLOG_(verbose, "Done");

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.theta.sd"] = theta_nk.log_sd(),
                              Rcpp::_["beta.list"] = beta_mean_list,
                              Rcpp::_["log.beta.sd.list"] = beta_log_sd_list,
                              Rcpp::_["log.beta.list"] = beta_log_mean_list);
}
