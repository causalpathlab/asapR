#include "rcpp_asap_pmf_delta.hh"

//' A quick PMF estimation based on alternating Poisson regressions
//' across multiple matrices with the same dimensionality
//'
//' @param y_ref reference data matrix (gene x sample)
//' @param y_dn_vec a list of non-negative data matrices (gene x sample)
//' @param maxK maximum number of factors
//' @param max_iter max number of optimization steps
//' @param min_iter min number of optimization steps
//' @param verbose verbosity
//' @param a0 gamma(a0, b0) default: a0 = 1
//' @param b0 gamma(a0, b0) default: b0 = 1
//' @param normalize_cols normalize columns by col_norm (default: FALSE)
//' @param do_log1p do log(1+y) transformation
//' @param rseed random seed (default: 1337)
//' @param svd_init initialize by SVD (default: FALSE)
//' @param EPS (default: 1e-8)
//' @param jitter (default: 1)
//'
//' @return a list that contains:
//'  \itemize{
//'   \item log.likelihood log-likelihood trace
//'   \item theta loading (sample x factor)
//'   \item log.theta log-loading (sample x factor)
//'   \item log.theta.sd sd(log-loading) (sample x factor)
//'   \item beta dictionary (gene x factor)
//'   \item log.beta log dictionary (gene x factor)
//'   \item log.beta.sd sd(log-dictionary) (gene x factor)
//'   \item delta a list of dictionary matrices (gene x factor)
//'   \item log.delta a list of log dictionary (gene x factor)
//'   \item log.delta.sd a list of standard deviations (gene x factor)
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_pmf_delta(const Eigen::MatrixXf y_ref,
                   const std::vector<Eigen::MatrixXf> y_dn_vec,
                   const std::size_t maxK,
                   const std::size_t max_iter = 100,
                   const bool verbose = true,
                   const double a0 = 1,
                   const double b0 = 1,
                   const bool do_log1p = false,
                   const std::size_t rseed = 1337,
                   const bool svd_init = false,
                   const bool do_degree_correction = false,
                   const bool normalize_cols = false,
                   const double EPS = 1e-8,
                   const double jitter = 1.0,
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

    log1p_op<Mat> log1p;
    Mat Yref_dn = do_log1p ? y_ref.unaryExpr(log1p) : y_ref;

    TLOG_(verbose, "Data: " << Yref_dn.rows() << " x " << Yref_dn.cols());

    const std::size_t D = Yref_dn.rows(), N = Yref_dn.cols();

    for (const Eigen::MatrixXf &y_dn : y_dn_vec) {
        TLOG_(verbose, "Y: " << y_dn.rows() << " x " << y_dn.cols());

        ASSERT_RETL(y_dn.cols() == N,
                    "Found inconsistent # columns: "
                        << "the previous data: " << N << " vs. "
                        << "this data:" << y_dn.cols());
        ASSERT_RETL(y_dn.rows() == D,
                    "Found inconsistent # rows: "
                        << "the previous data: " << N << " vs. "
                        << "this data:" << y_dn.rows());
    }

    const bool do_stdize_row = (N > D), do_stdize_col = (D >= N);

    using ref_model_t = factorization_t<gamma_t, gamma_t>;
    using delta_model_t = factorization_delta_t<gamma_t, gamma_t>;

    RNG rng(rseed);

    //////////////////////////////////////
    // We have:				//
    // 					//
    //   Y_ref_dn ~ beta_dk * theta_nk  //
    // 					//
    //////////////////////////////////////

    gamma_t theta_nk(N, K, a0, b0, rng);
    gamma_t beta_dk(D, K, a0, b0, rng);

    ref_model_t ref_model_dn(beta_dk, theta_nk, NThreads(nthreads));

    initialize_stat(ref_model_dn,
                    Yref_dn,
                    DO_SVD(svd_init),
                    DO_DEGREE_CORRECTION(do_degree_correction),
                    jitter);

    std::vector<std::shared_ptr<gamma_t>> delta_dk_ptr_vec;

    for (const Eigen::MatrixXf &y : y_dn_vec) {
        const std::size_t d = y.rows();
        delta_dk_ptr_vec.emplace_back(
            std::make_shared<gamma_t>(d, K, a0, b0, rng));
    }

    ///////////////////////////////////////////////////////
    // We have:				                 //
    // 					                 //
    //   Each Y(t)_dn ~ beta_dk * delta_dk(t) * theta_nk //
    // 					                 //
    ///////////////////////////////////////////////////////

    std::vector<std::shared_ptr<delta_model_t>> delta_model_dn_ptr_vec;

    for (auto delta_dk_ptr : delta_dk_ptr_vec) {
        gamma_t &delta_dk = *delta_dk_ptr.get();
        delta_dk.reset_stat_only();
        delta_dk.calibrate();

        delta_model_dn_ptr_vec.emplace_back(
            std::make_shared<delta_model_t>(beta_dk,
                                            delta_dk,
                                            theta_nk,
                                            NThreads(nthreads)));
    }

    Scalar llik = log_likelihood(ref_model_dn, Yref_dn);
    TLOG_(verbose, "Reference data:  " << llik);

    for (std::size_t m = 0; m < M; ++m) {
        delta_model_t &delta_model_dn = *delta_model_dn_ptr_vec[m].get();
        const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);
        const Scalar _llik = log_likelihood(delta_model_dn, y_dn);
        TLOG_(verbose, "Delta data [" << m << "]:  " << _llik);
        llik += _llik;
    }

    TLOG_(verbose, "Finished initialization: " << llik);

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter + 1);
    llik_trace.emplace_back(llik);

    for (std::size_t tt = 0; tt < (max_iter); ++tt) {

        //////////////////
        // global model //
        //////////////////

        theta_nk.reset_stat_only();
        add_stat_to_col(ref_model_dn, Yref_dn, DO_AUX_STD(do_stdize_col));
        theta_nk.calibrate();

        beta_dk.reset_stat_only();
        add_stat_to_row(ref_model_dn, Yref_dn, DO_AUX_STD(do_stdize_row));
        beta_dk.calibrate();

        ///////////////////////////////////////////
        // Add matrix data to each model's stats //
        ///////////////////////////////////////////

        for (std::size_t m = 0; m < M; ++m) {
            delta_model_t &delta_model_dn = *delta_model_dn_ptr_vec[m].get();
            gamma_t &delta_dk = *delta_dk_ptr_vec[m].get();
            const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);

            // a. Update theta based on the current beta and delta
            add_stat_to_col(delta_model_dn, y_dn, DO_AUX_STD(do_stdize_col));
            theta_nk.calibrate();

            // b. Update delta and beta factors based on the new theta
            delta_dk.reset_stat_only();
            add_stat_to_row(delta_model_dn, y_dn, DO_AUX_STD(do_stdize_row));
            delta_dk.calibrate();
            beta_dk.calibrate();
        }

        beta_dk.calibrate();
        theta_nk.calibrate();

        llik = log_likelihood(ref_model_dn, Yref_dn);
        // TLOG_(verbose, "Reference data:  " << llik);

        for (std::size_t m = 0; m < M; ++m) {
            delta_model_t &delta_model_dn = *delta_model_dn_ptr_vec[m].get();
            const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);
            const Scalar _llik = log_likelihood(delta_model_dn, y_dn);
            // TLOG_(verbose, "Delta data [" << m << "]:  " << _llik);
            llik += _llik;
        }

        const Scalar diff =
            (llik_trace.size() > 0 ?
                 (std::abs(llik - llik_trace.at(llik_trace.size() - 1)) /
                  std::abs(llik + EPS)) :
                 llik);

        TLOG_(verbose, "Delta [ " << tt << " ] " << llik << ", " << diff);

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

    Rcpp::List delta_log_mean_list(M);
    Rcpp::List delta_log_sd_list(M);
    Rcpp::List delta_mean_list(M);

    for (std::size_t m = 0; m < M; ++m) {
        delta_mean_list[m] = delta_dk_ptr_vec[m].get()->mean();
        delta_log_mean_list[m] = delta_dk_ptr_vec[m].get()->log_mean();
        delta_log_sd_list[m] = delta_dk_ptr_vec[m].get()->log_sd();
    }

    TLOG_(verbose, "Done");

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.theta.sd"] = theta_nk.log_sd(),
                              Rcpp::_["beta"] = beta_dk.mean(),
                              Rcpp::_["log.beta"] = beta_dk.log_mean(),
                              Rcpp::_["log.beta.sd"] = beta_dk.log_sd(),
                              Rcpp::_["delta.list"] = delta_mean_list,
                              Rcpp::_["log.delta.sd.list"] = delta_log_sd_list,
                              Rcpp::_["log.delta.list"] = delta_log_mean_list);
}
