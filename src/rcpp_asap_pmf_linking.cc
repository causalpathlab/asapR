#include "rcpp_asap_pmf_linking.hh"

//' A quick PMF estimation based on alternating Poisson regressions
//' across multiple matrices with the same dimensionality
//'
//' @param X_ global data matrix (global feature x sample)
//' @param y_dn_vec a list of non-negative data matrices (local feature x sample)
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
//'   \item beta dictionary (global feature x factor)
//'   \item log.beta log dictionary (global feature x factor)
//'   \item log.beta.sd sd(log-dictionary) (global feature x factor)
//'   \item alpha a list of dictionary matrices (local feature[t] x global feature)
//'   \item log.alpha a list of log dictionary (local feature[t] x global feature)
//'   \item log.alpha.sd a list of standard deviations (local feature[t] x global feature)
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_pmf_linking(const Eigen::MatrixXf X_,
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

    Mat x_ln = do_log1p ? X_.unaryExpr(log1p) : X_;

    TLOG_(verbose, "Data: " << x_ln.rows() << " x " << x_ln.cols());

    const std::size_t L = x_ln.rows(), N = x_ln.cols();

    for (const Eigen::MatrixXf &y_dn : y_dn_vec) {
        TLOG_(verbose, "Y: " << y_dn.rows() << " x " << y_dn.cols());

        ASSERT_RETL(y_dn.cols() == N,
                    "Found inconsistent # columns: "
                        << "the previous data: " << N << " vs. "
                        << "this data:" << y_dn.cols());
    }

    const bool do_stdize_mid = (N > L), do_stdize_col = (L >= N);
    const bool do_stdize_row = false;

    RNG rng(rseed);
    using ref_model_t = factorization_t<gamma_t, gamma_t>;
    using linking_model_t = factorization_linking_t<gamma_t, gamma_t, gamma_t>;

    gamma_t theta_nk(N, K, a0, b0, rng);
    gamma_t beta_lk(L, K, a0, b0, rng);

    std::vector<std::shared_ptr<gamma_t>> alpha_dl_ptr_vec;

    for (const Eigen::MatrixXf &y : y_dn_vec) {
        const std::size_t d = y.rows();
        alpha_dl_ptr_vec.emplace_back(
            std::make_shared<gamma_t>(d, L, a0, b0, rng));
    }

    ///////////////////////////////////////////////////////
    // We have:				                 //
    // 					                 //
    //   Each Y(t)_dn ~ alpha_dl(t) * beta_lk * theta_nk //
    // 					                 //
    ///////////////////////////////////////////////////////

    std::vector<std::shared_ptr<linking_model_t>> linking_model_dn_ptr_vec;

    for (auto alpha_dl_ptr : alpha_dl_ptr_vec) {
        gamma_t &alpha_dl = *alpha_dl_ptr.get();
        linking_model_dn_ptr_vec.emplace_back(
            std::make_shared<linking_model_t>(alpha_dl,
                                              beta_lk,
                                              theta_nk,
                                              NThreads(nthreads)));
    }

    for (std::size_t m = 0; m < M; ++m) {
        linking_model_t &linking_model_dn = *linking_model_dn_ptr_vec[m].get();
        const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);
        initialize_stat(linking_model_dn,
                        y_dn,
                        DO_SVD(svd_init),
                        DO_DEGREE_CORRECTION(do_degree_correction),
                        jitter);
    }

    ////////////////////////////
    // reference global model //
    ////////////////////////////

    ref_model_t ref_model_ln(beta_lk, theta_nk, NThreads(nthreads));
    initialize_stat(ref_model_ln,
                    x_ln,
                    DO_SVD(svd_init),
                    DO_DEGREE_CORRECTION(do_degree_correction),
                    jitter);

    Scalar llik = log_likelihood(ref_model_ln, x_ln);

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter + 1);
    llik_trace.emplace_back(llik);

    for (std::size_t tt = 0; tt < (max_iter); ++tt) {

        theta_nk.reset_stat_only();
        add_stat_to_col(ref_model_ln, x_ln, DO_AUX_STD(do_stdize_col));
        theta_nk.calibrate();

        beta_lk.reset_stat_only();
        add_stat_to_row(ref_model_ln, x_ln, DO_AUX_STD(do_stdize_mid));
        beta_lk.calibrate();

        llik = log_likelihood(ref_model_ln, x_ln);

        const Scalar diff =
            (llik_trace.size() > 0 ?
                 (std::abs(llik - llik_trace.at(llik_trace.size() - 1)) /
                  std::abs(llik + EPS)) :
                 llik);

        TLOG_(verbose, "Reference [ " << tt << " ] " << llik << ", " << diff);

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

    llik = log_likelihood(ref_model_ln, x_ln);
    TLOG_(verbose, "Reference data:  " << llik);

    for (std::size_t m = 0; m < M; ++m) {
        linking_model_t &linking_model_dn = *linking_model_dn_ptr_vec[m].get();
	linking_model_dn.alpha_dl.reset_stat_only();
	linking_model_dn.alpha_dl.calibrate();
        const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);
        const Scalar _llik = log_likelihood(linking_model_dn, y_dn);
        TLOG_(verbose, "linking data [" << m << "]:  " << _llik);
        llik += _llik;
    }

    TLOG_(verbose, "Finished initialization: " << llik);

    llik_trace.clear();
    llik_trace.reserve(max_iter + 1);

    for (std::size_t tt = 0; tt < (max_iter); ++tt) {

        // theta_nk.reset_stat_only();
        // beta_lk.reset_stat_only();

        ///////////////////////////////////////////
        // Add matrix data to each model's stats //
        ///////////////////////////////////////////

        for (std::size_t m = 0; m < M; ++m) {
            linking_model_t &linking_model_dn =
                *linking_model_dn_ptr_vec[m].get();

            const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);

            gamma_t &alpha_dl = *alpha_dl_ptr_vec[m].get();
            alpha_dl.reset_stat_only();
            add_stat_to_row(linking_model_dn, y_dn, DO_AUX_STD(do_stdize_row));
            alpha_dl.calibrate();

            // add_stat_to_mid(linking_model_dn, y_dn, DO_AUX_STD(do_stdize_mid));
            // add_stat_to_col(linking_model_dn, y_dn, DO_AUX_STD(do_stdize_col));
        }

        // add_stat_to_col(ref_model_ln, x_ln, DO_AUX_STD(do_stdize_col));
        // theta_nk.calibrate();

        // add_stat_to_row(ref_model_ln, x_ln, DO_AUX_STD(do_stdize_mid));
        // beta_lk.calibrate();

        llik = log_likelihood(ref_model_ln, x_ln);

        // TLOG_(verbose, "Reference data:  " << llik);

        for (std::size_t m = 0; m < M; ++m) {
            linking_model_t &linking_model_dn =
                *linking_model_dn_ptr_vec[m].get();
            const Eigen::MatrixXf &y_dn = y_dn_vec.at(m);
            const Scalar _llik = log_likelihood(linking_model_dn, y_dn);
            // TLOG_(verbose, "linking data [" << m << "]:  " << _llik);
            llik += _llik;
        }

        const Scalar diff =
            (llik_trace.size() > 0 ?
                 (std::abs(llik - llik_trace.at(llik_trace.size() - 1)) /
                  std::abs(llik + EPS)) :
                 llik);

        TLOG_(verbose, "Linking [ " << tt << " ] " << llik << ", " << diff);

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

    Rcpp::List alpha_log_mean_list(M);
    Rcpp::List alpha_log_sd_list(M);
    Rcpp::List alpha_mean_list(M);

    for (std::size_t m = 0; m < M; ++m) {
        alpha_mean_list[m] = alpha_dl_ptr_vec[m].get()->mean();
        alpha_log_mean_list[m] = alpha_dl_ptr_vec[m].get()->log_mean();
        alpha_log_sd_list[m] = alpha_dl_ptr_vec[m].get()->log_sd();
    }

    TLOG_(verbose, "Done");

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.theta.sd"] = theta_nk.log_sd(),
                              Rcpp::_["beta"] = beta_lk.mean(),
                              Rcpp::_["log.beta"] = beta_lk.log_mean(),
                              Rcpp::_["log.beta.sd"] = beta_lk.log_sd(),
                              Rcpp::_["alpha.list"] = alpha_mean_list,
                              Rcpp::_["log.alpha.sd.list"] = alpha_log_sd_list,
                              Rcpp::_["log.alpha.list"] = alpha_log_mean_list);
}
