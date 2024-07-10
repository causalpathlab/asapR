#include "rcpp_asap_pmf.hh"

//' A quick PMF estimation based on alternating Poisson regressions
//'
//' @param Y_ non-negative data matrix (gene x sample)
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
//' @param jitter (default: 0.1)
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
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_pmf(const Eigen::MatrixXf Y_,
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
    using model_t = factorization_t<gamma_t, gamma_t, RNG>;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;

    exp_op<Mat> exp;
    log1p_op<Mat> log1p;
    Mat Y_dn = do_log1p ? Y_.unaryExpr(log1p) : Y_;

    const RowVec row_sum = Y_dn.colwise().sum();

    if (normalize_cols) {
        const Scalar col_norm = Y_dn.rows();
        Y_dn.array().rowwise() /= (row_sum.array() + EPS);
        Y_dn *= col_norm;
    }

    TLOG_(verbose, "Data: " << Y_dn.rows() << " x " << Y_dn.cols());

    ///////////////////////
    // Create parameters //
    ///////////////////////

    const std::size_t D = Y_dn.rows(), N = Y_dn.cols();
    const std::size_t K = std::min(std::min(maxK, N), D);

    const bool do_stdize_row = (N > D), do_stdize_col = (D >= N);

    RNG rng(rseed);
    gamma_t beta_dk(D, K, a0, b0, rng);
    gamma_t theta_nk(N, K, a0, b0, rng);

    model_t model_dn(beta_dk, theta_nk, RSEED(rseed), NThreads(nthreads));

    Scalar llik = 0;
    initialize_stat(model_dn, Y_dn, DO_SVD(svd_init), jitter);
    llik = log_likelihood(model_dn, Y_dn);
    TLOG_(verbose, "Finished initialization: " << llik);

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter + 1);
    llik_trace.emplace_back(llik);

    for (std::size_t tt = 0; tt < max_iter; ++tt) {

        theta_nk.reset_stat_only();
        add_stat_to_col(model_dn,
                        Y_dn,
                        DO_AUX_STD(do_stdize_col),
                        DO_DEGREE_CORRECTION(do_degree_correction));
        theta_nk.calibrate();

        beta_dk.reset_stat_only();
        add_stat_to_row(model_dn,
                        Y_dn,
                        DO_AUX_STD(do_stdize_row),
                        DO_DEGREE_CORRECTION(do_degree_correction));
        beta_dk.calibrate();

        llik = log_likelihood(model_dn, Y_dn);

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

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["beta"] = beta_dk.mean(),
                              Rcpp::_["log.beta"] = beta_dk.log_mean(),
                              Rcpp::_["log.beta.sd"] = beta_dk.log_sd(),
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta.sd"] = theta_nk.log_sd(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["row.sum"] = row_sum);
}
