#include "rcpp_asap_pmf_linking.hh"

//' Estimate two-layered PMF (experimental)
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_pmf_linking(const Eigen::MatrixXf X_,
                     const Eigen::MatrixXf Y_,
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
                     const std::size_t NUM_THREADS = 0)
{

    const std::size_t nthreads =
        (NUM_THREADS > 0 ? NUM_THREADS : omp_get_max_threads());

    Eigen::setNbThreads(nthreads);
    TLOG_(verbose, Eigen::nbThreads() << " threads");

    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Eigen::MatrixXf, RNG>;

    exp_op<Mat> exp;
    log1p_op<Mat> log1p;
    const Mat X_ln = do_log1p ? X_.unaryExpr(log1p) : X_;
    const Mat Y_dn = do_log1p ? Y_.unaryExpr(log1p) : Y_;

    TLOG_(verbose, "X: " << X_ln.rows() << " x " << X_ln.cols());
    TLOG_(verbose, "Y: " << Y_dn.rows() << " x " << Y_dn.cols());

    ASSERT_RETL(Y_dn.cols() == X_ln.cols(), "Found inconsistent # columns");

    ///////////////////////
    // Create parameters //
    ///////////////////////

    const std::size_t D = Y_dn.rows(), N = Y_dn.cols(), L = X_ln.rows();
    const std::size_t K = std::min(std::min(std::min(maxK, N), D), L);
    RNG rng(rseed);

    gamma_t beta_dl(D, L, a0, b0, rng);
    gamma_t alpha_lk(L, K, a0, b0, rng);
    gamma_t theta_nk(N, K, a0, b0, rng);

    factorization_t<gamma_t, gamma_t, RNG> model_x_ln(alpha_lk,
                                                      theta_nk,
                                                      RSEED(rseed),
                                                      NThreads(nthreads));

    factorization_three_t<gamma_t, gamma_t, gamma_t, RNG>
        model_xy_dn(beta_dl,
                    alpha_lk,
                    theta_nk,
                    RSEED(rseed),
                    NThreads(nthreads));

    ////////////////////
    // initialization //
    ////////////////////

    Scalar llik = 0;
    initialize_stat(model_x_ln, X_ln, DO_SVD { svd_init });
    initialize_stat(model_xy_dn, Y_dn, DO_SVD { svd_init });

    //////////////////////
    // pretrain model_x //
    //////////////////////

    std::vector<Scalar> llik_x_trace;
    llik_x_trace.reserve(max_iter + 1);

    llik = log_likelihood(model_x_ln, X_ln);
    llik_x_trace.emplace_back(llik);

    for (std::size_t tt = 0; tt < max_iter; ++tt) {
        theta_nk.reset_stat_only();
        add_stat_to_col(model_x_ln, X_ln, STD(true));
        theta_nk.calibrate();

        alpha_lk.reset_stat_only();
        add_stat_to_row(model_x_ln, X_ln, STD(false));
        alpha_lk.calibrate();

        llik = log_likelihood(model_x_ln, X_ln);

        const Scalar diff =
            (llik_x_trace.size() > 0 ?
                 (std::abs(llik - llik_x_trace.at(llik_x_trace.size() - 1)) /
                  std::abs(llik + EPS)) :
                 llik);

        TLOG_(verbose, "PMF X [ " << tt << " ] " << llik << ", " << diff);

        llik_x_trace.emplace_back(llik);

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

    ////////////////////
    // joint training //
    ////////////////////

    llik = log_likelihood(model_xy_dn, Y_dn);
    TLOG_(verbose, "Finished initialization: " << llik);

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter + 1);
    llik_trace.emplace_back(llik);

    for (std::size_t tt = 0; tt < max_iter; ++tt) {

        beta_dl.reset_stat_only();
        add_stat_to_row(model_xy_dn, Y_dn, STD(false));
        beta_dl.calibrate();

        // alpha_lk.reset_stat_only();
        // add_stat_to_row(model_x_ln, X_ln, STD(false));
        // add_stat_to_mid(model_xy_dn, Y_dn, STD(false));
        // alpha_lk.calibrate();

        // theta_nk.reset_stat_only();
        // add_stat_to_col(model_x_ln, X_ln, STD(true));
        // add_stat_to_col(model_xy_dn, Y_dn, STD(true));
        // theta_nk.calibrate();

        llik = log_likelihood(model_xy_dn, Y_dn);

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
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.theta.sd"] = theta_nk.log_sd(),
                              Rcpp::_["alpha"] = alpha_lk.mean(),
                              Rcpp::_["log.alpha"] = alpha_lk.log_mean(),
                              Rcpp::_["log.alpha.sd"] = alpha_lk.log_sd(),
                              Rcpp::_["beta"] = beta_dl.mean(),
                              Rcpp::_["log.beta"] = beta_dl.log_mean(),
                              Rcpp::_["log.beta.sd"] = beta_dl.log_sd());
}
