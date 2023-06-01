#include "rcpp_asap.hh"

//' A quick NMF estimation based on alternating Poisson regressions
//'
//' @param Y_ non-negative data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param max_iter max number of optimization steps
//' @param min_iter min number of optimization steps
//' @param burnin number of initiation steps
//' @param verbose verbosity
//' @param a0 gamma(a0, b0) default: a0 = 1
//' @param b0 gamma(a0, b0) default: b0 = 1
//' @param do_log1p do log(1+y) transformation
//' @param rseed random seed (default: 1337)
//' @param svd_init initialize by SVD (default: TRUE)
//' @param EPS (default: 1e-4)
//'
//' @return a list that contains:
//'  \itemize{
//'   \item log.likelihood log-likelihood trace
//'   \item beta dictionary (gene x factor)
//'   \item log.beta log-dictionary (gene x factor)
//'   \item theta loading (sample x factor)
//'   \item log.theta log-loading (sample x factor)
//'   \item log.phi auxiliary variables (gene x factor)
//'   \item log.rho auxiliary variables (sample x factor)
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_nmf_alternate(const Eigen::MatrixXf Y_,
                       const std::size_t maxK,
                       const std::size_t max_iter = 100,
                       const std::size_t burnin = 10,
                       const bool verbose = true,
                       const double a0 = 1,
                       const double b0 = 1,
                       const bool do_log1p = false,
                       const std::size_t rseed = 1337,
                       const bool svd_init = true,
                       const double EPS = 1e-4)
{
    using RNG = dqrng::xoshiro256plus;
    using model_t = asap_nmf_model_t<RNG>;

    log1p_op<Mat> log1p;
    const Mat Y_dn = do_log1p ? Y_.unaryExpr(log1p) : Y_;

    model_t model(model_t::ROW(Y_dn.rows()),
                  model_t::COL(Y_dn.cols()),
                  model_t::FACT(maxK),
                  model_t::RSEED(rseed),
                  model_t::A0(a0),
                  model_t::B0(b0));

    model.precompute(Y_dn);

    if (svd_init) {
        model.initialize_by_svd(Y_dn);
    } else {
        model.initialize_random();
    }

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter + burnin);

    for (Index tt = 0; tt < (burnin + max_iter); ++tt) {

        model.update_by_row(Y_dn, tt <= burnin);
        model.update_by_col(Y_dn, EPS);

        const Scalar llik = model.log_likelihood(Y_dn);
        const Scalar diff = llik_trace.size() > 0 ?
            std::abs(llik - llik_trace.at(llik_trace.size() - 1)) /
                std::abs(llik + EPS) :
            llik;

        if (tt >= burnin) {
            llik_trace.emplace_back(llik);
        }

        if (verbose) {
            TLOG("NMF [ " << tt << " ] " << llik << ", " << diff);
        } else {
            Rcpp::Rcerr << "+ " << std::flush;
            if (tt > 0 && tt % 10 == 0) {
                Rcpp::Rcerr << "\r" << std::flush;
            }
        }

        if (tt > burnin && diff < EPS) {
            Rcpp::Rcerr << "\r" << std::endl;
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
    Rcpp::Rcerr << "\r" << std::endl;
    TLOG("Done");

    Mat log_x, R_nk;
    std::tie(log_x, R_nk) = model.log_topic_correlation(Y_dn);

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["beta"] = model.beta_dk.mean(),
                              Rcpp::_["log.beta"] = model.beta_dk.log_mean(),
                              Rcpp::_["log_x"] = log_x,
                              Rcpp::_["corr"] = R_nk,
                              Rcpp::_["theta"] = model.theta_nk.mean(),
                              Rcpp::_["log.theta"] = model.theta_nk.log_mean(),
                              Rcpp::_["log.phi"] = model.logPhi_dk,
                              Rcpp::_["log.rho"] = model.logRho_nk,
                              Rcpp::_["phi"] = model.phi_dk,
                              Rcpp::_["rho"] = model.rho_nk);
}
