#include "rcpp_asap.hh"

//' A quick NMF estimation based on alternating Poisson regressions
//'
//' @param Y_dn non-negative data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param max_iter max number of optimization steps
//' @param min_iter min number of optimization steps
//' @param burnin number of optimization steps w/o scaling
//' @param verbose verbosity
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param rseed random seed
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
asap_fit_nmf_alternate(const Eigen::MatrixXf Y_dn,
                       const std::size_t maxK,
                       const std::size_t max_iter = 100,
                       const std::size_t min_iter = 5,
                       const std::size_t burnin = 100,
                       const bool verbose = true,
                       const double a0 = 1e-4,
                       const double b0 = 1e-4,
                       const std::size_t rseed = 42,
                       const double EPS = 1e-6,
                       const double rate_m = 1,
                       const double rate_v = 1,
                       const bool init_stoch = true)
{

    const Index D = Y_dn.rows();
    const Index N = Y_dn.cols();
    const Index K = std::min(static_cast<Index>(maxK), N);

    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    using gamma_t = gamma_param_t<Mat, RNG>;

    gamma_t beta_dk(D, K, a0, b0, rng);  // dictionary
    gamma_t theta_nk(N, K, a0, b0, rng); // scaling for all the factor loading

    Mat logBeta_dk(D, K);              // row topic
    Mat logPhi_dk(D, K), phi_dk(D, K); // row to topic latent assignment
    Mat logTheta_nk(N, K);             // column loading
    Mat logRho_nk(N, K), rho_nk(N, K); // column to topic latent assignment

    using norm_dist_t = boost::random::normal_distribution<Scalar>;
    norm_dist_t norm_dist(0., 1.);

    // auto exp_op = [](const Scalar x) -> Scalar { return fastexp(x); };
    auto at_least_one = [](const Scalar x) -> Scalar {
        return (x < 1.) ? 1. : x;
    };

    const ColVec Y_n = Y_dn.colwise().sum().transpose();
    const ColVec Y_d = Y_dn.transpose().colwise().sum();
    const ColVec Y_n1 = Y_n.unaryExpr(at_least_one);
    const ColVec Y_d1 = Y_d.unaryExpr(at_least_one);
    const ColVec ones_n = ColVec::Ones(N);
    const ColVec ones_d = ColVec::Ones(D);

    softmax_op_t<Mat> softmax;

    ////////////////////
    // Initialization //
    ////////////////////

    auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };

    logPhi_dk = Mat::NullaryExpr(D, K, rnorm);
    for (Index ii = 0; ii < D; ++ii) {
        phi_dk.row(ii) = softmax.apply_row(logPhi_dk.row(ii));
    }

    logRho_nk = Mat::NullaryExpr(N, K, rnorm);
    for (Index jj = 0; jj < N; ++jj) {
        rho_nk.row(jj) = softmax.apply_row(logRho_nk.row(jj));
    }

    Mat temp_nk(N, K), temp_dk(D, K);
    stdizer_t<Mat> std_ln_phi_dk(logPhi_dk, rate_m, rate_v);
    stdizer_t<Mat> std_ln_rho_nk(logRho_nk, rate_m, rate_v);

    std_ln_rho_nk.colwise(EPS);
    std_ln_phi_dk.colwise(EPS);

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter + burnin);

    auto calc_log_lik = [&]() {
        Scalar llik = (phi_dk.cwiseProduct(beta_dk.log_mean()).transpose() *
                       Y_dn * rho_nk)
                          .sum();

        llik += (rho_nk.cwiseProduct(theta_nk.log_mean()).transpose() *
                 Y_dn.transpose() * phi_dk)
                    .sum();

        llik -= (ones_d.transpose() * beta_dk.mean() *
                 theta_nk.mean().transpose() * ones_n)
                    .sum();
        return llik;
    };

    auto update_theta = [&]() {
        temp_nk.setZero();
        temp_nk.array().rowwise() += beta_dk.mean().colwise().sum().array();
        theta_nk.update(Y_dn.transpose() * phi_dk, temp_nk);
        theta_nk.calibrate();
    };

    auto update_beta = [&]() {
        temp_dk.setZero();
        temp_dk.array().rowwise() += theta_nk.mean().colwise().sum().array();
        beta_dk.update(Y_dn * rho_nk, temp_dk);
        beta_dk.calibrate();
    };

    rowvec_sampler_t<Mat, RNG> sampler(rng, K);

    TLOG("Initialization of auxiliary variables");
    for (Index tt = 0; tt < burnin; ++tt) {
        logPhi_dk = Y_dn * std_ln_rho_nk.colwise(EPS);
        logPhi_dk.array().colwise() /= Y_d1.array();

        for (Index ii = 0; ii < D; ++ii) {
            phi_dk.row(ii) = softmax.apply_row(logPhi_dk.row(ii));
            if (init_stoch) {
                const Index k = sampler(phi_dk.row(ii));
                phi_dk.row(ii).setZero();
                phi_dk(ii, k) = 1.;
            }
        }

        update_theta();

        logRho_nk = Y_dn.transpose() * std_ln_phi_dk.colwise(EPS);
        logRho_nk.array().colwise() /= Y_n1.array();

        for (Index jj = 0; jj < N; ++jj) {
            rho_nk.row(jj) = softmax.apply_row(logRho_nk.row(jj));
            if (init_stoch) {
                const Index k = sampler(rho_nk.row(jj));
                rho_nk.row(jj).setZero();
                rho_nk(jj, k) = 1.;
            }
        }

        update_beta();

        // evaluate log-likelihood
        Scalar llik = calc_log_lik();
        llik_trace.emplace_back(llik);
        if (verbose) {
            TLOG("Burn-in the regressors [ " << tt << " ] " << llik);
        } else {
            Rcpp::Rcerr << "+ " << std::flush;
            if (tt > 0 && tt % 10 == 0) {
                Rcpp::Rcerr << "\r" << std::flush;
            }
        }

        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            WLOG("Interruption by the user at t=" << tt);
            break;
        }
    }
    Rcpp::Rcerr << "\r" << std::flush;
    TLOG("Finished burn-in iterations");

    logBeta_dk = beta_dk.log_mean();
    logTheta_nk = theta_nk.log_mean();

    stdizer_t<Mat> std_ln_beta_dk(logBeta_dk, rate_m, rate_v);
    stdizer_t<Mat> std_ln_theta_nk(logTheta_nk, rate_m, rate_v);

    RowVec tempK(K);

    for (Index tt = 0; tt < max_iter; ++tt) {

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (i,k)  //
        //////////////////////////////////////////////

        logPhi_dk = Y_dn * std_ln_theta_nk.colwise(EPS);
        logPhi_dk.array().colwise() /= Y_d1.array();
        logPhi_dk += logBeta_dk;

        std_ln_phi_dk.colwise(EPS);

        for (Index ii = 0; ii < D; ++ii) {
            tempK = logPhi_dk.row(ii);
            logPhi_dk.row(ii) = softmax.log_row(tempK);
        }
        phi_dk = logPhi_dk.array().exp();

        update_beta();
        logBeta_dk = beta_dk.log_mean();

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (j,k)  //
        //////////////////////////////////////////////

        logRho_nk = Y_dn.transpose() * std_ln_beta_dk.colwise(EPS);
        logRho_nk.array().colwise() /= Y_n1.array();
        logRho_nk += logTheta_nk;

        std_ln_rho_nk.colwise(EPS);

        for (Index jj = 0; jj < N; ++jj) {
            tempK = logRho_nk.row(jj);
            logRho_nk.row(jj) = softmax.log_row(tempK);
        }
        rho_nk = logRho_nk.array().exp();

        update_theta();
        logTheta_nk = theta_nk.log_mean();

        Scalar llik = calc_log_lik(); // evaluate log-likelihood

        const Scalar diff = llik_trace.size() > 0 ?
            std::abs(llik - llik_trace.at(llik_trace.size() - 1)) /
                std::abs(llik + EPS) :
            llik;

        llik_trace.emplace_back(llik);

        if (verbose) {
            TLOG("NMF by regressors [ " << tt << " ] " << llik << ", " << diff);
        } else {
            Rcpp::Rcerr << "+ " << std::flush;
            if (tt > 0 && tt % 10 == 0) {
                Rcpp::Rcerr << "\r" << std::flush;
            }
        }

        if (tt > min_iter && diff < EPS) {
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

    Mat log_x = beta_dk.log_mean();
    stdizer_t<Mat> std_ln_x(log_x, 1, 1);
    std_ln_x.colwise(EPS);

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["beta"] = beta_dk.mean(),
                              Rcpp::_["log.beta"] = beta_dk.log_mean(),
                              Rcpp::_["log_x"] = log_x,
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.phi"] = logPhi_dk,
                              Rcpp::_["log.rho"] = logRho_nk,
                              Rcpp::_["phi"] = phi_dk,
                              Rcpp::_["rho"] = rho_nk);
}
