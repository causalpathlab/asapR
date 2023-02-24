#include "rcpp_asap.hh"

//' A quick NMF estimation based on alternating Poisson regressions
//'
//' @param Y_dn non-negative data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param max_iter number of variation Expectation Maximization steps
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
                       const std::size_t burnin = 10,
                       const bool verbose = true,
                       const double a0 = 1.,
                       const double b0 = 1.,
                       const std::size_t rseed = 42,
                       const double EPS = 1e-4)
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

    Mat logPhi_dk(D, K), phi_dk(D, K); // row to topic latent assignment
    Mat logRho_nk(N, K), rho_nk(N, K); // column to topic latent assignment

    using norm_dist_t = boost::random::normal_distribution<Scalar>;
    norm_dist_t norm_dist(0., 1.);
    auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };
    auto exp_op = [](const Scalar x) -> Scalar { return fasterexp(x); };
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

    logPhi_dk = Mat::NullaryExpr(D, K, rnorm);
    for (Index ii = 0; ii < D; ++ii) {
        phi_dk.row(ii) = softmax.apply_row(logPhi_dk.row(ii));
    }

    logRho_nk = Mat::NullaryExpr(N, K, rnorm);
    for (Index jj = 0; jj < N; ++jj) {
        rho_nk.row(jj) = softmax.apply_row(logRho_nk.row(jj));
    }

    Mat X_nk(N, K), X_dk(D, K);

    TLOG("Initialization of auxiliary variables");
    for (Index tt = 0; tt < burnin; ++tt) {
        X_nk = standardize(logRho_nk, EPS);
        logPhi_dk = Y_dn * X_nk;
        logPhi_dk.array().colwise() /= Y_d1.array();

        X_dk = standardize(logPhi_dk, EPS);
        logRho_nk = Y_dn.transpose() * X_dk;
        logRho_nk.array().colwise() /= Y_n1.array();
        for (Index ii = 0; ii < D; ++ii) {
            phi_dk.row(ii) = softmax.apply_row(logPhi_dk.row(ii));
        }

        for (Index jj = 0; jj < N; ++jj) {
            rho_nk.row(jj) = softmax.apply_row(logRho_nk.row(jj));
        }
        Rcpp::Rcerr << "+ " << std::flush;
        if (tt > 0 && tt % 10 == 0) {
            Rcpp::Rcerr << "\r" << std::flush;
        }
    }
    Rcpp::Rcerr << "\r" << std::flush;
    TLOG("Finished burn-in iterations");

    {
        // Column: update theta_k
        theta_nk.update(Y_dn.transpose() * phi_dk,                //
                        ones_n * beta_dk.mean().colwise().sum()); //
        theta_nk.calibrate();

        // Update row topic factors
        beta_dk.update(Y_dn * rho_nk,                             //
                       ones_d * theta_nk.mean().colwise().sum()); //
        beta_dk.calibrate();
    }

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter);

    RowVec tempK(K);

    for (Index tt = 0; tt < max_iter; ++tt) {

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (i,k)  //
        //////////////////////////////////////////////

        X_nk = standardize(theta_nk.log_mean(), EPS);
        logPhi_dk = Y_dn * X_nk;
        logPhi_dk.array().colwise() /= Y_d1.array();
        logPhi_dk += beta_dk.log_mean();

        for (Index ii = 0; ii < D; ++ii) {
            tempK = logPhi_dk.row(ii);
            logPhi_dk.row(ii) = softmax.log_row(tempK);
        }
        phi_dk = logPhi_dk.unaryExpr(exp_op);

        // Update column topic factors, theta(j, k)
        theta_nk.update(rho_nk.cwiseProduct(Y_dn.transpose() * phi_dk), //
                        ones_n * beta_dk.mean().colwise().sum());       //
        theta_nk.calibrate();

        // Update row topic factors
        beta_dk.update((phi_dk.array().colwise() * Y_d.array()).matrix(), //
                       ones_d * theta_nk.mean().colwise().sum());         //
        beta_dk.calibrate();

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (j,k)  //
        //////////////////////////////////////////////

        X_dk = standardize(beta_dk.log_mean(), EPS);
        logRho_nk = Y_dn.transpose() * X_dk;
        logRho_nk.array().colwise() /= Y_n1.array();
        logRho_nk += theta_nk.log_mean();

        for (Index jj = 0; jj < N; ++jj) {
            tempK = logRho_nk.row(jj);
            logRho_nk.row(jj) = softmax.log_row(tempK);
        }
        rho_nk = logRho_nk.unaryExpr(exp_op);

        // Update row topic factors
        beta_dk.update(phi_dk.cwiseProduct(Y_dn * rho_nk),        //
                       ones_d * theta_nk.mean().colwise().sum()); //
        beta_dk.calibrate();

        // Update column topic factors
        theta_nk.update((rho_nk.array().colwise() * Y_n.array()).matrix(), //
                        ones_n * beta_dk.mean().colwise().sum());          //
        theta_nk.calibrate();

        // evaluate log-likelihood
        Scalar llik = (phi_dk.cwiseProduct(beta_dk.log_mean()).transpose() *
                       Y_dn * rho_nk)
                          .sum();
        llik += (rho_nk.cwiseProduct(theta_nk.log_mean()) * Y_dn.transpose() *
                 phi_dk)
                    .sum();
        llik -= (ones_d.transpose() * beta_dk.mean() *
                 theta_nk.mean().transpose() * ones_n)
                    .sum();

        llik_trace.emplace_back(llik);

        const Scalar diff =
            tt > 0 ? abs(llik_trace.at(tt - 1) - llik) / abs(llik + EPS) : 0;

        if (verbose) {
            TLOG("NMF by regressors [ " << tt << " ] " << llik << ", "
                                         << diff);
        } else {
            Rcpp::Rcerr << "+ " << std::flush;
            if (tt > 0 && tt % 10 == 0) {
                Rcpp::Rcerr << "\r" << std::flush;
            }
        }

        if (tt > 0 && diff < EPS) {
            Rcpp::Rcerr << "\r" << std::endl;
            TLOG("Converged at " << tt << ", " << diff);
            break;
        }

        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            WLOG("Interruption by a user at t=" << tt);
            break;
        }
    }
    Rcpp::Rcerr << "\r" << std::endl;
    TLOG("Done");

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["beta"] = beta_dk.mean(),
                              Rcpp::_["log.beta"] = beta_dk.log_mean(),
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.phi"] = logPhi_dk,
                              Rcpp::_["log.rho"] = logRho_nk);
}
