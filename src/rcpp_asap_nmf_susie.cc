#include "rcpp_asap.hh"

//' A quick NMF estimation based on SuSiE-like Poisson regression
//'
//' @param Y data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param max_iter number of variation Expectation Maximization steps
//' @param burnin number of burn-in iterations to apply re-scaling
//' @param verbose verbosity
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param rseed random seed
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_nmf_susie(const Eigen::MatrixXf Y_dn,
                   const std::size_t maxK,
                   const std::size_t max_iter = 100,
                   const std::size_t burnin = 10,
                   const bool verbose = true,
                   const double a0 = 1.,
                   const double b0 = 1.,
                   const std::size_t rseed = 42)
{

    const Index D = Y_dn.rows();
    const Index N = Y_dn.cols();
    const Index K = std::min(static_cast<Index>(maxK), N);

    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    using gamma_t = gamma_param_t<Mat, RNG>;

    gamma_t theta_nk(N, K, a0, b0, rng); // scaling for all the factor loading
    gamma_t alpha_k(1, K, a0, b0, rng);  // scaling for the dictionary
    Mat logRho_nk(N, K), rho_nk(N, K);   // column to topic latent assignment
    Mat logitPi_dk(D, K);                // topic-specific row/feature selection
    Mat logscale_pi_dk(D, K);            //

    const ColVec Y_n = Y_dn.colwise().sum().transpose();
    const ColVec Y_d = Y_dn.transpose().colwise().sum();
    const ColVec ones_n = ColVec::Ones(N);
    const RowVec ones_d = RowVec::Ones(D);

    using norm_dist_t = boost::random::normal_distribution<Scalar>;
    norm_dist_t norm_dist(0., 1.);
    auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };
    auto exp_op = [](const Scalar x) -> Scalar { return fasterexp(x); };

    softmax_op_t<Mat> softmax;

    ////////////////////
    // Initialization //
    ////////////////////

    // Initialize the pi matrix ~ prob(d, k)
    logitPi_dk = Mat::NullaryExpr(D, K, rnorm);
    for (Index kk = 0; kk < K; ++kk) {
        logscale_pi_dk.col(kk) = softmax.log_col(logitPi_dk.col(kk));
    }

    // Column: initialization of the rho matrix ~ prob(n,k)
    logRho_nk = Mat::NullaryExpr(N, K, rnorm);
    for (Index jj = 0; jj < N; ++jj) {
        rho_nk.row(jj) = softmax.apply_row(logRho_nk.row(jj));
    }

    // Column: update theta_k
    for (Index kk = 0; kk < K; ++kk) {
        theta_nk.update_col(rho_nk.col(kk).cwiseProduct(Y_n), ones_n, kk);
    }
    theta_nk.calibrate();

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter);

    Mat beta_dk(D, K);
    RowVec tempK(K);

    for (Index tt = 0; tt < (max_iter + burnin); ++tt) {

        ////////////////////////
        // row factor updates //
        ////////////////////////

        // Row: update dictionary
        logitPi_dk = Y_dn * (rho_nk.cwiseProduct(theta_nk.log_mean()));
        logitPi_dk.array().colwise() /= Y_d.array();

        // For each factor k, centre and re-scale
        tempK = logitPi_dk.colwise().mean();
        logitPi_dk.array().rowwise() -= tempK.array();

        if (tt < burnin) {
            tempK = logitPi_dk.cwiseProduct(logitPi_dk)
                        .colwise()
                        .mean()
                        .cwiseSqrt();
            tempK.array() += 1e-4;
            logitPi_dk.array().rowwise() /= tempK.array();
        }

        for (Index kk = 0; kk < K; ++kk) {
            logscale_pi_dk.col(kk) = softmax.log_col(logitPi_dk.col(kk));
        }

        // Row: update factor loading alpha
        for (Index kk = 0; kk < K; ++kk) {
            Scalar zz = logscale_pi_dk.col(kk).mean();
            Scalar exp_zz = fasterexp(zz);
            alpha_k.update_col((logscale_pi_dk.array().col(kk) - zz)
                                       .matrix()
                                       .unaryExpr(exp_op)
                                       .transpose() *
                                   Y_dn * rho_nk.col(kk) * exp_zz,
                               rho_nk.col(kk).transpose() * ones_n,
                               kk);
        }
        alpha_k.calibrate();

        beta_dk = (logscale_pi_dk.array().rowwise() +
                   alpha_k.log_mean().row(0).array())
                      .matrix()
                      .unaryExpr(exp_op);

        ///////////////////////////
        // column factor updates //
        ///////////////////////////

        // Column: Update rho_nk
        logRho_nk = Y_dn.transpose() * logitPi_dk;
        logRho_nk.array().colwise() /= Y_n.array();
        logRho_nk += theta_nk.log_mean();

        for (Index jj = 0; jj < N; ++jj) {
            rho_nk.row(jj) = softmax.apply_row(logRho_nk.row(jj));
        }

        // Column: update theta_k
        for (Index kk = 0; kk < K; ++kk) {
            Scalar beta_dk_sum = alpha_k.mean().coeff(0, kk);     //
            theta_nk.update_col(rho_nk.col(kk).cwiseProduct(Y_n), //
                                ones_n * beta_dk_sum,             //
                                kk);                              //
        }
        theta_nk.calibrate();

        // evaluate log-likelihood
        Scalar llik =
            (Y_dn.transpose() * logitPi_dk).cwiseProduct(rho_nk).sum();
        llik += (Y_dn * (rho_nk.cwiseProduct(theta_nk.log_mean()))).sum();
        llik -= (ones_d * beta_dk * theta_nk.mean().transpose() * ones_n).sum();
        llik_trace.emplace_back(llik);

        if (verbose) {
            TLOG("SuSiE NMF [ " << tt << " ] " << llik);
        } else {
            Rcpp::Rcerr << "+ " << std::flush;
        }
        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            WLOG("Interruption by a user at t=" << tt);
            break;
        }
    }

    Rcpp::Rcerr << std::endl;
    TLOG("Done");

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["log.dict.scale"] = alpha_k.log_mean(),
                              Rcpp::_["log.dict.normalized"] = logscale_pi_dk,
                              Rcpp::_["log.dict"] = logitPi_dk,
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.rho"] = logRho_nk);
}
