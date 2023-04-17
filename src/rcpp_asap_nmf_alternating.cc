#include "rcpp_asap.hh"

//' A quick NMF estimation based on alternating Poisson regressions
//'
//' @param Y_dn non-negative data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param max_iter max number of optimization steps
//' @param min_iter min number of optimization steps
//' @param burnin number of initiation steps
//' @param verbose verbosity
//' @param a0 gamma(a0, b0) default: a0 = 1
//' @param b0 gamma(a0, b0) default: b0 = 1
//' @param rseed random seed (default: 1337)
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
                       const double a0 = 1,
                       const double b0 = 1,
                       const std::size_t rseed = 1337,
                       const double EPS = 1e-6,
                       const double rate_m = 1,
                       const double rate_v = 1,
                       const bool svd_init = true)
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

    exp_op<Mat> exp;
    log1p_op<Mat> log1p;
    at_least_one_op<Mat> at_least_one;

    const ColVec Y_n = Y_dn.colwise().sum().transpose();
    const ColVec Y_d = Y_dn.transpose().colwise().sum();
    const ColVec Y_n1 = Y_n.unaryExpr(at_least_one);
    const ColVec Y_d1 = Y_d.unaryExpr(at_least_one);
    const ColVec ones_n = ColVec::Ones(N);
    const ColVec ones_d = ColVec::Ones(D);

    softmax_op_t<Mat> softmax;

    rowvec_sampler_t<Mat, RNG> sampler(rng, K);

    ////////////////////
    // Initialization //
    ////////////////////

    if (svd_init) {
        const std::size_t lu_iter = 5;      // this should be good
        RandomizedSVD<Mat> svd(K, lu_iter); //
        const Mat yy = standardize(Y_dn.unaryExpr(log1p));
        svd.compute(yy);

        {
            Mat a = standardize(svd.matrixU()).unaryExpr(exp);
            a += beta_dk.sample() / static_cast<Scalar>(D);
            Mat b = Mat::Ones(D, K);
            beta_dk.update(a, b);
        }
        {
            Mat a = standardize(svd.matrixV()).unaryExpr(exp);
            a += theta_nk.sample() / static_cast<Scalar>(N);
            Mat b = Mat::Ones(N, K);
            theta_nk.update(a, b);
        }

    } else {

        {
            Mat a = beta_dk.sample();
            Mat b = Mat::Ones(D, K);
            beta_dk.update(a / static_cast<Scalar>(D), b);
        }
        {
            Mat a = theta_nk.sample();
            Mat b = Mat::Ones(N, K);
            theta_nk.update(a / static_cast<Scalar>(N), b);
        }
    }

    /////////////////////////
    // auxiliary variables //
    /////////////////////////

    logPhi_dk = Mat::Random(D, K);
    for (Index ii = 0; ii < D; ++ii) {
        phi_dk.row(ii) = softmax.apply_row(logPhi_dk.row(ii));
    }

    logRho_nk = Mat::Random(N, K);
    for (Index jj = 0; jj < N; ++jj) {
        rho_nk.row(jj) = softmax.apply_row(logRho_nk.row(jj));
    }

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter + burnin);

    stdizer_t<Mat> std_ln_rho_nk(logRho_nk, rate_m, rate_v);
    stdizer_t<Mat> std_ln_phi_dk(logPhi_dk, rate_m, rate_v);

    ////////////////////////////////
    // log-likelihood computation //
    ////////////////////////////////

    auto calc_log_lik = [&]() {
        Scalar llik = 0;
        Scalar denom = N * D;

        llik +=
            (phi_dk.cwiseProduct(beta_dk.log_mean()).transpose() * Y_dn).sum() /
            denom;

        llik += (phi_dk.cwiseProduct(Y_dn * theta_nk.log_mean())).sum() / denom;

        llik -=
            (phi_dk.cwiseProduct(logPhi_dk).transpose() * Y_dn).sum() / denom;

        llik -= (ones_d.transpose() * beta_dk.mean() *
                 theta_nk.mean().transpose() * ones_n)
                    .sum() /
            denom;

        return llik;
    };

    RowVec tempK(K);

    for (Index tt = 0; tt < (burnin + max_iter); ++tt) {

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (i,k)  //
        //////////////////////////////////////////////

        logPhi_dk = Y_dn * theta_nk.log_mean();
        logPhi_dk.array().colwise() /= Y_d1.array();
        logPhi_dk += beta_dk.log_mean();

        for (Index ii = 0; ii < D; ++ii) {
            tempK = logPhi_dk.row(ii);
            logPhi_dk.row(ii) = softmax.log_row(tempK);
        }

        if (tt <= burnin) {
            phi_dk.setZero();
            for (Index ii = 0; ii < D; ++ii) {
                Index k = sampler(logPhi_dk.row(ii).unaryExpr(exp));
                phi_dk(ii, k) = 1.;
            }
        } else {
            phi_dk = logPhi_dk.unaryExpr(exp);
        }

        ///////////////////////
        // update parameters //
        ///////////////////////

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

        logRho_nk = Y_dn.transpose() * beta_dk.log_mean();
        logRho_nk.array().colwise() /= Y_n1.array();
        logRho_nk += theta_nk.log_mean();

        ///////////////////////////////////
        // this helps spread the columns //
        ///////////////////////////////////

        std_ln_rho_nk.colwise(EPS);

        for (Index jj = 0; jj < N; ++jj) {
            tempK = logRho_nk.row(jj);
            logRho_nk.row(jj) = softmax.log_row(tempK);
        }

        rho_nk = logRho_nk.unaryExpr(exp);

        ///////////////////////
        // update parameters //
        ///////////////////////

        // Update row topic factors
        beta_dk.update(phi_dk.cwiseProduct(Y_dn * rho_nk),        //
                       ones_d * theta_nk.mean().colwise().sum()); //
        beta_dk.calibrate();

        // Update column topic factors
        theta_nk.update((rho_nk.array().colwise() * Y_n.array()).matrix(), //
                        ones_n * beta_dk.mean().colwise().sum());          //
        theta_nk.calibrate();

        //////////////////////////////
        // calculate log-likelihood //
        //////////////////////////////

        Scalar llik = calc_log_lik(); // evaluate log-likelihood

        const Scalar diff = llik_trace.size() > 0 ?
            std::abs(llik - llik_trace.at(llik_trace.size() - 1)) /
                std::abs(llik + EPS) :
            llik;

        if (tt >= burnin) {
            llik_trace.emplace_back(llik);
        }

        if (verbose) {
            TLOG("NMF by regressors [ " << tt << " ] " << llik << ", " << diff);
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

    Mat log_x = standardize(beta_dk.log_mean());
    Mat R_nk = (Y_dn.transpose() * log_x).array().colwise() / Y_n1.array();

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["beta"] = beta_dk.mean(),
                              Rcpp::_["log.beta"] = beta_dk.log_mean(),
                              Rcpp::_["log_x"] = log_x,
                              Rcpp::_["corr"] = R_nk,
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.phi"] = logPhi_dk,
                              Rcpp::_["log.rho"] = logRho_nk,
                              Rcpp::_["phi"] = phi_dk,
                              Rcpp::_["rho"] = rho_nk);
}
