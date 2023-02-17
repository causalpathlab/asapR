#include "rcpp_asap.hh"

//' Poisson regression to estimate factor loading
//'
//' @param Y data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param mcem number of Monte Carl Expectation Maximization
//' @param burnin burn-in period
//' @param latent_iter latent sampling steps
//' @param thining thining interval in record keeping
//' @param verbose verbosity
//' @param eval_llik evaluate log-likelihood
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param rseed random seed
//' @param NUM_THREADS number of parallel jobs
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_nmf_susie(const Eigen::MatrixXf Y,
                   const std::size_t maxK,
                   const std::size_t mcem = 100,
                   const std::size_t burnin = 10,
                   const std::size_t latent_iter = 10,
                   const std::size_t thining = 3,
                   const bool verbose = true,
                   const bool eval_llik = true,
                   const double a0 = 1.,
                   const double b0 = 1.,
                   const std::size_t rseed = 42,
                   const std::size_t NUM_THREADS = 1)
{

    const Index D = Y.rows();
    const Index N = Y.cols();
    const Index K = std::min(static_cast<Index>(maxK), N);

    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    using gamma_t = gamma_param_t<Mat, RNG>;
    rowvec_sampler_t<Mat, RNG> sampler(rng, K);

    gamma_t beta_k(1, K, a0, b0, rng), theta_k(1, K, a0, b0, rng);
    Mat logC_dk(D, K), C_dk(D, K), X_dk(D, K);
    Mat logZ_nk(N, K), Z_nk(N, K), X_nk(N, K);

    // random initialization Z and C matrices
    softmax_op_t<Mat> softmax;
    for (Index ii = 0; ii < D; ++ii) {
        logC_dk.row(ii).setZero();
        C_dk.row(ii).setZero();
        Index k = sampler(softmax(logC_dk.row(ii)));
        C_dk(ii, k) = 1.;
    }

    for (Index jj = 0; jj < N; ++jj) {
        logZ_nk.row(jj).setZero();
        Z_nk.row(jj).setZero();
        Index k = sampler(softmax(logZ_nk.row(jj)));
        Z_nk(jj, k) = 1.;
    }

    RowVec onesN(N), onesD(D);
    onesN.setOnes();
    onesD.setOnes();

    ColVec y_d = Y.rowwise().sum();
    ColVec y_n = Y.transpose().rowwise().sum();

    // update beta_k
    beta_k.update(onesN * Y.transpose() * C_dk.transpose(),
                  (onesD * C_dk)
                      .cwiseProduct(theta_k.mean())
                      .cwiseProduct(onesN * Z_nk));
    beta_k.calibrate();

    // update theta_k
    theta_k.update(onesD * Y * Z_nk,
                   (onesN * Z_nk)
                       .cwiseProduct(beta_k.mean())
                       .cwiseProduct(onesD * C_dk));
    theta_k.calibrate();

    for (Index tt = 0; tt < (mcem + burnin); ++tt) {

        // sample or optimize C
        X_nk = standardize(Z_nk);
        logC_dk =
            (Y * X_nk).array().rowwise() * theta_k.log_mean().row(0).array();
        logC_dk.array().colwise() /= y_d.array();
        logC_dk.array().rowwise() += beta_k.log_mean().row(0).array();

        for (Index ii = 0; ii < D; ++ii) {
            C_dk.row(ii) = softmax(logC_dk.row(ii));
        }

        // update beta_k
        beta_k.update(onesN * Y.transpose() * C_dk.transpose(),
                      theta_k.mean().cwiseProduct(onesN * Z_nk));
        beta_k.calibrate();

        // sample or optimize Z
        X_dk = standardize(C_dk);
        logZ_nk = (Y.transpose() * X_dk).array().rowwise() *
            beta_k.log_mean().row(0).array();
        logZ_nk.array().colwise() /= y_n.array();
        logZ_nk.array().rowwise() += theta_k.log_mean().row(0).array();

        for (Index jj = 0; jj < N; ++jj) {
            Z_nk.row(jj) = softmax(logZ_nk.row(jj));
        }

        // update theta_k
        theta_k.update(onesD * Y * Z_nk,
                       beta_k.mean().cwiseProduct(onesD * C_dk));
        theta_k.calibrate();

        // evaluate log-likelihood
        Rcpp::Rcerr << "+ " << std::flush;
    }

    Rcpp::Rcerr << std::endl;
    TLOG("Done");

    // TODO

    return Rcpp::List::create(Rcpp::_["C"] = C_dk,
                              Rcpp::_["Z"] = Z_nk,
                              Rcpp::_["beta"] = beta_k.mean(),
                              Rcpp::_["theta"] = theta_k.mean());
}
