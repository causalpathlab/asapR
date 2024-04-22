#include "rcpp_asap.hh"
#include "rcpp_util.hh"
#include "rcpp_asap_pmf_train.hh"

using RNG = dqrng::xoshiro256plus;
using model_t = asap_pmf_model_t<RNG>;

//' A quick PMF estimation based on alternating Poisson regressions
//'
//' @param Y_ non-negative data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param max_iter max number of optimization steps
//' @param min_iter min number of optimization steps
//' @param burnin number of initiation steps (default: 50)
//' @param verbose verbosity
//' @param a0 gamma(a0, b0) default: a0 = 1
//' @param b0 gamma(a0, b0) default: b0 = 1
//' @param do_scale scale each column by standard deviation (default: TRUE)
//' @param do_log1p do log(1+y) transformation
//' @param rseed random seed (default: 1337)
//' @param svd_init initialize by SVD (default: FALSE)
//' @param EPS (default: 1e-8)
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
             const Rcpp::Nullable<Rcpp::List> r_A_dd_list = R_NilValue,
             const Rcpp::Nullable<Rcpp::List> r_A_nn_list = R_NilValue,
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

    log1p_op<Mat> log1p;
    Mat Y_dn = do_log1p ? Y_.unaryExpr(log1p) : Y_;

    TLOG_(verbose, "Data: " << Y_dn.rows() << " x " << Y_dn.cols());

    model_t model(model_t::ROW(Y_dn.rows()),
                  model_t::COL(Y_dn.cols()),
                  model_t::FACT(maxK),
                  model_t::RSEED(rseed),
                  model_t::A0(a0),
                  model_t::B0(b0));

    std::vector<Scalar> llik_trace;
    model_t::NULL_DATA null_data;

    train_pmf_options_t options;
    options.max_iter = max_iter;
    options.burnin = burnin;
    options.eps = EPS;
    options.verbose = verbose;
    options.svd_init = svd_init;

    SpMat A_dd, A_nn;

    if (r_A_dd_list.isNotNull()) {
        rcpp::util::build_sparse_mat(Rcpp::List(r_A_dd_list),
                                     Y_dn.rows(),
                                     Y_dn.rows(),
                                     A_dd);
        TLOG_(verbose, "Row Network: " << A_dd.rows() << " x " << A_dd.cols());
    }

    if (r_A_nn_list.isNotNull()) {
        rcpp::util::build_sparse_mat<SpMat>(Rcpp::List(r_A_nn_list),
                                            Y_dn.cols(),
                                            Y_dn.cols(),
                                            A_nn);
        TLOG_(verbose, "Col Network: " << A_nn.rows() << " x " << A_nn.cols());
    }

    if (A_dd.nonZeros() > 0 && A_nn.nonZeros() > 0) {
        train_pmf(model, Y_dn, A_dd, A_nn, llik_trace, options);
    } else if (A_dd.nonZeros() > 0 && A_nn.nonZeros() == 0) {
        train_pmf(model, Y_dn, A_dd, null_data, llik_trace, options);
    } else if (A_dd.nonZeros() == 0 && A_nn.nonZeros() > 0) {
        train_pmf(model, Y_dn, null_data, A_nn, llik_trace, options);
    } else {
        train_pmf(model, Y_dn, null_data, null_data, llik_trace, options);
    }

    // Mat std_log_x, R_nk;
    // std::tie(std_log_x, R_nk) = model.log_topic_correlation(Y_dn);

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["beta"] = model.beta_dk.mean(),
                              Rcpp::_["log.beta"] = model.beta_dk.log_mean(),
                              Rcpp::_["log.beta.sd"] = model.beta_dk.log_sd(),
                              Rcpp::_["theta"] = model.theta_nk.mean(),
                              Rcpp::_["log.theta.sd"] = model.theta_nk.log_sd(),
                              Rcpp::_["log.theta"] = model.theta_nk.log_mean());
}
