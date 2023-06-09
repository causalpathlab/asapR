#include "rcpp_asap.hh"
#include "rcpp_asap_nmf_train.hh"

using RNG = dqrng::xoshiro256plus;
using model_t = asap_nmf_model_t<RNG>;

//' A quick NMF estimation based on alternating Poisson regressions
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
asap_fit_nmf(const Eigen::MatrixXf &Y_,
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
             const std::size_t NUM_THREADS = 1)
{
    Eigen::setNbThreads(NUM_THREADS);
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

    train_nmf_options_t options;
    options.max_iter = max_iter;
    options.burnin = burnin;
    options.eps = EPS;
    options.verbose = verbose;
    options.svd_init = svd_init;

    train_nmf(model, Y_dn, null_data, llik_trace, options);

    Mat log_x, R_nk;
    std::tie(log_x, R_nk) = model.log_topic_correlation(Y_dn);

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["log_x"] = log_x,
                              Rcpp::_["corr"] = R_nk,
                              Rcpp::_["model"] = rcpp_list_out(model));
}

//' Estimate NMF dictionary with some adjacency matrix (gene x gene)
//'
//' @param Y_ non-negative data matrix (gene x sample)
//' @param A_ sparse adjacency matrix gene x gene
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
//'   \item corr correlation matrix (sample x factor)
//'   \item model$beta dictionary (gene x factor)
//'   \item model$log.beta log-dictionary (gene x factor)
//'   \item model$log.beta.sd sd(log-dictionary) (gene x factor)
//'   \item model$theta loading (sample x factor)
//'   \item model$log.theta log-loading (sample x factor)
//'   \item model$log.theta sd(log-loading) (sample x factor)
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_nmf_network(const Eigen::MatrixXf &Y_,
                     const Eigen::SparseMatrix<float, Eigen::ColMajor> &A_dd,
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
                     const std::size_t NUM_THREADS = 1)
{
    Eigen::setNbThreads(NUM_THREADS);
    TLOG_(verbose, Eigen::nbThreads() << " threads");

    log1p_op<Mat> log1p;
    Mat Y_dn = do_log1p ? Y_.unaryExpr(log1p) : Y_;

    TLOG_(verbose, "Data: " << Y_dn.rows() << " x " << Y_dn.cols());
    TLOG_(verbose, "Network: " << A_dd.rows() << " x " << A_dd.cols());
    ASSERT_RETL(A_dd.rows() == Y_dn.rows() && A_dd.cols() == Y_dn.rows(),
                "rows(A) or cols(A) don't match with row(Y_dn)");

    model_t model(model_t::ROW(Y_dn.rows()),
                  model_t::COL(Y_dn.cols()),
                  model_t::FACT(maxK),
                  model_t::RSEED(rseed),
                  model_t::A0(a0),
                  model_t::B0(b0));

    train_nmf_options_t options;
    options.max_iter = max_iter;
    options.burnin = burnin;
    options.eps = EPS;
    options.verbose = verbose;
    options.svd_init = svd_init;

    std::vector<Scalar> llik_trace;
    train_nmf(model, Y_dn, A_dd, llik_trace, options);

    Mat log_x, R_nk;
    std::tie(log_x, R_nk) = model.log_topic_correlation(Y_dn);

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["log_x"] = log_x,
                              Rcpp::_["corr"] = R_nk,
                              Rcpp::_["model"] = rcpp_list_out(model));
}