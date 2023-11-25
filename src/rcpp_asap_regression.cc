#include "rcpp_asap_regression.hh"

//' Calibrate topic proportions based on sufficient statistics
//'
//' @param beta_dk dictionary matrix (feature D  x factor K)
//' @param R_nk correlation matrix (sample N x factor K)
//' @param Y_n sum vector (sample N x 1)
//' @param a0 gamma(a0, b0) (default: 1)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param max_iter maximum iterations (default: 10)
//' @param NUM_THREADS number of parallel threads (default: 1)
//' @param stdize_r standardize correlation matrix R (default: TRUE)
//' @param verbose (default: TRUE)
//'
//' @return a list that contains:
//' \itemize{
//'  \item beta (D x K) matrix
//'  \item theta (N x K) matrix
//'  \item log.theta (N x K) log matrix
//'  \item log.theta.sd (N x K) standard deviation matrix
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_topic_pmf(const Eigen::MatrixXf beta_dk,
               const Eigen::MatrixXf R_nk,
               const Eigen::MatrixXf Y_n,
               const double a0 = 1.0,
               const double b0 = 1.0,
               const std::size_t max_iter = 10,
               const std::size_t NUM_THREADS = 1,
               const bool stdize_r = true,
               const bool verbose = true)
{

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    exp_op<Mat> exp;
    softmax_op_t<Mat> softmax;

    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Mat, RNG>;
    RNG rng;

    Eigen::setNbThreads(NUM_THREADS);

    const Index D = beta_dk.rows();
    const Index K = beta_dk.cols();
    const Index N = R_nk.rows();

    ASSERT_RETL(Y_n.rows() == R_nk.rows(),
                "R and Y must have the same number of rows");

    Mat x_nk = ColVec::Ones(N) * beta_dk.colwise().sum(); // N x K

    Mat logRho_nk(N, K), rho_nk(N, K);
    gamma_t theta_nk(N, K, a0, b0, rng); // N x K

    RowVec tempK(K);

    Mat r_nk = R_nk;
    if (stdize_r) {
        standardize_columns_inplace(r_nk);
    }

    for (std::size_t t = 0; t < max_iter; ++t) {

        logRho_nk = r_nk + theta_nk.log_mean();

        for (Index jj = 0; jj < N; ++jj) {
            tempK = logRho_nk.row(jj);
            logRho_nk.row(jj) = softmax.log_row(tempK);
        }
        rho_nk = logRho_nk.unaryExpr(exp);

        theta_nk
            .update((rho_nk.array().colwise() * Y_n.col(0).array()).matrix(),
                    x_nk);
        theta_nk.calibrate();
    }

    return Rcpp::List::create(Rcpp::_["beta"] = beta_dk,
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.theta.sd"] = theta_nk.log_sd());
}

//' Topic statistics to estimate factor loading
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param row_file row names file (D x 1)
//' @param col_file column names file (N x 1)
//' @param idx_file matrix-market colum index file
//' @param log_beta D x K log dictionary/design matrix
//' @param x_row_names row names log_beta (D vector)
//' @param do_log1p do log(1+y) transformation
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param MAX_ROW_WORD maximum words per line in `row_files[i]`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @param MAX_COL_WORD maximum words per line in `col_files[i]`
//' @param COL_WORD_SEP word separation character to replace white space
//'
//' @return a list that contains:
//' \itemize{
//'  \item beta dictionary matrix (row x factor)
//'  \item corr empirical correlation (column x factor)
//'  \item colsum the sum of each column (column x 1)
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_topic_stat(const std::string mtx_file,
                const std::string row_file,
                const std::string col_file,
                const std::string idx_file,
                const Eigen::MatrixXf log_beta,
                const Rcpp::StringVector &x_row_names,
                const bool do_log1p = false,
                const bool verbose = false,
                const std::size_t NUM_THREADS = 1,
                const std::size_t BLOCK_SIZE = 100,
                const std::size_t MAX_ROW_WORD = 2,
                const char ROW_WORD_SEP = '_',
                const std::size_t MAX_COL_WORD = 100,
                const char COL_WORD_SEP = '@')
{

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    std::vector<std::string> pos2row;
    rcpp::util::copy(x_row_names, pos2row);

    topic_stat_options_t options;

    options.do_log1p = do_log1p;
    options.verbose = verbose;
    options.NUM_THREADS = NUM_THREADS;
    options.BLOCK_SIZE = BLOCK_SIZE;
    options.MAX_ROW_WORD = MAX_ROW_WORD;
    options.ROW_WORD_SEP = ROW_WORD_SEP;
    options.MAX_COL_WORD = MAX_COL_WORD;
    options.COL_WORD_SEP = COL_WORD_SEP;

    Mat Rtot_nk, Ytot_n;

    CHK_RETL_(asap_topic_stat_mtx(mtx_file,
                                  row_file,
                                  col_file,
                                  idx_file,
                                  log_beta,
                                  pos2row,
                                  options,
                                  Rtot_nk,
                                  Ytot_n),
              "unable to compute topic statistics");

    const Index N = Rtot_nk.rows(), K = Rtot_nk.cols();

    using namespace rcpp::util;
    using namespace Rcpp;

    std::vector<std::string> &d_ = pos2row;
    std::vector<std::string> k_;
    for (std::size_t k = 1; k <= K; ++k) {
        k_.push_back(std::to_string(k));
    }
    const std::vector<std::string> file_ { mtx_file };
    exp_op<Mat> exp;

    std::vector<std::string> coln;
    CHK_RETL_(read_line_file(col_file, coln, MAX_COL_WORD, COL_WORD_SEP),
              "Failed to read the column name file: " << col_file);

    ASSERT_RETL(N == coln.size(),
                "Different #columns: " << N << " vs. " << coln.size());

    return List::create(_["beta"] = named(log_beta.unaryExpr(exp), d_, k_),
                        _["corr"] = named(Rtot_nk, coln, k_),
                        _["colsum"] = named(Ytot_n, coln, file_),
                        _["rownames"] = pos2row,
                        _["colnames"] = coln);
}

//' Poisson regression to estimate factor loading
//'
//' @param Y D x N data matrix
//' @param log_beta D x K log dictionary/design matrix
//' @param a0 gamma(a0, b0) (default: 1e-8)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param do_log1p do log(1+y) transformation (default: FALSE)
//' @param verbose verbosity (default: false)
//'
//' @return a list that contains:
//' \itemize{
//'  \item beta (D x K) matrix
//'  \item theta (N x K) matrix
//'  \item log.theta (N x K) log matrix
//'  \item log.theta.sd (N x K) standard deviation matrix
//'  \item corr (N x K) topic correlation matrix
//'  \item colsum (N x 1) column sum vector
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_regression(const Eigen::MatrixXf Y_,
                const Eigen::MatrixXf log_beta,
                const double a0 = 1e-8,
                const double b0 = 1.0,
                const std::size_t max_iter = 10,
                const bool do_log1p = false,
                const bool verbose = true)
{

    exp_op<Mat> exp;
    at_least_one_op<Mat> at_least_one;
    softmax_op_t<Mat> softmax;
    log1p_op<Mat> log1p;

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    Mat Y_dn = do_log1p ? Y_.unaryExpr(log1p) : Y_;

    const Index D = Y_dn.rows();     // number of features
    const Index N = Y_dn.cols();     // number of samples
    const Index K = log_beta.cols(); // number of topics

    Mat logX_dk = log_beta;

    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Mat, RNG>;
    RNG rng;

    Mat logRho_nk(Y_dn.cols(), K), rho_nk(Y_dn.cols(), K);

    gamma_t theta_nk(Y_dn.cols(), K, a0, b0, rng); // N x K

    const ColVec Y_n = Y_dn.colwise().sum().transpose();
    const ColVec Y_n1 = Y_n.unaryExpr(at_least_one);

    // X[j,k] = sum_i X[i,k]
    RowVec Xsum = logX_dk.unaryExpr(exp).colwise().sum();
    Mat x_nk = ColVec::Ones(N) * Xsum; // n x K
    Mat R_nk = (Y_dn.transpose() * logX_dk).array().colwise() / Y_n1.array();

    if (verbose)
        TLOG("Correlation with the topics");

    RowVec tempK(K);

    for (std::size_t t = 0; t < max_iter; ++t) {

        logRho_nk = R_nk + theta_nk.log_mean();

        for (Index jj = 0; jj < Y_dn.cols(); ++jj) {
            tempK = logRho_nk.row(jj);
            logRho_nk.row(jj) = softmax.log_row(tempK);
        }
        rho_nk = logRho_nk.unaryExpr(exp);

        theta_nk.update((rho_nk.array().colwise() * Y_n.array()).matrix(),
                        x_nk);
        theta_nk.calibrate();
    }

    if (verbose)
        TLOG("Calibrated topic portions");

    return Rcpp::List::create(Rcpp::_["beta"] = logX_dk.unaryExpr(exp),
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.theta.sd"] = theta_nk.log_sd(),
                              Rcpp::_["corr"] = R_nk,
                              Rcpp::_["colsum"] = Y_n);
}
