#include "rcpp_asap_regression.hh"

//' PMF regression
//'
//' @param y_dn sparse data matrix (D x N)
//' @param log_beta D x K log dictionary/design matrix
//' @param beta_row_names row names log_beta (D vector)
//' @param r_log_delta D x B log batch effect matrix
//' @param do_stdize_beta use standardized log_beta (Default: TRUE)
//' @param do_stdize_r standardize correlation matrix R (default: TRUE)
//' @param do_log1p do log(1+y) transformation
//' @param verbose verbosity
//' @param a0 gamma(a0, b0) (default: 1)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param max_iter maximum iterations (default: 10)
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
//'  \item delta the dictionary matrix of batch effects (row x batch)
//'  \item corr empirical correlation (column x factor)
//'  \item theta factor loading (column x factor)
//'  \item log.theta log-scaled factor loading (column x factor)
//'  \item colsum the sum of each column (column x 1)
//'  \item rownames row/feature names
//'  \item colnames column/sample names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_pmf_regression(
    const Eigen::SparseMatrix<float> &y_dn,
    const Eigen::MatrixXf log_beta,
    const Rcpp::StringVector beta_row_names,
    const Rcpp::Nullable<Eigen::MatrixXf> r_log_delta = R_NilValue,
    const bool do_stdize_beta = true,
    const bool do_stdize_r = true,
    const bool do_log1p = false,
    const double a0 = 1.0,
    const double b0 = 1.0,
    const std::size_t max_iter = 10,
    const bool verbose = false,
    const std::size_t NUM_THREADS = 0,
    const std::size_t BLOCK_SIZE = 1000)
{

    using namespace asap::regression;

    std::vector<std::string> row_names;
    rcpp::util::copy(beta_row_names, row_names);

    stat_options_t options;

    options.do_stdize_x = do_stdize_beta;
    options.do_stdize_r = do_stdize_r;
    options.do_log1p = do_log1p;
    options.verbose = verbose;
    options.NUM_THREADS = NUM_THREADS;
    options.BLOCK_SIZE = BLOCK_SIZE;
    options.max_iter = max_iter;
    options.a0 = a0;
    options.b0 = b0;

    eigenSparse_data_t data(y_dn, row_names);

    Mat log_delta;

    if (r_log_delta.isNotNull()) {
        log_delta = Rcpp::as<Mat>(r_log_delta);
    } else {
        log_delta = Mat::Ones(log_beta.rows(), 1);
    }

    std::vector<std::string> col_names;
    for (std::size_t k = 1; k <= y_dn.rows(); ++k) {
        col_names.push_back(std::to_string(k));
    }

    return run_pmf_regression(data,
                              log_beta,
                              log_delta,
                              row_names,
                              col_names,
                              options);
}

//' PMF regression
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param row_file row names file (D x 1)
//' @param col_file column names file (N x 1)
//' @param idx_file matrix-market colum index file
//' @param log_beta D x K log dictionary/design matrix (default: TRUE)
//' @param do_stdize_r standardize correlation matrix R (default: TRUE)
//' @param beta_row_names row names log_beta (D vector)
//' @param r_log_delta D x B log batch effect matrix
//' @param do_stdize_beta use standardized log_beta (Default: FALSE)
//' @param do_log1p do log(1+y) transformation
//' @param verbose verbosity
//' @param a0 gamma(a0, b0) (default: 1)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param max_iter maximum iterations (default: 10)
//' @param NUM_THREADS number of threads in data reading
//'
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param MAX_ROW_WORD maximum words per line in `row_files[i]`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @param MAX_COL_WORD maximum words per line in `col_files[i]`
//' @param COL_WORD_SEP word separation character to replace white space
//'
//' @return a list that contains:
//' \itemize{
//'  \item beta dictionary matrix (row x factor)
//'  \item delta the dictionary matrix of batch effects (row x batch)
//'  \item corr empirical correlation (column x factor)
//'  \item theta factor loading (column x factor)
//'  \item log.theta log-scaled factor loading (column x factor)
//'  \item colsum the sum of each column (column x 1)
//'  \item rownames row/feature names
//'  \item colnames column/sample names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_pmf_regression_mtx(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    const std::string idx_file,
    const Eigen::MatrixXf log_beta,
    const Rcpp::StringVector beta_row_names,
    const Rcpp::Nullable<Eigen::MatrixXf> r_log_delta = R_NilValue,
    const bool do_stdize_beta = true,
    const bool do_stdize_r = true,
    const bool do_log1p = false,
    const double a0 = 1.0,
    const double b0 = 1.0,
    const std::size_t max_iter = 10,
    const bool verbose = false,
    const std::size_t NUM_THREADS = 0,
    const std::size_t BLOCK_SIZE = 1000,
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@')
{

    std::vector<std::string> row_names;
    rcpp::util::copy(beta_row_names, row_names);

    asap::regression::stat_options_t options;

    options.do_stdize_x = do_stdize_beta;
    options.do_log1p = do_log1p;
    options.verbose = verbose;
    options.NUM_THREADS = NUM_THREADS;
    options.BLOCK_SIZE = BLOCK_SIZE;
    options.max_iter = max_iter;
    options.a0 = a0;
    options.b0 = b0;

    options.MAX_ROW_WORD = MAX_ROW_WORD;
    options.ROW_WORD_SEP = ROW_WORD_SEP;
    options.MAX_COL_WORD = MAX_COL_WORD;
    options.COL_WORD_SEP = COL_WORD_SEP;

    mtx_tuple_t tup(mtx_tuple_t::MTX { mtx_file },
                    mtx_tuple_t::ROW { row_file },
                    mtx_tuple_t::COL { col_file },
                    mtx_tuple_t::IDX { idx_file });

    mtx_data_t data(tup, options.MAX_ROW_WORD, options.ROW_WORD_SEP);

    Mat log_delta;

    if (r_log_delta.isNotNull()) {
        log_delta = Rcpp::as<Mat>(r_log_delta);
    } else {
        log_delta = Mat::Ones(log_beta.rows(), 1);
    }

    std::vector<std::string> col_names;
    CHK_RETL_(read_line_file(col_file, col_names, MAX_COL_WORD, COL_WORD_SEP),
              "Failed to read the column name file: " << col_file);

    return run_pmf_regression(data,
                              log_beta,
                              log_delta,
                              row_names,
                              col_names,
                              options);
}

//' Calibrate topic proportions based on sufficient statistics
//'
//' @param beta_dk dictionary matrix (feature D  x factor K)
//' @param R_nk correlation matrix (sample N x factor K)
//' @param Ysum_n sum vector (sample N x 1)
//' @param a0 gamma(a0, b0) (default: 1)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param max_iter maximum iterations (default: 10)
//' @param NUM_THREADS number of parallel threads (default: 1)
//'
//' @param do_stdize_r standardize correlation matrix R (default: FALSE)
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
               const Eigen::MatrixXf Ysum_n,
               const double a0 = 1.0,
               const double b0 = 1.0,
               const std::size_t max_iter = 10,
               const std::size_t NUM_THREADS = 0,
               const bool do_stdize_r = true,
               const bool verbose = true)
{

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    exp_op<Mat> exp;
    softmax_op_t<Mat> softmax;

    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Mat, RNG>;
    RNG rng;

    const std::size_t nthreads =
        (NUM_THREADS > 0 ? NUM_THREADS : omp_get_max_threads());

    Eigen::setNbThreads(nthreads);

    const Index D = beta_dk.rows();
    const Index K = beta_dk.cols();
    const Index N = R_nk.rows();

    ASSERT_RETL(Ysum_n.rows() == R_nk.rows(),
                "R and Y must have the same number of rows");

    Mat logRho_nk(N, K), rho_nk(N, K);   // N x K
    gamma_t theta_nk(N, K, a0, b0, rng); // N x K

    ColVec degree_n = Ysum_n / static_cast<Scalar>(D);
    Mat x_nk = degree_n * beta_dk.colwise().sum(); // N x K

    Mat r_nk = R_nk;
    if (do_stdize_r) {
        asap::util::stretch_matrix_columns_inplace(r_nk);
    }

    for (std::size_t t = 0; t < max_iter; ++t) {

        logRho_nk = r_nk + theta_nk.log_mean();

        for (Index jj = 0; jj < N; ++jj) {
            logRho_nk.row(jj) = softmax.log_row(logRho_nk.row(jj));
        }
        rho_nk = logRho_nk.unaryExpr(exp);

        theta_nk.update(Ysum_n.asDiagonal() * rho_nk, x_nk);
        theta_nk.calibrate();
    }

    return Rcpp::List::create(Rcpp::_["beta"] = beta_dk,
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.theta.sd"] = theta_nk.log_sd());
}
