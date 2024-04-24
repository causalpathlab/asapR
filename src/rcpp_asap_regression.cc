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
//'
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
               const std::size_t NUM_THREADS = 0,
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

    const std::size_t nthreads =
        (NUM_THREADS > 0 ? NUM_THREADS : omp_get_max_threads());

    Eigen::setNbThreads(nthreads);

    const Index D = beta_dk.rows();
    const Index K = beta_dk.cols();
    const Index N = R_nk.rows();

    ASSERT_RETL(Y_n.rows() == R_nk.rows(),
                "R and Y must have the same number of rows");

    Mat x_nk = ColVec::Ones(N) * beta_dk.colwise().sum(); // N x K

    Mat logRho_nk(N, K), rho_nk(N, K);   // N x K
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

//' PMF statistics to estimate factor loading
//'
//' @param y_dn sparse data matrix (D x N)
//' @param log_beta D x K log dictionary/design matrix
//' @param beta_row_names row names log_beta (D vector)
//' @param r_log_delta D x B log batch effect matrix
//' @param do_stdize_beta use standardized log_beta (Default: TRUE)
//' @param do_log1p do log(1+y) transformation
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param CELL_NORM sample normalization constant (default: 1e4)
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
//'  \item colsum the sum of each column (column x 1)
//'  \item rownames row names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_pmf_stat(const Eigen::SparseMatrix<float> &y_dn,
              const Eigen::MatrixXf log_beta,
              const Rcpp::StringVector beta_row_names,
              const Rcpp::Nullable<Eigen::MatrixXf> r_log_delta = R_NilValue,
              const bool do_stdize_beta = true,
              const bool do_log1p = false,
              const bool verbose = false,
              const std::size_t NUM_THREADS = 0,
              const double CELL_NORM = 1e4,
              const std::size_t BLOCK_SIZE = 1000)
{

    using namespace asap::regression;

    std::vector<std::string> pos2row;
    rcpp::util::copy(beta_row_names, pos2row);

    stat_options_t options;

    options.do_stdize_x = do_stdize_beta;
    options.do_log1p = do_log1p;
    options.verbose = verbose;
    options.NUM_THREADS = NUM_THREADS;
    options.BLOCK_SIZE = BLOCK_SIZE;
    options.CELL_NORM = CELL_NORM;

    eigenSparse_data_t data(y_dn, pos2row);

    Mat Rtot_nk, Ytot_n, delta_db;
    exp_op<Mat> exp;

    if (r_log_delta.isNotNull()) {
        Mat log_delta = Rcpp::as<Mat>(r_log_delta);

        ASSERT_RETL(log_delta.rows() == log_beta.rows(),
                    "rows(delta) != rows(beta)");

        CHK_RETL_(run_pmf_stat_adj(data,
                                   log_beta,
                                   log_delta,
                                   pos2row,
                                   options,
                                   Rtot_nk,
                                   Ytot_n),
                  "failed to compute topic pmf ipw stat");

        delta_db = log_delta.unaryExpr(exp);
    } else {

        CHK_RETL_(run_pmf_stat(data,
                               log_beta,
                               pos2row,
                               options,
                               Rtot_nk,
                               Ytot_n),
                  "failed to compute topic pmf stat");
    }

    const Index N = Rtot_nk.rows(), K = Rtot_nk.cols();

    using namespace rcpp::util;
    using namespace Rcpp;

    std::vector<std::string> &d_ = pos2row;
    std::vector<std::string> k_;
    for (std::size_t k = 1; k <= K; ++k) {
        k_.push_back(std::to_string(k));
    }

    return List::create(_["beta"] = named(log_beta.unaryExpr(exp), d_, k_),
                        _["delta"] = delta_db,
                        _["corr"] = Rtot_nk,
                        _["colsum"] = Ytot_n,
                        _["rownames"] = pos2row);
}

//' PMF statistics to estimate factor loading
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param row_file row names file (D x 1)
//' @param col_file column names file (N x 1)
//' @param idx_file matrix-market colum index file
//' @param log_beta D x K log dictionary/design matrix
//' @param beta_row_names row names log_beta (D vector)
//' @param r_log_delta D x B log batch effect matrix
//' @param do_stdize_beta use standardized log_beta (Default: TRUE)
//' @param do_log1p do log(1+y) transformation
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param CELL_NORM sample normalization constant (default: 1e4)
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param MAX_ROW_WORD maximum words per line in `row_files[i]`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @param MAX_COL_WORD maximum words per line in `col_files[i]`
//' @param COL_WORD_SEP word separation character to replace white space
//'
//' @return a list that contains:
//' \itemize{
//'  \item beta the dictionary matrix of topics (row x factor)
//'  \item delta the dictionary matrix of batch effects (row x batch)
//'  \item corr empirical correlation (column x factor)
//'  \item colsum the sum of each column (column x 1)
//'  \item rownames row names
//'  \item rownames column names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_pmf_stat_mtx(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    const std::string idx_file,
    const Eigen::MatrixXf log_beta,
    const Rcpp::StringVector beta_row_names,
    const Rcpp::Nullable<Eigen::MatrixXf> r_log_delta = R_NilValue,
    const bool do_stdize_beta = true,
    const bool do_log1p = false,
    const bool verbose = false,
    const std::size_t NUM_THREADS = 0,
    const double CELL_NORM = 1e4,
    const std::size_t BLOCK_SIZE = 1000,
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@')
{

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    std::vector<std::string> pos2row;
    rcpp::util::copy(beta_row_names, pos2row);

    asap::regression::stat_options_t options;

    options.do_stdize_x = do_stdize_beta;
    options.do_log1p = do_log1p;
    options.verbose = verbose;
    options.NUM_THREADS = NUM_THREADS;
    options.BLOCK_SIZE = BLOCK_SIZE;
    options.MAX_ROW_WORD = MAX_ROW_WORD;
    options.ROW_WORD_SEP = ROW_WORD_SEP;
    options.MAX_COL_WORD = MAX_COL_WORD;
    options.COL_WORD_SEP = COL_WORD_SEP;
    options.CELL_NORM = CELL_NORM;

    Mat Rtot_nk, Ytot_n, delta_db;
    exp_op<Mat> exp;

    mtx_tuple_t tup(mtx_tuple_t::MTX { mtx_file },
                    mtx_tuple_t::ROW { row_file },
                    mtx_tuple_t::COL { col_file },
                    mtx_tuple_t::IDX { idx_file });

    mtx_data_t data(tup, options.MAX_ROW_WORD, options.ROW_WORD_SEP);

    if (r_log_delta.isNotNull()) {
        Mat log_delta = Rcpp::as<Mat>(r_log_delta);

        ASSERT_RETL(log_delta.rows() == log_beta.rows(),
                    "rows(delta) != rows(beta)");

        CHK_RETL_(asap::regression::run_pmf_stat_adj(data,
                                                     log_beta,
                                                     log_delta,
                                                     pos2row,
                                                     options,
                                                     Rtot_nk,
                                                     Ytot_n),
                  "unable to compute topic statistics");

        delta_db = log_delta.unaryExpr(exp);
    } else {

        CHK_RETL_(asap::regression::run_pmf_stat(data,
                                                 log_beta,
                                                 pos2row,
                                                 options,
                                                 Rtot_nk,
                                                 Ytot_n),
                  "unable to compute topic statistics");
    }

    const Index N = Rtot_nk.rows(), K = Rtot_nk.cols();

    using namespace rcpp::util;
    using namespace Rcpp;

    std::vector<std::string> &d_ = pos2row;
    std::vector<std::string> k_;
    for (std::size_t k = 1; k <= K; ++k) {
        k_.push_back(std::to_string(k));
    }
    const std::vector<std::string> file_ { mtx_file };

    std::vector<std::string> coln;
    CHK_RETL_(read_line_file(col_file, coln, MAX_COL_WORD, COL_WORD_SEP),
              "Failed to read the column name file: " << col_file);

    ASSERT_RETL(N == coln.size(),
                "Different #columns: " << N << " vs. " << coln.size());

    return List::create(_["beta"] = named(log_beta.unaryExpr(exp), d_, k_),
                        _["delta"] = delta_db,
                        _["corr"] = named(Rtot_nk, coln, k_),
                        _["colsum"] = named(Ytot_n, coln, file_),
                        _["rownames"] = pos2row,
                        _["colnames"] = coln);
}
