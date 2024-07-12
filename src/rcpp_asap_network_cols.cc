#include "rcpp_asap_network_cols.hh"

//' Identify pairs of columns interacting with one another
//'
//' @param y_dn sparse data matrix (D x N)
//' @param z_dm sparse data matrix (D x M)
//'
//' @param log_beta D x K log dictionary/design matrix
//' @param beta_row_names row names log_beta (D vector)
//'
//' @param knn How many nearest neighbours we want (default: 10)
//'
//' @param r_log_delta D x B log batch effect matrix
//'
//' @param do_stdize_beta use standardized log_beta (Default: TRUE)
//' @param do_log1p do log(1+y) transformation
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param CELL_NORM sample normalization constant (default: 1e4)
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//'
// [[Rcpp::export]]
Rcpp::List
asap_build_interacting_columns(
    const Eigen::SparseMatrix<float> &y_dn,
    const Eigen::SparseMatrix<float> &z_dm,
    const Eigen::MatrixXf log_beta,
    const Rcpp::StringVector beta_row_names,
    const std::size_t knn = 10,
    const Rcpp::Nullable<Eigen::MatrixXf> r_log_delta = R_NilValue,
    const bool do_stdize_beta = true,
    const bool do_log1p = false,
    const bool verbose = true,
    const std::size_t NUM_THREADS = 1,
    const double CELL_NORM = 1e4,
    const std::size_t BLOCK_SIZE = 1000)
{

    asap::regression::stat_options_t regOpt;

    regOpt.do_stdize_x = do_stdize_beta;
    regOpt.do_log1p = do_log1p;
    regOpt.verbose = verbose;
    regOpt.NUM_THREADS = NUM_THREADS;
    regOpt.BLOCK_SIZE = BLOCK_SIZE;
    regOpt.CELL_NORM = CELL_NORM;

    std::vector<std::string> pos2row;
    rcpp::util::copy(beta_row_names, pos2row);
    eigenSparse_data_t lhs_data(y_dn, pos2row);
    eigenSparse_data_t rhs_data(z_dm, pos2row);

    TLOG_(verbose, "Loaded two data sets");

    Mat Q_lhs_kn, Q_rhs_km;
    const bool self_interaction = false;

    CHK_RETL_(run_asap_regression_both(lhs_data,
                                       rhs_data,
                                       log_beta,
                                       r_log_delta,
                                       pos2row,
                                       regOpt,
                                       self_interaction,
                                       Q_lhs_kn,
                                       Q_rhs_km),
              "failed to evaluate regression");

    TLOG_(verbose, "Computed feature matrices");

    std::vector<std::tuple<Index, Index, Scalar>> match_result;

    standardize_columns_inplace(Q_lhs_kn);
    standardize_columns_inplace(Q_rhs_km);

    mmutil::match::options_t matchOpt;
    matchOpt.block_size = BLOCK_SIZE;
    matchOpt.num_threads = NUM_THREADS;
    matchOpt.knn_per_batch = knn;

    TLOG_(verbose, "standardized the feature matrices");

    CHK_RETL_(match_euclidean_annoy(Q_lhs_kn, Q_rhs_km, matchOpt, match_result),
              "Couldn't match between the lhs and rhs data");

    TLOG_(verbose, "Found matching between them");

    SpMat W = build_eigen_sparse(match_result,
                                 lhs_data.max_col(),
                                 rhs_data.max_col());

    return Rcpp::List::create(Rcpp::_["knn"] = rcpp::util::build_sparse_list(W),
                              Rcpp::_["W"] = W);

    return Rcpp::List::create();
}

//' Identify pairs of columns interacting with one another
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param row_file row names file (D x 1)
//' @param col_file column names file (N x 1)
//' @param idx_file matrix-market colum index file
//'
//' @param log_beta D x K log dictionary/design matrix
//' @param beta_row_names row names log_beta (D vector)
//' @param knn How many nearest neighbours we want (default: 10)
//'
//' @param r_log_delta D x B log batch effect matrix
//'
//' @param mtx_file_rhs right-hand-side matrix-market-formatted data file (bgzip)
//' @param row_file_rhs right-hand-side row names (gene/feature names)
//' @param col_file_rhs right-hand-side column names (cell/column names)
//' @param idx_file_rhs right-hand-side matrix-market colum index file
//'
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
// [[Rcpp::export]]
Rcpp::List
asap_build_interaction_columns_mtx(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    const std::string idx_file,
    const Eigen::MatrixXf log_beta,
    const Rcpp::StringVector beta_row_names,
    const std::size_t knn = 10,
    const Rcpp::Nullable<Eigen::MatrixXf> r_log_delta = R_NilValue,
    const Rcpp::Nullable<std::string> mtx_file_rhs = R_NilValue,
    const Rcpp::Nullable<std::string> row_file_rhs = R_NilValue,
    const Rcpp::Nullable<std::string> col_file_rhs = R_NilValue,
    const Rcpp::Nullable<std::string> idx_file_rhs = R_NilValue,
    const bool do_stdize_beta = true,
    const bool do_log1p = false,
    const bool verbose = true,
    const std::size_t NUM_THREADS = 1,
    const double CELL_NORM = 1e4,
    const std::size_t BLOCK_SIZE = 1000,
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@')
{

    std::vector<std::string> pos2row;
    rcpp::util::copy(beta_row_names, pos2row);

    // regression by log-beta
    asap::regression::stat_options_t regOpt;

    regOpt.do_stdize_x = do_stdize_beta;
    regOpt.do_log1p = do_log1p;
    regOpt.verbose = verbose;
    regOpt.NUM_THREADS = NUM_THREADS;
    regOpt.BLOCK_SIZE = BLOCK_SIZE;
    regOpt.CELL_NORM = CELL_NORM;

    std::string _mtx_file_rhs = mtx_file, _row_file_rhs = row_file,
                _col_file_rhs = col_file, _idx_file_rhs = idx_file;

    bool self_interaction = true;

    if (mtx_file_rhs.isNotNull() && row_file_rhs.isNotNull() &&
        col_file_rhs.isNotNull() && idx_file_rhs.isNotNull()) {

        _mtx_file_rhs = Rcpp::as<std::string>(mtx_file_rhs);
        _row_file_rhs = Rcpp::as<std::string>(row_file_rhs);
        _col_file_rhs = Rcpp::as<std::string>(col_file_rhs);
        _idx_file_rhs = Rcpp::as<std::string>(idx_file_rhs);

        self_interaction = false;
    }

    mtx_data_t lhs_data(mtx_tuple_t(mtx_tuple_t::MTX { mtx_file },
                                    mtx_tuple_t::ROW { row_file },
                                    mtx_tuple_t::COL { col_file },
                                    mtx_tuple_t::IDX { idx_file }),
                        MAX_ROW_WORD,
                        ROW_WORD_SEP,
                        MAX_COL_WORD,
                        COL_WORD_SEP);

    mtx_data_t rhs_data(mtx_tuple_t(mtx_tuple_t::MTX { _mtx_file_rhs },
                                    mtx_tuple_t::ROW { _row_file_rhs },
                                    mtx_tuple_t::COL { _col_file_rhs },
                                    mtx_tuple_t::IDX { _idx_file_rhs }),
                        MAX_ROW_WORD,
                        ROW_WORD_SEP,
                        MAX_COL_WORD,
                        COL_WORD_SEP);

    TLOG_(verbose, "Loaded two data sets");

    Mat Q_lhs_kn, Q_rhs_km;

    CHK_RETL_(run_asap_regression_both(lhs_data,
                                       rhs_data,
                                       log_beta,
                                       r_log_delta,
                                       pos2row,
                                       regOpt,
                                       self_interaction,
                                       Q_lhs_kn,
                                       Q_rhs_km),
              "failed to evaluate regression");

    TLOG_(verbose, "Computed feature matrices");

    std::vector<std::tuple<Index, Index, Scalar>> match_result;

    standardize_columns_inplace(Q_lhs_kn);
    standardize_columns_inplace(Q_rhs_km);

    TLOG_(verbose, "standardized the feature matrices");

    mmutil::match::options_t matchOpt;
    matchOpt.block_size = BLOCK_SIZE;
    matchOpt.num_threads = NUM_THREADS;
    matchOpt.knn_per_batch = knn;

    CHK_RETL_(match_euclidean_annoy(Q_lhs_kn, Q_rhs_km, matchOpt, match_result),
              "Couldn't match between the lhs and rhs data");

    TLOG_(verbose, "Found matching between them");

    SpMat W = build_eigen_sparse(match_result,
                                 lhs_data.max_col(),
                                 rhs_data.max_col());

    return Rcpp::List::create(Rcpp::_["lhs.names"] = lhs_data.col_names(),
                              Rcpp::_["rhs.names"] = rhs_data.col_names(),
                              Rcpp::_["W"] = W,
                              Rcpp::_["knn"] =
                                  rcpp::util::build_sparse_list(W));
}
