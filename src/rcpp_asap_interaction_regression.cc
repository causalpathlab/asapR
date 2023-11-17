#include "rcpp_asap_regression.hh"

//' Topic statistics to estimate factor loading
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param row_file row names file (D x 1)
//' @param col_file column names file (N x 1)
//' @param idx_file matrix-market colum index file
//' @param log_x D x K log dictionary/design matrix
//' @param x_row_names row names log_x (D vector)
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
asap_interaction_topic_stat(const std::string mtx_file,
                            const std::string row_file,
                            const std::string col_file,
                            const std::string idx_file,
                            const Eigen::MatrixXf log_x,
                            const Rcpp::StringVector &x_row_names,
                            const std::vector<std::size_t> knn_src,
                            const std::vector<std::size_t> knn_tgt,
                            const std::vector<float> knn_weight,
                            const bool do_log1p = false,
                            const bool verbose = false,
                            const std::size_t NUM_THREADS = 1,
                            const std::size_t BLOCK_SIZE = 100,
                            const std::size_t MAX_ROW_WORD = 2,
                            const char ROW_WORD_SEP = '_',
                            const std::size_t MAX_COL_WORD = 100,
                            const char COL_WORD_SEP = '@')
{
    return Rcpp::List::create();
}
