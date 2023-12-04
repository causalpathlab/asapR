#include "rcpp_asap_regression_cbind.hh"

//' Topic statistics to estimate factor loading
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param row_file row names file (D x 1)
//' @param col_file column names file (N x 1)
//' @param idx_file matrix-market colum index file
//' @param log_beta D x K log dictionary/design matrix
//' @param beta_row_names row names log_beta (D vector)
//' @param do_stdize_beta use standardized log_beta (default: TRUE)
//' @param do_log1p do log(1+y) transformation (default: FALSE)
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
//'  \item beta.list a list of dictionary matrices (row x factor)
//'  \item corr.list a list of empirical correlation matrices (column x factor)
//'  \item colsum.list a list of column sum vectors (column x 1)
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_topic_stat_cbind(const std::vector<std::string> mtx_files,
                      const std::vector<std::string> row_files,
                      const std::vector<std::string> col_files,
                      const std::vector<std::string> idx_files,
                      const Eigen::MatrixXf log_beta,
                      const Rcpp::StringVector &beta_row_names,
                      const bool do_stdize_beta = false,
                      const bool do_log1p = false,
                      const bool verbose = false,
                      const std::size_t NUM_THREADS = 1,
                      const std::size_t BLOCK_SIZE = 100,
                      const std::size_t MAX_ROW_WORD = 2,
                      const char ROW_WORD_SEP = '_',
                      const std::size_t MAX_COL_WORD = 100,
                      const char COL_WORD_SEP = '@')
{
  

  // 1. create a feature matrix concatenating everything


  // 2. regress out batch-specific effects by SVD?


  // 3. create mtx data and build 


  // for each read_matched_reloc




}
