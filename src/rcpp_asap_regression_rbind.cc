#include "rcpp_asap_regression_rbind.hh"

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
//'  \item beta.list a list of dictionary matrices (row x factor)
//'  \item corr.list a list of empirical correlation matrices (column x factor)
//'  \item colsum.list a list of column sum vectors (column x 1)
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_topic_stat_rbind(const std::vector<std::string> mtx_files,
                      const std::vector<std::string> row_files,
                      const std::vector<std::string> col_files,
                      const std::vector<std::string> idx_files,
                      const std::vector<Eigen::MatrixXf> &logX_vec,
                      const std::vector<Rcpp::StringVector> &x_row_names_vec,
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

    topic_stat_options_t options;

    options.do_log1p = do_log1p;
    options.verbose = verbose;
    options.NUM_THREADS = NUM_THREADS;
    options.BLOCK_SIZE = BLOCK_SIZE;
    options.MAX_ROW_WORD = MAX_ROW_WORD;
    options.ROW_WORD_SEP = ROW_WORD_SEP;
    options.MAX_COL_WORD = MAX_COL_WORD;
    options.COL_WORD_SEP = COL_WORD_SEP;

    const Index B = mtx_files.size();

    ASSERT_RETL(B > 0, "Empty mtx file names");
    ASSERT_RETL(row_files.size() == B, "Need a row file for each data type");
    ASSERT_RETL(col_files.size() == B, "Need a col file for each data type");
    ASSERT_RETL(logX_vec.size() == B, "Need a logX matrix for each data type");
    ASSERT_RETL(x_row_names_vec.size() == B,
                "Need a rownames vector for each data type");

    ASSERT_RETL(all_files_exist(mtx_files, verbose),
                "missing in the mtx files");
    ASSERT_RETL(all_files_exist(row_files, verbose),
                "missing in the row files");
    ASSERT_RETL(all_files_exist(col_files, verbose),
                "missing in the col files");

    for (Index b = 0; b < B; ++b) {
        CHK_RETL(convert_bgzip(mtx_files.at(b)));
    }

    ASSERT_RETL(idx_files.size() == B, "Need an index file for each data type");

    for (Index b = 0; b < B; ++b) {
        if (!file_exists(idx_files.at(b))) {
            CHK_RETL(build_mmutil_index(mtx_files.at(b), idx_files.at(b)));
            TLOG_(verbose, "built the missing index: " << idx_files.at(b));
        }
    }

    TLOG_(verbose, "Checked the files");

    exp_op<Mat> exp;
    std::list<Rcpp::NumericMatrix> beta_dk_list, R_nk_list, Y_n_list;
    std::list<Rcpp::StringVector> colnames_list;

    for (Index b = 0; b < B; ++b) {

        const Mat log_x = logX_vec.at(b);
        const std::vector<std::string> pos2row =
            Rcpp::as<std::vector<std::string>>(x_row_names_vec.at(b));

        Mat Rtot_nk, Ytot_n;
        CHK_RETL_(asap_topic_stat_mtx(mtx_files.at(b),
                                      row_files.at(b),
                                      col_files.at(b),
                                      idx_files.at(b),
                                      log_x,
                                      pos2row,
                                      options,
                                      Rtot_nk,
                                      Ytot_n),
                  "unable to compute topic statistics [ " << (b + 1) << " ]");

        const Index N = Rtot_nk.rows(), K = Rtot_nk.cols();

        const std::vector<std::string> &d_ = pos2row;
        std::vector<std::string> k_;
        {
            for (std::size_t k = 1; k <= K; ++k) {
                k_.push_back(std::to_string(k));
            }
        }
        std::vector<std::string> coln;
        {
            CHK_RETL_(read_line_file(col_files.at(b),
                                     coln,
                                     MAX_COL_WORD,
                                     COL_WORD_SEP),
                      "Failed to read the column name file: "
                          << col_files.at(b));

            ASSERT_RETL(N == coln.size(),
                        "Different #columns: " << N << " vs. " << coln.size());
        }
        const std::vector<std::string> file_ { mtx_files.at(b) };
        beta_dk_list.push_back(rcpp::util::named(log_x.unaryExpr(exp), d_, k_));
        R_nk_list.push_back(rcpp::util::named(Rtot_nk, coln, k_));
        Y_n_list.push_back(rcpp::util::named(Ytot_n, coln, file_));
        colnames_list.push_back(Rcpp::wrap(coln));

        TLOG_(verbose, "Finished on [" << (b + 1) << " / " << B << "] stat");
    }

    return Rcpp::List::create(Rcpp::_["beta.list"] = beta_dk_list,
                              Rcpp::_["corr.list"] = R_nk_list,
                              Rcpp::_["colsum.list"] = Y_n_list,
                              Rcpp::_["rownames.list"] = x_row_names_vec,
                              Rcpp::_["colnames.list"] = colnames_list);
}
