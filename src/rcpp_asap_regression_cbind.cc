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
asap_topic_stat_cbind(
    const std::vector<std::string> mtx_files,
    const std::vector<std::string> row_files,
    const std::vector<std::string> col_files,
    const std::vector<std::string> idx_files,
    const Eigen::MatrixXf log_beta,
    const Rcpp::StringVector beta_row_names,
    const Rcpp::Nullable<Eigen::MatrixXf> log_delta = R_NilValue,
    const Rcpp::Nullable<Eigen::MatrixXf> r_batch_names = R_NilValue,
    const bool rename_columns = true,
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

    topic_stat_options_t options;

    options.do_stdize_x = do_stdize_beta;
    options.do_log1p = do_log1p;
    options.verbose = verbose;
    options.NUM_THREADS = NUM_THREADS;
    options.BLOCK_SIZE = BLOCK_SIZE;
    options.MAX_ROW_WORD = MAX_ROW_WORD;
    options.ROW_WORD_SEP = ROW_WORD_SEP;
    options.MAX_COL_WORD = MAX_COL_WORD;
    options.COL_WORD_SEP = COL_WORD_SEP;

    /////////////////
    // check input //
    /////////////////

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    const Index B = mtx_files.size();

    ASSERT_RETL(B > 0, "Empty mtx file names");
    ASSERT_RETL(row_files.size() == B, "Need a row file for each batch");
    ASSERT_RETL(col_files.size() == B, "Need a col file for each batch");

    ERR_RET(!all_files_exist(mtx_files, verbose), "missing in the mtx files");
    ERR_RET(!all_files_exist(row_files, verbose), "missing in the row files");
    ERR_RET(!all_files_exist(col_files, verbose), "missing in the col files");

    for (Index b = 0; b < B; ++b) {
        CHK_RETL(convert_bgzip(mtx_files.at(b)));
    }

    ASSERT_RETL(idx_files.size() == B, "Need an index file for each batch");

    for (Index b = 0; b < B; ++b) {
        if (!file_exists(idx_files.at(b))) {
            CHK_RETL(build_mmutil_index(mtx_files.at(b), idx_files.at(b)));
            TLOG_(verbose, "built the missing index: " << idx_files.at(b));
        }
    }

    std::vector<std::string> batch_names;

    if (rename_columns) {
        if (r_batch_names.isNotNull()) {
            batch_names = rcpp::util::copy(Rcpp::StringVector(r_batch_names));
        } else {
            for (Index b = 0; b < B; ++b) {
                batch_names.emplace_back(std::to_string(b + 1));
            }
        }
        ASSERT_RETL(batch_names.size() == B, "check the r_batch_names");
    }

    TLOG_(verbose, "Checked the files");

    auto count_add_cols = [](Index a, std::string mtx) -> Index {
        mm_info_reader_t info;
        CHECK(peek_bgzf_header(mtx, info));
        return a + info.max_col;
    };

    const Index Ntot =
        std::accumulate(mtx_files.begin(), mtx_files.end(), 0, count_add_cols);

    TLOG_(verbose, "Found " << Ntot << " columns");

    /////////////////////////////////////////////////////////////////////
    // 1. Compute correlation matrices based on log_beta and log_delta //
    /////////////////////////////////////////////////////////////////////

    std::vector<std::string> pos2row;
    rcpp::util::copy(beta_row_names, pos2row);

    const Index D = pos2row.size();  // number of features
    const Index K = log_beta.cols(); // number of topics

    ASSERT_RETL(log_beta.rows() == D, "beta and beta row names should match");

    Mat logBeta_dk(D, K), logDelta_db(D, B);

    logBeta_dk = log_beta;

    if (log_delta.isNotNull()) {
        logDelta_db = Rcpp::as<Mat>(log_delta);
    } else {
        logDelta_db.setZero();
    }

    Mat R1_nk = Mat::Zero(Ntot, K), Y1_n = Mat::Zero(Ntot, 1);
    Mat R0_nb = Mat::Zero(Ntot, B);

    std::vector<std::string> columns;
    columns.reserve(Ntot);

    TLOG_(verbose, "Accumulating statistics...");

    for (Index b = 0; b < B; ++b) {

        const Index lb = columns.size();

        std::vector<std::string> col_b;
        CHK_RETL_(read_line_file(col_files.at(b),
                                 col_b,
                                 MAX_COL_WORD,
                                 COL_WORD_SEP),
                  "unable to read " << col_files.at(b))

        if (rename_columns) {
            const std::string bname = batch_names.at(b);
            auto app_b = [&bname](std::string &x) { x + "_" + bname; };
            std::for_each(col_b.begin(), col_b.end(), app_b);
        }

        std::copy(col_b.begin(), col_b.end(), std::back_inserter(columns));

        const Index ub = columns.size();

        Mat r1_b_nk, y1_b_n;

        CHK_RETL_(asap_topic_stat_mtx(mtx_files.at(b),
                                      row_files.at(b),
                                      col_files.at(b),
                                      idx_files.at(b),
                                      logBeta_dk,
                                      pos2row,
                                      options,
                                      r1_b_nk,
                                      y1_b_n),
                  "unable to compute beta stat: " << (b + 1) << "/B");

        R1_nk.middleRows(lb, r1_b_nk.rows()) = r1_b_nk;
        Y1_n.middleRows(lb, y1_b_n.rows()) = y1_b_n;

        Mat r0_b_nb, y0_b_n;

        CHK_RETL_(asap_topic_stat_mtx(mtx_files.at(b),
                                      row_files.at(b),
                                      col_files.at(b),
                                      idx_files.at(b),
                                      logDelta_db,
                                      pos2row,
                                      options,
                                      r0_b_nb,
                                      y0_b_n),
                  "unable to compute delta stat: " << (b + 1) << "/B");

        R0_nb.middleRows(lb, r0_b_nb.rows()) = r0_b_nb;
    }

    TLOG_(verbose, "Got em all");

    ///////////////////////////////////////////
    // 2. Regress out batch-specific effects //
    ///////////////////////////////////////////

    TLOG_(verbose, "Regressing out putative batch effects");
    residual_columns_inplace(R1_nk, R0_nb);

    using namespace rcpp::util;
    using namespace Rcpp;

    const std::vector<std::string> &d_ = pos2row;
    std::vector<std::string> k_;
    for (std::size_t k = 1; k <= K; ++k) {
        k_.push_back(std::to_string(k));
    }

    TLOG_(verbose, "Done");

    exp_op<Mat> exp;

    return List::create(_["beta"] = named(logBeta_dk.unaryExpr(exp), d_, k_),
                        _["corr"] = named(R1_nk, columns, k_),
                        _["colsum"] = named_rows(Y1_n, columns),
                        _["rownames"] = pos2row,
                        _["colnames"] = columns);
}
