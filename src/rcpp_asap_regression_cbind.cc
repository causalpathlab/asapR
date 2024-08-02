#include "rcpp_asap_regression_cbind.hh"

//' PMF statistics to estimate factor loading
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param row_file row names file (D x 1)
//' @param col_file column names file (N x 1)
//' @param idx_file matrix-market colum index file
//' @param log_beta D x K log dictionary/design matrix
//' @param beta_row_names row names log_beta (D vector)
//' @param log_delta D x B log batch effects
//' @param batch_names batch names (optional)
//' @param rename_columns append batch name at the end of each column name (default: FALSE)
//' @param do_stdize_beta use standardized log_beta (default: TRUE)
//' @param do_stdize_r standardize correlation matrix R (default: TRUE)
//' @param do_log1p do log(1+y) transformation (default: FALSE)
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
//'  \item beta the dictionary matrix (row x factor)
//'  \item delta the dictionary matrix of batch effects (row x batch)
//'  \item corr empirical correlation (column x factor)
//'  \item theta factor loading (column x factor)
//'  \item log.theta log-scaled factor loading (column x factor)
//'  \item colsum column sum (column x 1)
//'  \item rownames row names
//'  \item batch.names batch names (based on
//'  \item batch.index
//'  \item colnames column names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_pmf_regression_cbind_mtx(
    const std::vector<std::string> mtx_files,
    const std::vector<std::string> row_files,
    const std::vector<std::string> col_files,
    const std::vector<std::string> idx_files,
    const Eigen::MatrixXf log_beta,
    const Rcpp::StringVector beta_row_names,
    const Rcpp::Nullable<Eigen::MatrixXf> log_delta = R_NilValue,
    const Rcpp::Nullable<Eigen::MatrixXf> batch_names = R_NilValue,
    const bool rename_columns = false,
    const bool do_stdize_beta = true,
    const bool do_stdize_r = true,
    const bool do_log1p = false,
    const bool verbose = false,
    const double a0 = 1.0,
    const double b0 = 1.0,
    const std::size_t max_iter = 10,
    const std::size_t NUM_THREADS = 0,
    const std::size_t BLOCK_SIZE = 1000,
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@')
{

    using RNG = dqrng::xoshiro256plus;
    RNG rng;

    using namespace asap::regression;

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

    /////////////////
    // check input //
    /////////////////

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    const Index num_data_batch = mtx_files.size();

    ASSERT_RETL(num_data_batch > 0, "Empty mtx file names");
    ASSERT_RETL(row_files.size() == num_data_batch,
                "Need a row file for each batch");
    ASSERT_RETL(col_files.size() == num_data_batch,
                "Need a col file for each batch");

    ERR_RET(!all_files_exist(mtx_files, verbose), "missing in the mtx files");
    ERR_RET(!all_files_exist(row_files, verbose), "missing in the row files");
    ERR_RET(!all_files_exist(col_files, verbose), "missing in the col files");

    for (Index b = 0; b < num_data_batch; ++b) {
        CHK_RETL(convert_bgzip(mtx_files.at(b)));
    }

    ASSERT_RETL(idx_files.size() == num_data_batch,
                "Need an index file for each batch");

    for (Index b = 0; b < num_data_batch; ++b) {
        if (!file_exists(idx_files.at(b))) {
            CHK_RETL(build_mmutil_index(mtx_files.at(b), idx_files.at(b)));
            TLOG_(verbose, "built the missing index: " << idx_files.at(b));
        }
    }

    std::vector<std::string> _batch_names;

    if (rename_columns) {
        if (batch_names.isNotNull()) {
            _batch_names = rcpp::util::copy(Rcpp::StringVector(batch_names));
        } else {
            for (Index b = 0; b < num_data_batch; ++b) {
                _batch_names.emplace_back(std::to_string(b + 1));
            }
        }
        ASSERT_RETL(_batch_names.size() == num_data_batch,
                    "check the batch_names");
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

    std::vector<std::string> pos2row;
    rcpp::util::copy(beta_row_names, pos2row);

    const Index D = pos2row.size();  // number of features
    const Index K = log_beta.cols(); // number of topics

    ASSERT_RETL(log_beta.rows() == D, "beta and beta row names should match");

    Mat logBeta_dk(D, K);

    logBeta_dk = log_beta;

    Mat logDelta_db;

    Mat R_nk = Mat::Zero(Ntot, K), Y_n = Mat::Zero(Ntot, 1);

    std::vector<std::string> columns;
    std::vector<Index> batch_indexes;
    columns.reserve(Ntot);
    batch_indexes.reserve(Ntot);

    using gamma_t = gamma_param_t<Mat, RNG>;

    TLOG_(verbose, "Accumulating statistics...");
    std::vector<std::size_t> batch_sizes;
    batch_sizes.reserve(num_data_batch);

    for (Index b = 0; b < num_data_batch; ++b) {

        const Index lb = columns.size();

        std::vector<std::string> col_b;
        CHK_RETL_(read_line_file(col_files.at(b),
                                 col_b,
                                 MAX_COL_WORD,
                                 COL_WORD_SEP),
                  "unable to read " << col_files.at(b))

        if (rename_columns) {
            const std::string bname = _batch_names.at(b);
            auto app_b = [&bname](std::string &x) { x += "_" + bname; };
            std::for_each(col_b.begin(), col_b.end(), app_b);
        }

        // batch index
        std::vector<Index> bidx_b(col_b.size());
        std::fill(bidx_b.begin(), bidx_b.end(), b + 1);

        std::copy(col_b.begin(), col_b.end(), std::back_inserter(columns));
        std::copy(bidx_b.begin(),
                  bidx_b.end(),
                  std::back_inserter(batch_indexes));

        const Index ub = columns.size();

        /////////////////////////////
        // compute regression stat //
        /////////////////////////////

        mtx_data_t data(mtx_tuple_t(mtx_tuple_t::MTX { mtx_files.at(b) },
                                    mtx_tuple_t::ROW { row_files.at(b) },
                                    mtx_tuple_t::COL { col_files.at(b) },
                                    mtx_tuple_t::IDX { idx_files.at(b) }),
                        options.MAX_ROW_WORD,
                        options.ROW_WORD_SEP);

        Mat r_b_nk, y_b_n;

        CHK_RETL_(run_pmf_stat(data,
                               logBeta_dk,
                               pos2row,
                               options,
                               r_b_nk,
                               y_b_n),
                  "unable to compute topic statistics");

        TLOG_(verbose, "computed topic statistics");

        R_nk.middleRows(lb, r_b_nk.rows()) = r_b_nk;
        Y_n.middleRows(lb, y_b_n.rows()) = y_b_n;

        batch_sizes.emplace_back(r_b_nk.rows());

        TLOG_(verbose,
              "processed  [ " << (b + 1) << " ] / [ " << num_data_batch
                              << " ]");
    }

    if (options.do_stdize_r) {
        asap::util::stretch_matrix_columns_inplace(R_nk);
    }

    std::vector<std::size_t> lb_index;
    std::vector<std::size_t> ub_index;

    {
        std::size_t n_sofar = 0;
        for (Index b = 0; b < num_data_batch; ++b) {
            const std::size_t lb = n_sofar;
            const std::size_t bs = batch_sizes.at(b);
            n_sofar += bs;
            const std::size_t ub = n_sofar;
            lb_index.emplace_back(lb);
            ub_index.emplace_back(ub);
        }
    }

    exp_op<Mat> exp;
    softmax_op_t<Mat> softmax;

    Mat theta_ntot_k(Ntot, K), log_theta_ntot_k(Ntot, K);
    theta_ntot_k.setZero();
    log_theta_ntot_k.setZero();

    if (log_delta.isNull()) {
        logDelta_db.setZero();

        TLOG_(verbose, "Calibrating the loading coefficients (theta)");

        for (Index b = 0; b < num_data_batch; ++b) {
            const std::size_t lb = lb_index.at(b);
            const std::size_t bs = batch_sizes.at(b);

            TLOG_(verbose, "N = " << bs << " cells");

            Mat theta_b, log_theta_b;

            CHECK_(run_pmf_theta(logBeta_dk,
                                 R_nk.middleRows(lb, bs),
                                 Y_n.middleRows(lb, bs),
                                 theta_b,
                                 log_theta_b,
                                 a0,
                                 b0,
                                 max_iter,
                                 options.do_stdize_r,
                                 verbose),
                   "unable to calibrate theta values");

            theta_ntot_k.middleRows(lb, bs) = theta_b;
            log_theta_ntot_k.middleRows(lb, bs) = log_theta_b;
        }

    } else if (log_delta.isNotNull()) {

        logDelta_db = Rcpp::as<Mat>(log_delta);
        const Index B = logDelta_db.cols();

        // 1. Estimate correlation induced by delta
        TLOG_(verbose, "Estimate correlations with " << B << " delta vars.");
        Mat R0_nb = Mat::Zero(Ntot, B);

        for (Index b = 0; b < num_data_batch; ++b) {

            /////////////////////////////
            // compute regression stat //
            /////////////////////////////

            mtx_data_t data(mtx_tuple_t(mtx_tuple_t::MTX { mtx_files.at(b) },
                                        mtx_tuple_t::ROW { row_files.at(b) },
                                        mtx_tuple_t::COL { col_files.at(b) },
                                        mtx_tuple_t::IDX { idx_files.at(b) }),
                            options.MAX_ROW_WORD,
                            options.ROW_WORD_SEP);

            const std::size_t lb = lb_index.at(b);
            const std::size_t bs = batch_sizes.at(b);

            {
                mm_info_reader_t info;
                CHECK(peek_bgzf_header(mtx_files.at(b), info));
                TLOG_(verbose,
                      "N " << bs << " cells in data "
                           << " [" << info.max_row << " x " << info.max_col
                           << "]");
            }

            Mat r0_b_nb, y0_b_n;

            CHK_RETL_(run_pmf_stat(data,
                                   logDelta_db,
                                   pos2row,
                                   options,
                                   r0_b_nb,
                                   y0_b_n),
                      "unable to compute topic statistics");

            TLOG_(verbose,
                  "computed batch statistics: [" << lb << " .. " << (bs + lb)
                                                 << "]");

            R0_nb.middleRows(lb, bs) = r0_b_nb;

            TLOG_(verbose,
                  "step 2: processed  [ " << (b + 1) << " ] / [ "
                                          << num_data_batch << " ]");
        }

        // 2. Take residuals
        TLOG_(verbose, "Regress out the batch effect correlations");
        residual_columns_inplace(R_nk, R0_nb, 1e-4, verbose);

        if (options.do_stdize_r) {
            asap::util::stretch_matrix_columns_inplace(R_nk);
        }

        // 3. Estimate theta based on the new R_nk
        TLOG_(verbose, "Calibrating the loading coefficients (theta)");

        for (Index b = 0; b < num_data_batch; ++b) {
            const std::size_t lb = lb_index.at(b);
            const std::size_t bs = batch_sizes.at(b);

            Mat theta_b, log_theta_b;

            CHECK_(run_pmf_theta(logBeta_dk,
                                 R_nk.middleRows(lb, bs),
                                 Y_n.middleRows(lb, bs),
                                 theta_b,
                                 log_theta_b,
                                 a0,
                                 b0,
                                 max_iter,
                                 options.do_stdize_r,
                                 verbose),
                   "unable to calibrate theta values");

            theta_ntot_k.middleRows(lb, bs) = theta_b;
            log_theta_ntot_k.middleRows(lb, bs) = log_theta_b;

            TLOG_(verbose,
                  "Step 3: processed  [ " << (b + 1) << " ] / [ "
                                          << num_data_batch << " ]");
        }
    }

    TLOG_(verbose, "Got em all");

    using namespace rcpp::util;
    using namespace Rcpp;

    const std::vector<std::string> &d_ = pos2row;
    std::vector<std::string> k_;
    for (std::size_t k = 1; k <= K; ++k) {
        k_.push_back(std::to_string(k));
    }

    TLOG_(verbose, "Done");

    return List::create(_["beta"] = named(logBeta_dk.unaryExpr(exp), d_, k_),
                        _["corr"] = named(R_nk, columns, k_),
                        _["theta"] = named(theta_ntot_k, columns, k_),
                        _["log.theta"] = named(log_theta_ntot_k, columns, k_),
                        _["colsum"] = named_rows(Y_n, columns),
                        _["batch.names"] = _batch_names,
                        _["batch.index"] = batch_indexes,
                        _["rownames"] = pos2row,
                        _["colnames"] = columns);
}
