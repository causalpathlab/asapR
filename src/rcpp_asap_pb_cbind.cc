#include "rcpp_asap_pb_cbind.hh"

//' Generate approximate pseudo-bulk data by random projections
//' while sharing rows/features across multiple data sets.
//' Horizontal concatenation.
//'
//' @param y_dn_vec a list of sparse matrices
//' @param num_factors a desired number of random factors
//' @param take_union_rows take union of rows (default: FALSE)
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param CELL_NORM normalization constant per each data point
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_batch_adj (default: FALSE)
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_down_sample down-sampling (default: TRUE)
//' @param save_aux_data save auxiliary data (default: FALSE)
//' @param weighted_rand_proj save random projection (default: FALSE)
//' @param KNN_CELL k-NN cells per batch between different batches (default: 10)
//' @param CELL_PER_SAMPLE down-sampling cell per sample (default: 100)
//' @param BATCH_ADJ_ITER batch Adjustment steps (default: 100)
//' @param a0 gamma(a0, b0) (default: 1e-8)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param MAX_ROW_WORD maximum words per line in `row_files[i]`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @param MAX_COL_WORD maximum words per line in `col_files[i]`
//' @param COL_WORD_SEP word separation character to replace white space
//'
//' @return a list
//' \itemize{
//' \item `PB` pseudobulk (average) data (feature x sample)
//' \item `sum` pseudobulk (sum) data (feature x sample)
//' \item `matched.sum` kNN-matched pseudobulk data (feature x sample)
//' \item `sum_db` batch-specific sum (feature x batch)
//' \item `size` size per sample (sample x 1)
//' \item `prob_bs` batch-specific frequency (batch x sample)
//' \item `size_bs` batch-specific size (batch x sample)
//' \item `batch.effect` batch effect (feature x batch)
//' \item `log.batch.effect` log batch effect (feature x batch)
//' \item `batch.names` batch names (batch x 1)
//' \item `positions` pseudobulk sample positions (cell x 1)
//' \item `rand.dict` random dictionary (proj factor x feature)
//' \item `rand.proj` random projection results (sample x proj factor)
//' \item `colnames` column (cell) names
//' \item `rownames` feature (gene) names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_random_bulk_cbind(
    const std::vector<Eigen::SparseMatrix<float>> y_dn_vec,
    const std::size_t num_factors,
    const Rcpp::Nullable<Rcpp::StringVector> r_row_names = R_NilValue,
    const Rcpp::Nullable<Rcpp::StringVector> r_batch_names = R_NilValue,
    const std::size_t rseed = 42,
    const bool verbose = true,
    const std::size_t NUM_THREADS = 0,
    const double CELL_NORM = 1e4,
    const std::size_t BLOCK_SIZE = 1000,
    const bool do_batch_adj = true,
    const bool do_log1p = false,
    const bool do_down_sample = true,
    const bool save_aux_data = false,
    const std::size_t KNN_CELL = 10,
    const std::size_t CELL_PER_SAMPLE = 100,
    const std::size_t BATCH_ADJ_ITER = 100,
    const double a0 = 1,
    const double b0 = 1)
{
    using namespace asap::pb;
    log1p_op<Mat> log1p;

    const Index B = y_dn_vec.size();
    ASSERT_RETL(B > 0, "at least 1 batch is needed");

    auto count_add_cols = [](Index a, auto &y) -> Index {
        return a + y.cols();
    };

    const Index Ntot =
        std::accumulate(y_dn_vec.begin(), y_dn_vec.end(), 0, count_add_cols);

    const Index K = num_factors; // tree depths in implicit bisection
    Index D = 0;                 // dimensionality

    for (const auto &y_dn : y_dn_vec) {
        if (D == 0) {
            D = y_dn.rows();
        } else {
            ASSERT_RETL(y_dn.rows() == D,
                        "Found inconsistent # rows: "
                            << "the previous data: " << D << " vs. "
                            << "this data:" << y_dn.rows());
        }
    }

    TLOG_(verbose, Ntot << " columns");

    std::vector<std::string> batch_names, pos2row;

    if (r_batch_names.isNotNull()) {
        rcpp::util::copy(Rcpp::StringVector(r_batch_names), batch_names);
    } else {
        for (Index b = 0; b < B; ++b) {
            batch_names.emplace_back(std::to_string(b + 1));
        }
    }

    if (r_row_names.isNotNull()) {
        rcpp::util::copy(Rcpp::StringVector(r_row_names), pos2row);
    } else {
        for (Index r = 0; r < D; ++r) {
            pos2row.emplace_back(std::to_string(r + 1));
        }
    }

    std::unordered_map<std::string, Index> row2pos;
    for (Index r = 0; r < pos2row.size(); ++r) {
        row2pos[pos2row.at(r)] = r;
    }

    ASSERT_RETL(batch_names.size() == B, "check the r_batch_names");
    ASSERT_RETL(pos2row.size() == D, "check the rows_restrict");

    std::vector<std::string> columns;
    columns.reserve(Ntot);

    std::vector<eigenSparse_data_t> data_loaders;

    for (Index b = 0; b < B; ++b) {
        data_loaders.emplace_back(eigenSparse_data_t(y_dn_vec.at(b), pos2row));
    }

    for (Index b = 0; b < B; ++b) {
        data_loaders.at(b).relocate_rows(row2pos);
    }

    // assign unique column names
    {
        for (Index b = 0; b < B; ++b) {
            const std::string bname = batch_names.at(b);
            for (Index j = 0; j < data_loaders.at(b).max_col(); ++j) {
                columns.emplace_back(std::to_string(j + 1) + "_" + bname);
            }
        }
    }

    /////////////////////////
    // run the PB routines //
    /////////////////////////

    options_t options;

    options.K = num_factors;

    options.do_batch_adj = do_batch_adj;
    options.do_log1p = do_log1p;
    options.do_down_sample = do_down_sample;
    options.save_aux_data = save_aux_data;
    options.KNN_CELL = KNN_CELL;
    options.CELL_PER_SAMPLE = CELL_PER_SAMPLE;
    options.BATCH_ADJ_ITER = BATCH_ADJ_ITER;
    options.a0 = a0;
    options.b0 = b0;
    options.rseed = rseed;
    options.verbose = verbose;
    options.NUM_THREADS = NUM_THREADS;
    options.CELL_NORM = CELL_NORM;
    options.BLOCK_SIZE = BLOCK_SIZE;

    return run_asap_pb_cbind(data_loaders,
                             pos2row,
                             columns,
                             batch_names,
                             options);
}

//' Generate approximate pseudo-bulk data by random projections
//' while sharing rows/features across multiple data sets.
//' Horizontal concatenation.
//'
//' @param mtx_files matrix-market-formatted data files (bgzip)
//' @param row_files row names (gene/feature names)
//' @param col_files column names (cell/column names)
//' @param idx_files matrix-market colum index files
//' @param num_factors a desired number of random factors
//' @param take_union_rows take union of rows (default: FALSE)
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param CELL_NORM normalization constant per each data point
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_batch_adj (default: FALSE)
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_down_sample down-sampling (default: TRUE)
//' @param save_aux_data save random projection (default: FALSE)
//' @param KNN_CELL k-NN cells per batch between different batches (default: 10)
//' @param CELL_PER_SAMPLE down-sampling cell per sample (default: 100)
//' @param BATCH_ADJ_ITER batch Adjustment steps (default: 100)
//' @param a0 gamma(a0, b0) (default: 1e-8)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param MAX_ROW_WORD maximum words per line in `row_files[i]`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @param MAX_COL_WORD maximum words per line in `col_files[i]`
//' @param COL_WORD_SEP word separation character to replace white space
//'
//' @return a list
//' \itemize{
//' \item `PB` pseudobulk (average) data (feature x sample)
//' \item `sum` pseudobulk (sum) data (feature x sample)
//' \item `matched.sum` kNN-matched pseudobulk data (feature x sample)
//' \item `sum_db` batch-specific sum (feature x batch)
//' \item `size` size per sample (sample x 1)
//' \item `prob_bs` batch-specific frequency (batch x sample)
//' \item `size_bs` batch-specific size (batch x sample)
//' \item `batch.effect` batch effect (feature x batch)
//' \item `log.batch.effect` log batch effect (feature x batch)
//' \item `batch.names` batch names (batch x 1)
//' \item `positions` pseudobulk sample positions (cell x 1)
//' \item `rand.dict` random dictionary (proj factor x feature)
//' \item `rand.proj` random projection results (sample x proj factor)
//' \item `colnames` column (cell) names
//' \item `rownames` feature (gene) names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_random_bulk_cbind_mtx(
    const std::vector<std::string> mtx_files,
    const std::vector<std::string> row_files,
    const std::vector<std::string> col_files,
    const std::vector<std::string> idx_files,
    const std::size_t num_factors,
    const Rcpp::Nullable<Rcpp::StringVector> r_batch_names = R_NilValue,
    const Rcpp::Nullable<Rcpp::StringVector> rows_restrict = R_NilValue,
    const bool rename_columns = true,
    const bool take_union_rows = false,
    const std::size_t rseed = 42,
    const bool verbose = true,
    const std::size_t NUM_THREADS = 0,
    const double CELL_NORM = 1e4,
    const std::size_t BLOCK_SIZE = 1000,
    const bool do_batch_adj = true,
    const bool do_log1p = false,
    const bool do_down_sample = true,
    const bool save_aux_data = false,
    const std::size_t KNN_CELL = 10,
    const std::size_t CELL_PER_SAMPLE = 100,
    const std::size_t BATCH_ADJ_ITER = 100,
    const double a0 = 1,
    const double b0 = 1,
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@')
{

    using namespace asap::pb;

    options_t options;

    options.K = num_factors;

    options.do_batch_adj = do_batch_adj;
    options.do_log1p = do_log1p;
    options.do_down_sample = do_down_sample;
    options.save_aux_data = save_aux_data;
    options.KNN_CELL = KNN_CELL;
    options.CELL_PER_SAMPLE = CELL_PER_SAMPLE;
    options.BATCH_ADJ_ITER = BATCH_ADJ_ITER;
    options.a0 = a0;
    options.b0 = b0;
    options.rseed = rseed;
    options.verbose = verbose;
    options.NUM_THREADS = NUM_THREADS;
    options.CELL_NORM = CELL_NORM;
    options.BLOCK_SIZE = BLOCK_SIZE;

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
            rcpp::util::copy(Rcpp::StringVector(r_batch_names), batch_names);
        } else {
            for (Index b = 0; b < B; ++b) {
                batch_names.emplace_back(std::to_string(b + 1));
            }
        }

        if (verbose) {
            for (auto b : batch_names)
                TLOG("batch: " << b)
        }

        ASSERT_RETL(batch_names.size() == B, "check the r_batch_names");
    }

    TLOG_(verbose, "Checked the files");

    //////////////////////////////////
    // first figure out global rows //
    //////////////////////////////////

    std::vector<std::string> pos2row;
    std::unordered_map<std::string, Index> row2pos;

    if (rows_restrict.isNotNull()) {

        rcpp::util::copy(Rcpp::StringVector(rows_restrict), pos2row);
        make_position_dict(pos2row, row2pos);

    } else {

        rcpp::util::take_common_names(row_files,
                                      pos2row,
                                      row2pos,
                                      take_union_rows,
                                      MAX_ROW_WORD,
                                      ROW_WORD_SEP);

        TLOG_(verbose, "Found " << row2pos.size() << " row names");
    }

    ASSERT_RETL(pos2row.size() > 0, "Empty row names!");

    auto count_add_cols = [](Index a, std::string mtx) -> Index {
        mm_info_reader_t info;
        CHECK(peek_bgzf_header(mtx, info));
        return a + info.max_col;
    };
    const Index Ntot =
        std::accumulate(mtx_files.begin(), mtx_files.end(), 0, count_add_cols);

    std::vector<std::string> columns;
    columns.reserve(Ntot);

    for (Index b = 0; b < B; ++b) {
        std::vector<std::string> col_b;
        CHK_RETL_(read_line_file(col_files.at(b),
                                 col_b,
                                 MAX_COL_WORD,
                                 COL_WORD_SEP),
                  "unable to read " << col_files.at(b))
        if (rename_columns) {
            const std::string bname = batch_names.at(b);
            auto app_b = [&bname](std::string &x) { x += "_" + bname; };
            std::for_each(col_b.begin(), col_b.end(), app_b);
        }

        std::copy(col_b.begin(), col_b.end(), std::back_inserter(columns));
    }

    //////////////////////////////////
    // build a list of data loaders //
    //////////////////////////////////

    std::vector<mtx_data_t> data_loaders;

    for (Index b = 0; b < B; ++b) {

        const mtx_tuple_t tup(mtx_tuple_t::MTX { mtx_files.at(b) },
                              mtx_tuple_t::ROW { row_files.at(b) },
                              mtx_tuple_t::COL { col_files.at(b) },
                              mtx_tuple_t::IDX { idx_files.at(b) });

        data_loaders.emplace_back(
            mtx_data_t { tup, MAX_ROW_WORD, ROW_WORD_SEP });
    }

    for (Index b = 0; b < B; ++b) {
        data_loaders.at(b).relocate_rows(row2pos);
    }

    return run_asap_pb_cbind(data_loaders,
                             pos2row,
                             columns,
                             batch_names,
                             options);
}
