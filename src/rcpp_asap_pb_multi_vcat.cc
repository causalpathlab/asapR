#include "rcpp_asap_pb_multi_vcat.hh"

//' Generate approximate pseudo-bulk data by random projections
//' while sharing columns/cells across multiple data sets.
//' Vertical concatenation.
//'
//' @param mtx_files matrix-market-formatted data files (bgzip)
//' @param row_files row names (gene/feature names)
//' @param col_files column names (cell/column names)
//' @param idx_files matrix-market colum index files
//' @param num_factors a desired number of random factors per data set
//'
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_batch_adj (default: FALSE)
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_down_sample down-sampling (default: FALSE)
//' @param save_rand_proj save random projection (default: FALSE)
//' @param KNN_CELL k-NN cells per batch between different batches (default: 3)
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
asap_random_bulk_multi_vcat(const std::vector<std::string> mtx_files,
                            const std::vector<std::string> row_files,
                            const std::vector<std::string> col_files,
                            const std::vector<std::string> idx_files,
                            const std::size_t num_factors,
                            const std::size_t rseed = 42,
                            const bool verbose = true,
                            const std::size_t NUM_THREADS = 1,
                            const std::size_t BLOCK_SIZE = 100,
                            const bool do_batch_adj = true,
                            const bool do_log1p = false,
                            const bool do_down_sample = false,
                            const bool save_rand_proj = false,
                            const std::size_t CELL_PER_SAMPLE = 100,
                            const std::size_t BATCH_ADJ_ITER = 100,
                            const double a0 = 1e-8,
                            const double b0 = 1,
                            const std::size_t MAX_ROW_WORD = 2,
                            const char ROW_WORD_SEP = '_',
                            const std::size_t MAX_COL_WORD = 100,
                            const char COL_WORD_SEP = '@')
{

    log1p_op<Mat> log1p;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    const Index B = mtx_files.size();

    ASSERT_RETL(B > 0, "Empty mtx file names");
    ASSERT_RETL(row_files.size() == B, "Need a row file for each data type");
    ASSERT_RETL(col_files.size() == B, "Need a col file for each data type");

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

    TLOG_(verbose, "Checked the files");

    //////////////////////////////////////////////////////
    // First figure out the intersection of the columns //
    //////////////////////////////////////////////////////

    std::vector<std::string> columns;
    std::unordered_map<std::string, Index> col2pos;

    rcpp::util::take_common_names(col_files,
                                  columns,
                                  col2pos,
                                  false,
                                  MAX_COL_WORD,
                                  COL_WORD_SEP);

    TLOG_(verbose, "Found " << col2pos.size() << " column names");

    ASSERT_RETL(columns.size() > 0, "Empty column names!");

    const Index block_size = BLOCK_SIZE;
    const Index Ntot = columns.size();
    const Index K = num_factors; // tree depths in implicit bisection

    ////////////////////////////////////////////////////////////////
    // Step 1. sample random projection matrix for each data type //
    ////////////////////////////////////////////////////////////////



    return Rcpp::List::create();
}
