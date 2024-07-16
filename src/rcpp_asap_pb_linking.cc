#include "rcpp_asap_pb_linking.hh"

//' Generate approximate pseudo-bulk data by random projections
//' while linking features across multiple mtx files
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
//' @param CELL_NORM normalization constant per each data point
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_down_sample down-sampling (default: FALSE)
//' @param save_rand_proj save random projection (default: FALSE)
//' @param weighted_rand_proj save random projection (default: FALSE)
//' @param CELL_PER_SAMPLE down-sampling cell per sample (default: 100)
//' @param a0 gamma(a0, b0) (default: 1e-8)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param MAX_ROW_WORD maximum words per line in `row_files[i]`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @param MAX_COL_WORD maximum words per line in `col_files[i]`
//' @param COL_WORD_SEP word separation character to replace white space
//'
//' @return a list
//' \itemize{
//' \item `PB.list` pseudobulk (average) data (feature x sample) for each type
//' \item `sum.list` pseudobulk (sum) data (feature x sample) for each type
//' \item `size.list` size per sample (sample x 1) for each type
//' \item `rownames.list` feature (gene) names for each type
//' \item `colnames` column (cell) names across data types
//' \item `positions` pseudobulk sample positions (cell x 1)
//' \item `rand.proj` random projection results (sample x proj factor)
//' \item `colnames` column (cell) names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_random_bulk_linking_mtx(const std::vector<std::string> mtx_files,
                             const std::vector<std::string> row_files,
                             const std::vector<std::string> col_files,
                             const std::vector<std::string> idx_files,
                             const std::size_t num_factors,
                             const std::size_t rseed = 42,
                             const bool verbose = true,
                             const std::size_t NUM_THREADS = 1,
                             const double CELL_NORM = 1e4,
                             const std::size_t BLOCK_SIZE = 1000,
                             const bool do_log1p = false,
                             const bool do_down_sample = false,
                             const bool save_rand_proj = false,
                             const bool weighted_rand_proj = false,
                             const std::size_t CELL_PER_SAMPLE = 100,
                             const double a0 = 1e-8,
                             const double b0 = 1,
                             const std::size_t MAX_ROW_WORD = 2,
                             const char ROW_WORD_SEP = '_',
                             const std::size_t MAX_COL_WORD = 100,
                             const char COL_WORD_SEP = '@')
{

    using namespace asap::pb;

    options_t options;

    options.K = num_factors;

    // options.do_batch_adj = do_batch_adj;
    // options.BATCH_ADJ_ITER = BATCH_ADJ_ITER;
    // options.save_aux_data = save_aux_data;
    // options.KNN_CELL = KNN_CELL;

    options.do_log1p = do_log1p;
    options.do_down_sample = do_down_sample;

    options.CELL_PER_SAMPLE = CELL_PER_SAMPLE;
    options.a0 = a0;
    options.b0 = b0;
    options.rseed = rseed;
    options.verbose = verbose;
    options.NUM_THREADS = NUM_THREADS;
    options.CELL_NORM = CELL_NORM;
    options.BLOCK_SIZE = BLOCK_SIZE;

    const Index B = mtx_files.size();

    ASSERT_RETL(B > 0, "Empty mtx file names");
    ASSERT_RETL(row_files.size() == B, "Need a row file for each data type");
    ASSERT_RETL(col_files.size() == B, "Need a col file for each data type");

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



    return Rcpp::List::create();
}
