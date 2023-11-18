#include "rcpp_asap_interaction_regression.hh"

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

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    std::vector<std::string> pos2row;

    rcpp::util::copy(x_row_names, pos2row);

    //////////////////////////////////////
    // take care of different row names //
    //////////////////////////////////////

    const Index D = pos2row.size(); // dimensionality

    ASSERT_RETL(log_x.rows() == D,
                "#Rows in the log_x matrix !=  the size of x_row_names: "
                    << log_x.rows() << " != " << D);

    std::unordered_map<std::string, Index> row2pos;
    for (Index r = 0; r < pos2row.size(); ++r) {
        row2pos[pos2row.at(r)] = r;
    }

    ASSERT_RETL(row2pos.size() == D, "Redundant row names exist");
    TLOG_(verbose, "Found " << row2pos.size() << " unique row names");

    ///////////////////////////////
    // read mtx data information //
    ///////////////////////////////

    CHK_RETL_(convert_bgzip(mtx_file),
              "mtx file " << mtx_file << " was not bgzipped.");

    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    const Index N = info.max_col;        // number of cells
    const Index K = log_x.cols();        // number of topics
    const Index block_size = BLOCK_SIZE; // memory block size

    std::vector<std::string> coln;

    CHK_RETL_(read_line_file(col_file, coln, MAX_COL_WORD, COL_WORD_SEP),
              "Failed to read the column name file: " << col_file);

    ASSERT_RETL(N == coln.size(),
                "Different #columns: " << N << " vs. " << coln.size());

    mtx_data_t data(mtx_data_t::MTX { mtx_file },
                    mtx_data_t::ROW { row_file },
                    mtx_data_t::IDX { idx_file },
                    MAX_ROW_WORD,
                    ROW_WORD_SEP);

    /////////////////////////////////////////////
    // Build weighted kNN graph from the input //
    /////////////////////////////////////////////

    SpMat W;
    CHK_RETL_(build_knn_graph(knn_src, knn_tgt, knn_weight, N, W),
              "Failed to build the kNN graph");

    ////////////////////////////////
    // Take sufficient statistics //
    ////////////////////////////////

    Mat logX_dk = log_x;
    TLOG_(verbose, "lnX: " << logX_dk.rows() << " x " << logX_dk.cols());

    const Index Nedge = W.nonZeros();

    Mat Rtot_nk(Nedge, K);
    Mat Ytot_n(Nedge, 1);
    std::vector<Index> src_out(Nedge);
    std::vector<Index> tgt_out(Nedge);

    if (verbose) {
        Rcpp::Rcerr << "Calibrating " << Nedge << " columns..." << std::endl;
    }

    data.relocate_rows(row2pos);

    at_least_one_op<Mat> at_least_one;
    at_least_zero_op<Mat> at_least_zero;
    exp_op<Mat> exp;
    log1p_op<Mat> log1p;

    Index Nprocessed = 0;
    const Scalar one = 1.;
#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index i = 0; i < W.outerSize(); ++i) {
        SpMat y_i = data.read_reloc(i, i + 1);
        for (SpMat::InnerIterator jt(W, i); jt; ++jt) {
            const Index j = jt.col();
            const Scalar w_ij = jt.value();
            SpMat y_j = data.read_reloc(j, j + 1);

            // yij <- (yi + yj) * wij (D x 1)
            Mat wy_ij;
            if (do_log1p) {
                wy_ij = w_ij * (y_i.unaryExpr(log1p) + y_j.unaryExpr(log1p));
            } else {
                wy_ij = w_ij * (y_i + y_j);
            }
#pragma omp critical
            {
                const Scalar denom = wy_ij.sum();
                Rtot_nk.row(Nprocessed) =
                    (wy_ij.transpose() * logX_dk) / std::max(denom, one);
                Ytot_n(Nprocessed, 0) = denom;
                src_out[Nprocessed] = i + 1; // 1-based
                tgt_out[Nprocessed] = j + 1; // 1-based

                ++Nprocessed;
                if (verbose) {
                    Rcpp::Rcerr << "\rProcessed: " << Nprocessed << std::flush;
                } else {
                    Rcpp::Rcerr << "+ " << std::flush;
                    if (Nprocessed % 1000 == 0)
                        Rcpp::Rcerr << "\r" << std::flush;
                }
            } // end of omp critical
        }
    }

    if (!verbose)
        Rcpp::Rcerr << std::endl;
    TLOG_(verbose, "Done");

    using namespace rcpp::util;
    using namespace Rcpp;

    std::vector<std::string> d_ = pos2row;
    std::vector<std::string> k_;
    for (std::size_t k = 1; k <= K; ++k) {
        k_.push_back(std::to_string(k));
    }

    return List::create(_["beta"] = named(logX_dk.unaryExpr(exp), d_, k_),
                        _["corr"] = Rtot_nk,
                        _["colsum"] = Ytot_n,
                        _["src.index"] = src_out,
                        _["tgt.index"] = tgt_out,
                        _["rownames"] = pos2row,
                        _["colnames"] = coln);
}
