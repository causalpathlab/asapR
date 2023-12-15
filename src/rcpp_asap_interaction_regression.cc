#include "rcpp_asap_interaction_regression.hh"

//' Topic statistics to estimate factor loading
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param row_file row names file (D x 1)
//' @param col_file column names file (N x 1)
//' @param idx_file matrix-market colum index file
//' @param log_beta D x K log dictionary/design matrix
//' @param beta_row_names row names log_beta (D vector)
//' @param W_nn_list list(src.index, tgt.index, [weights]) for columns
//'
//' @param A_dd_list list(src.index, tgt.index, [weights]) for features
//'
//' @param do_stdize_beta use standardized log_beta (default: TRUE)
//' @param do_product yi * yj for interaction (default: FALSE)
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param MAX_ROW_WORD maximum words per line in `row_files[i]`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @param MAX_COL_WORD maximum words per line in `col_files[i]`
//' @param COL_WORD_SEP word separation character to replace white space
//' @param verbose verbosity
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
asap_interaction_topic_stat(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    const std::string idx_file,
    const Eigen::MatrixXf log_beta,
    const Rcpp::StringVector beta_row_names,
    const Rcpp::List W_nn_list,
    const Rcpp::Nullable<Rcpp::List> A_dd_list = R_NilValue,
    const bool do_stdize_beta = true,
    const bool do_product = false,
    const std::size_t NUM_THREADS = 1,
    const std::size_t BLOCK_SIZE = 100,
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@',
    const bool verbose = false)
{

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    std::vector<std::string> pos2row;

    rcpp::util::copy(beta_row_names, pos2row);

    //////////////////////////////////////
    // take care of different row names //
    //////////////////////////////////////

    const Index D = pos2row.size(); // dimensionality

    ASSERT_RETL(log_beta.rows() == D,
                "#Rows in the log_beta matrix !=  the size of beta_row_names: "
                    << log_beta.rows() << " != " << D);

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
    const Index K = log_beta.cols();     // number of topics
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

    SpMat W_nn;
    rcpp::util::build_sparse_mat(W_nn_list, N, N, W_nn);

    SpMat A_dd;
    if (A_dd_list.isNotNull()) {
        rcpp::util::build_sparse_mat(Rcpp::List(A_dd_list), D, D, A_dd);
        TLOG_(verbose, "Row Network: " << A_dd.rows() << " x " << A_dd.cols());
    } else {
        build_diagonal(D, 1., A_dd);
    }

    ////////////////////////////////
    // Take sufficient statistics //
    ////////////////////////////////

    Mat logX_dk = log_beta;
    if (do_stdize_beta) {
        standardize_columns_inplace(logX_dk);
    }
    TLOG_(verbose, "lnX: " << logX_dk.rows() << " x " << logX_dk.cols());

    const Index Nedge = W_nn.nonZeros();

    Mat Rtot_mk = Mat::Zero(Nedge, K);
    Mat Ytot_m = Mat::Zero(Nedge, 1);
    std::vector<Index> src_out(Nedge);
    std::vector<Index> tgt_out(Nedge);
    std::vector<Scalar> weight_out(Nedge);

    if (verbose) {
        Rcpp::Rcerr << "Calibrating " << Nedge << " columns..." << std::endl;
    }

    data.relocate_rows(row2pos);

    Index Nprocessed = 0;
    const Scalar one = 1.;
#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index i = 0; i < W_nn.outerSize(); ++i) {
        SpMat y_di = data.read_reloc(i, i + 1);
        for (SpMat::InnerIterator jt(W_nn, i); jt; ++jt) {
            const Index j = jt.index();
            SpMat y_dj = data.read_reloc(j, j + 1);

            const Scalar w_ij = jt.value() * product_similarity(y_di, y_dj);

            // yij <- (yi + yj) * wij (D x 1)
            Mat wy_dij(D, 1);
            if (do_product) {
                wy_dij = w_ij * y_di.cwiseProduct(y_dj);
            } else {
                wy_dij = w_ij * 0.5 * (A_dd * y_di + y_dj + y_di + A_dd * y_dj);
            }
#pragma omp critical
            {
                const Scalar denom = std::max(wy_dij.sum(), one);
                Rtot_mk.row(Nprocessed) =
                    (wy_dij.transpose() * logX_dk) / denom;
                Ytot_m(Nprocessed, 0) = denom;
                src_out[Nprocessed] = i + 1;   // 1-based
                tgt_out[Nprocessed] = j + 1;   // 1-based
                weight_out[Nprocessed] = w_ij; // readjusted weight

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

    if (!verbose) {
        Rcpp::Rcerr << std::endl;
    } else {
        TLOG("Done");
    }

    using namespace rcpp::util;
    using namespace Rcpp;

    std::vector<std::string> d_ = pos2row;
    std::vector<std::string> k_;
    for (std::size_t k = 1; k <= K; ++k) {
        k_.push_back(std::to_string(k));
    }

    exp_op<Mat> exp;

    return List::create(_["beta"] = named(logX_dk.unaryExpr(exp), d_, k_),
                        _["corr"] = Rtot_mk,
                        _["colsum"] = Ytot_m,
                        _["src.index"] = src_out,
                        _["tgt.index"] = tgt_out,
                        _["weight"] = weight_out,
                        _["rownames"] = pos2row,
                        _["colnames"] = coln);
}
