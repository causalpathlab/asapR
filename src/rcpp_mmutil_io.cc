#include "mmutil.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "mmutil_io.hh"
#include "tuple_util.hh"

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>

using namespace mmutil::io;
using namespace mmutil::bgzf;

using triplet_reader_t = eigen_triplet_reader_remapped_cols_t;
using triplet_subrow_reader_t = eigen_triplet_reader_remapped_rows_cols_t;
using Index = triplet_reader_t::index_t;

//' Just read the header information
//'
//' @param mtx_file data file
//'
//' @return info
//'
// [[Rcpp::export]]
Rcpp::List
mmutil_info(const std::string mtx_file)
{
    using namespace mmutil::io;
    using namespace mmutil::bgzf;

    CHK_RETL(convert_bgzip(mtx_file));

    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the mtx file:" << mtx_file);

    return Rcpp::List::create(Rcpp::_["max.row"] = info.max_row,
                              Rcpp::_["max.col"] = info.max_col,
                              Rcpp::_["max.elem"] = info.max_elem);
}

///////////////////////////////
// Writing out sparse matrix //
///////////////////////////////

//' Write down sparse matrix to the disk
//' @param X sparse matrix
//' @param mtx_file file name
//'
//' @return EXIT_SUCCESS or EXIT_FAILURE
// [[Rcpp::export]]
int
mmutil_write_mtx(const Eigen::SparseMatrix<float, Eigen::ColMajor> &X,
                 const std::string mtx_file)
{
    const std::string idx_file = mtx_file + ".index";

    const Index max_row = X.rows();
    const Index max_col = X.cols();
    const Index max_elem = X.nonZeros();
    const char FS = ' ';

    obgzf_stream ofs;
    ofs.open(mtx_file.c_str(), std::ios::out);

    ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
    ofs << max_row << FS << max_col << FS << max_elem << std::endl;

    TLOG("Start writing down " << max_row << " x " << max_col);

    using SpMat = Eigen::SparseMatrix<float, Eigen::ColMajor>;
    for (Index j = 0; j < X.outerSize(); ++j) {
        for (SpMat::InnerIterator it(X, j); it; ++it) {
            const Index ii = it.row() + 1; // fix zero-based to one-based
            const Index jj = j + 1;        // fix zero-based to one-based
            const Scalar weight = it.value();
            ofs << ii << FS << jj << FS << weight << std::endl;
        }
    }

    ofs.close();

    if (file_exists(idx_file)) {
        WLOG("Rewrite the existing index file: " << idx_file);
        rename_file(idx_file, idx_file + ".old");
    }

    CHK_RET_(build_mmutil_index(mtx_file, idx_file),
             "Unable to create the index file");

    return EXIT_SUCCESS;
}

////////////////////////////
// Reading random columns //
////////////////////////////
template <typename VEC1, typename VEC2>
int _mmutil_read_columns_triplets(const std::string mtx_file,
                                  const VEC1 &memory_location,
                                  const VEC2 &r_column_index,
                                  triplet_reader_t::TripletVec &Tvec,
                                  Index &max_row,
                                  Index &max_col,
                                  const bool verbose);

/////////////////////////////////////
// Reading random rows and columns //
/////////////////////////////////////
template <typename VEC1, typename VEC2, typename VEC3>
int
_mmutil_read_rows_columns_triplets(const std::string mtx_file,
                                   const VEC1 &memory_location,
                                   const VEC2 &r_row_index,
                                   const VEC3 &r_column_index,
                                   triplet_subrow_reader_t::TripletVec &Tvec,
                                   Index &max_row,
                                   Index &max_col,
                                   const bool verbose);

//' Read a subset of columns from the data matrix
//' @param mtx_file data file
//' @param memory_location column -> memory location
//' @param r_column_index column indexes to retrieve (1-based)
//'
//' @return lists of rows, columns, values
//'
// [[Rcpp::export]]
Rcpp::List
mmutil_read_columns_sparse(const std::string mtx_file,
                           const Rcpp::NumericVector &memory_location,
                           const Rcpp::NumericVector &r_column_index,
                           const bool verbose = false)
{
    Index max_row = 0, max_col = 0;
    triplet_reader_t::TripletVec Tvec;

    const std::size_t N = r_column_index.size();

    CHK_RETL_(_mmutil_read_columns_triplets(mtx_file,
                                            memory_location,
                                            r_column_index,
                                            Tvec,
                                            max_row,
                                            max_col,
                                            verbose),
              "Failed to read the triplets");

    /////////////////////////////////////////
    // populate items in the return matrix //
    /////////////////////////////////////////
    const std::size_t nout = Tvec.size();

    if (nout < 1)
        return Rcpp::List::create();

    Rcpp::IntegerVector row_index(nout, NA_INTEGER);
    Rcpp::IntegerVector col_index(nout, NA_INTEGER);
    Rcpp::NumericVector val_vec(nout, NA_REAL);

    for (std::size_t i = 0; i < nout; ++i) {
        row_index[i] = Tvec.at(i).row() + 1; // convert to 1-based
        col_index[i] = Tvec.at(i).col() + 1; // convert to 1-based
        val_vec[i] = Tvec.at(i).value();
    }

    return Rcpp::List::create(Rcpp::_["row"] = row_index,
                              Rcpp::_["col"] = col_index,
                              Rcpp::_["val"] = val_vec,
                              Rcpp::_["max.row"] = max_row,
                              Rcpp::_["max.col"] = N,
                              Rcpp::_[".true.max.col"] = max_col);
}

//' Read a subset of columns from the data matrix
//' @param mtx_file data file
//' @param memory_location column -> memory location
//' @param r_column_index column indexes to retrieve (1-based)
//'
//' @return a dense sub-matrix
//'
//' @examples
//'
//' rr <- rgamma(100, 1, 1) # one hundred cells
//' mm <- matrix(rgamma(10 * 3, 1, 1), 10, 3)
//' data.hdr <- "test_sim"
//' .files <- asapR::mmutil_simulate_poisson(mm, rr, data.hdr)
//' data.file <- .files$mtx
//' idx.file <- .files$idx
//' mtx.idx <- asapR::mmutil_read_index(idx.file)
//' Y <- as.matrix(Matrix::readMM(data.file))
//' col.pos <- c(1,13,77) # 1-based
//' yy <- asapR::mmutil_read_columns(
//'                  data.file, mtx.idx, col.pos)
//' all(Y[, col.pos, drop = FALSE] == yy)
//' print(head(Y[, col.pos, drop = FALSE]))
//' print(head(yy))
//' unlink(list.files(pattern = data.hdr))
//'
// [[Rcpp::export]]
Rcpp::NumericMatrix
mmutil_read_columns(const std::string mtx_file,
                    const Rcpp::NumericVector &memory_location,
                    const Rcpp::NumericVector &r_column_index,
                    const bool verbose = false)
{

    Index max_row = 0, max_col = 0;
    triplet_reader_t::TripletVec Tvec;

    const std::size_t N = r_column_index.size();

    CHK_RETM_(_mmutil_read_columns_triplets(mtx_file,
                                            memory_location,
                                            r_column_index,
                                            Tvec,
                                            max_row,
                                            max_col,
                                            verbose),
              "Failed to read the triplets");

    if (N > max_col) {
        WLOG("Found " << (N - max_col) << " empty columns");
    }

    SpMat X(max_row, N);
    X.setZero();
    X.reserve(Tvec.size());
    X.setFromTriplets(Tvec.begin(), Tvec.end());

    SEXP xx = Rcpp::wrap(Mat(X));
    Rcpp::NumericMatrix ret(xx);
    return ret;
}

//' Read a subset of rows and columns from the data matrix
//' @param mtx_file data file
//' @param memory_location column -> memory location
//' @param r_row_index row indexes to retrieve (1-based)
//' @param r_column_index column indexes to retrieve (1-based)
//' @param verbose verbosity
//'
//' @return a dense sub-matrix
//'
//' @examples
//'
//' rr <- rgamma(100, 1, 1) # one hundred cells
//' mm <- matrix(rgamma(10 * 3, 1, 1), 10, 3)
//' data.hdr <- "test_sim"
//' .files <- asapR::mmutil_simulate_poisson(mm, rr, data.hdr)
//' data.file <- .files$mtx
//' idx.file <- .files$idx
//' mtx.idx <- asapR::mmutil_read_index(idx.file)
//' Y <- as.matrix(Matrix::readMM(data.file))
//' col.pos <- c(1,13,77) # 1-based
//' row.pos <- 1:10
//' yy <- asapR::mmutil_read_rows_columns(
//'                  data.file, mtx.idx, row.pos, col.pos)
//' all(Y[, col.pos, drop = FALSE] == yy)
//' print(head(Y[, col.pos, drop = FALSE]))
//' print(head(yy))
//' print(tail(yy))
//' unlink(list.files(pattern = data.hdr))
//'
// [[Rcpp::export]]
Rcpp::NumericMatrix
mmutil_read_rows_columns(const std::string mtx_file,
                         const Rcpp::NumericVector &memory_location,
                         const Rcpp::NumericVector &r_row_index,
                         const Rcpp::NumericVector &r_column_index,
                         const bool verbose = false)
{

    Index max_row = 0, max_col = 0;
    triplet_reader_t::TripletVec Tvec;

    const std::size_t N = r_column_index.size();

    CHK_RETM_(_mmutil_read_rows_columns_triplets(mtx_file,
                                                 memory_location,
                                                 r_row_index,
                                                 r_column_index,
                                                 Tvec,
                                                 max_row,
                                                 max_col,
                                                 verbose),
              "Failed to read the triplets");

    if (N > max_col) {
        WLOG("Found " << (N - max_col) << " empty columns");
    }

    SpMat X(max_row, N);
    X.setZero();
    X.reserve(Tvec.size());
    X.setFromTriplets(Tvec.begin(), Tvec.end());

    SEXP xx = Rcpp::wrap(Mat(X));
    Rcpp::NumericMatrix ret(xx);
    return ret;
}

///////////////////////////
// actual implementation //
///////////////////////////
template <typename VEC1, typename VEC2>
int
_mmutil_read_columns_triplets(const std::string mtx_file,
                              const VEC1 &memory_location,
                              const VEC2 &r_column_index,
                              triplet_reader_t::TripletVec &Tvec,
                              Index &max_row,
                              Index &max_col,
                              const bool verbose)
{

    mm_info_reader_t info;
    CHK_RET_(peek_bgzf_header(mtx_file, info),
             "Failed to read the mtx file:" << mtx_file);

    ///////////////////////////////////////////////
    // We normally expect 1-based columns from R //
    ///////////////////////////////////////////////

    // convert 1-based to 0-based
    std::vector<Index> column_index;
    column_index.reserve(r_column_index.size());
    for (auto k : r_column_index) {
        if (k >= 1 && k <= info.max_col) {
            column_index.emplace_back(k - 1);
        }
    }

    max_col = 0;

    if (column_index.size() < 0)
        return EXIT_FAILURE;

    /////////////////////
    // check the files //
    /////////////////////

    max_row = info.max_row;
    CHK_RET(convert_bgzip(mtx_file));

    if (verbose)
        TLOG("info: " << info.max_row << " x " << info.max_col << " ("
                      << info.max_elem << " elements)");

    triplet_reader_t::index_map_t column_index_order; // we keep the same order
    for (auto k : column_index) {                     // of column_index
        const Index kk = static_cast<const Index>(k);
        if (column_index_order.count(kk) < 1)
            column_index_order[kk] = max_col++;
    }

    Tvec.clear();

    const auto blocks = find_consecutive_blocks(memory_location, column_index);

    for (auto block : blocks) {

        triplet_reader_t::index_map_t loc_map;
        for (Index j = block.lb; j < block.ub; ++j) {

            /////////////////////////////////////////////////
            // Sometimes...				   //
            // we may encounter discontinuous_index vector //
            // So, we need to check again                  //
            /////////////////////////////////////////////////

            if (column_index_order.count(j) > 0) {
                loc_map[j] = column_index_order[j];
            }
        }

        triplet_reader_t reader(Tvec, loc_map);

        CHK_RET_(visit_bgzf_block(mtx_file, block.lb_mem, block.ub_mem, reader),
                 "Failed on [" << block.lb << ", " << block.ub << ")");
    }

    if (verbose)
        TLOG("Successfully read " << blocks.size() << " block(s)");

    return EXIT_SUCCESS;
}

template <typename VEC1, typename VEC2, typename VEC3>
int
_mmutil_read_rows_columns_triplets(const std::string mtx_file,
                                   const VEC1 &memory_location,
                                   const VEC2 &r_row_index,
                                   const VEC3 &r_column_index,
                                   triplet_subrow_reader_t::TripletVec &Tvec,
                                   Index &max_row,
                                   Index &max_col,
                                   const bool verbose)
{

    mm_info_reader_t info;
    CHK_RET_(peek_bgzf_header(mtx_file, info),
             "Failed to read the mtx file:" << mtx_file);

    ///////////////////////////////////////////////
    // We normally expect 1-based columns from R //
    ///////////////////////////////////////////////

    // convert 1-based to 0-based
    std::vector<Index> column_index;
    column_index.reserve(r_column_index.size());
    for (auto k : r_column_index) {
        if (k >= 1 && k <= info.max_col) {
            column_index.emplace_back(k - 1);
        }
    }

    max_col = 0;

    if (column_index.size() < 0)
        return EXIT_FAILURE;

    /////////////////////
    // check the files //
    /////////////////////

    max_row = info.max_row;
    CHK_RET(convert_bgzip(mtx_file));

    if (verbose)
        TLOG("info: " << info.max_row << " x " << info.max_col << " ("
                      << info.max_elem << " elements)");

    triplet_subrow_reader_t::index_map_t column_index_order;
    for (auto k : column_index) {
        const Index kk = static_cast<const Index>(k);
        if (column_index_order.count(kk) < 1)
            column_index_order[kk] = max_col++;
    }

    triplet_subrow_reader_t::index_map_t row_loc_map;
    for (auto k : r_row_index) {
        if (k >= 1 && k <= info.max_row) {
            const Index r = k - 1; // one-based
            row_loc_map[r] = r;    // to zero-based
        }
    }

    Tvec.clear();

    const auto blocks = find_consecutive_blocks(memory_location, column_index);

    for (auto block : blocks) {

        triplet_subrow_reader_t::index_map_t col_loc_map;
        for (Index j = block.lb; j < block.ub; ++j) {

            /////////////////////////////////////////////////
            // Sometimes...				   //
            // we may encounter discontinuous_index vector //
            // So, we need to check again                  //
            /////////////////////////////////////////////////

            if (column_index_order.count(j) > 0) {
                col_loc_map[j] = column_index_order[j];
            }
        }

        triplet_subrow_reader_t reader(Tvec, row_loc_map, col_loc_map);

        CHK_RET_(visit_bgzf_block(mtx_file, block.lb_mem, block.ub_mem, reader),
                 "Failed on [" << block.lb << ", " << block.ub << ")");
    }

    return EXIT_SUCCESS;
}
