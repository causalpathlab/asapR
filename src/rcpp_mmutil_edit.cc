#include "mmutil.hh"
#include "mmutil_select.hh"
#include "mmutil_filter.hh"
#include "rcpp_util.hh"

#include <unordered_set>

//' Take a subset of rows and create a new MTX file-set
//'
//' @description For the new mtx file, empty columns with only zero
//'   elements will be removed.
//'
//' @param mtx_file data file
//' @param row_file row file
//' @param col_file column file
//' @param selected selected row names
//' @param output output header
//' @param MAX_ROW_WORD maximum words per line in `row_file`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @return a list of file names: {output}.{mtx,rows,cols}.gz
//'
//' @examples
//'
//' options(stringsAsFactors=FALSE)
//' rr <- rgamma(20, 1, 1)
//' mm <- matrix(rgamma(10 * 2, 1, 1), 10, 2)
//' src.hdr <- "test_org"
//' src.files <- mmutil_simulate_poisson(mm, rr, src.hdr)
//' Y <- Matrix::readMM(src.files$mtx)
//' rownames(Y) <- read.table(src.files$row)$V1
//' print(Y)
//' sub.rows <- sort(read.table(src.files$row)$V1[sample(10,3)])
//' print(sub.rows)
//' tgt.hdr <- "test_sub"
//' tgt.files <- mmutil_copy_selected_rows(
//'                src.files$mtx,
//'                src.files$row,
//'                src.files$col,
//'                sub.rows,
//'                tgt.hdr)
//' Y <- Matrix::readMM(tgt.files$mtx)
//' colnames(Y) <- read.table(tgt.files$col)$V1
//' rownames(Y) <- read.table(tgt.files$row)$V1
//' print(Y)
//' unlink(list.files(pattern = src.hdr))
//' unlink(list.files(pattern = tgt.hdr))
//'
// [[Rcpp::export]]
Rcpp::List
mmutil_copy_selected_rows(const std::string mtx_file,
                          const std::string row_file,
                          const std::string col_file,
                          const Rcpp::StringVector r_selected,
                          const std::string output,
                          const std::size_t MAX_ROW_WORD = 2,
                          const char ROW_WORD_SEP = '_',
                          const std::size_t MAX_COL_WORD = 100,
                          const char COL_WORD_SEP = '@')
{

    ASSERT_RETL(file_exists(mtx_file), "missing the MTX file");
    ASSERT_RETL(file_exists(row_file), "missing the ROW file");
    ASSERT_RETL(file_exists(col_file), "missing the COL file");

    std::vector<std::string> selected = rcpp::util::copy(r_selected);

    // First pass: select rows and create temporary MTX
    copy_selected_rows(mtx_file,
                       row_file,
                       selected,
                       output + "-temp",
                       MAX_ROW_WORD,
                       ROW_WORD_SEP);

    std::string temp_mtx_file = output + "-temp.mtx.gz";

    // Second pass: squeeze out empty columns
    CHK_RETL(filter_col_by_nnz(1,
                               temp_mtx_file,
                               col_file,
                               output,
                               MAX_COL_WORD,
                               COL_WORD_SEP));

    if (file_exists(temp_mtx_file)) {
        std::remove(temp_mtx_file.c_str());
    }

    std::string out_row_file = output + ".rows.gz";
    std::string out_col_file = output + ".cols.gz";
    std::string out_mtx_file = output + ".mtx.gz";

    if (file_exists(out_row_file)) {
        std::string temp_row_file = output + ".rows.gz-backup";
        WLOG("Remove existing output row file: " << out_row_file);
        copy_file(out_row_file, temp_row_file);
        remove_file(out_row_file);
    }

    {
        std::string temp_row_file = output + "-temp.rows.gz";
        rename_file(temp_row_file, out_row_file);
    }

    ASSERT_RETL(file_exists(out_row_file),
                "unable to create the row file: " << out_row_file)

    ASSERT_RETL(file_exists(out_col_file),
                "unable to create the column file: " << out_col_file)

    ASSERT_RETL(file_exists(out_mtx_file),
                "unable to create the mtx file: " << out_mtx_file)

    ////////////////////////
    // index new mtx file //
    ////////////////////////

    const std::string out_idx_file = out_mtx_file + ".index";

    if (file_exists(out_idx_file))
        remove_file(out_idx_file);

    CHECK(mmutil::index::build_mmutil_index(out_mtx_file, out_idx_file));

    return Rcpp::List::create(Rcpp::_["mtx"] = out_mtx_file,
                              Rcpp::_["row"] = out_row_file,
                              Rcpp::_["col"] = out_col_file,
                              Rcpp::_["idx"] = out_idx_file);
}

//' Take a subset of columns and create a new MTX file-set
//'
//' @param mtx_file data file
//' @param row_file row file
//' @param col_file column file
//' @param selected selected column names
//' @param output output header
//' @param MAX_COL_WORD maximum words per line in `col_file`
//' @param COL_WORD_SEP word separation character to replace white space
//'
//' @examples
//'
//' options(stringsAsFactors=FALSE)
//' rr <- rgamma(20, 1, 1)
//' mm <- matrix(rgamma(10 * 2, 1, 1), 10, 2)
//' src.hdr <- "test_org"
//' src.files <- mmutil_simulate_poisson(mm, rr, src.hdr)
//' Y <- Matrix::readMM(src.files$mtx)
//' colnames(Y) <- read.table(src.files$col)$V1
//' print(Y)
//' sub.cols <- sort(read.table(src.files$col)$V1[sample(20,3)])
//' print(sub.cols)
//' tgt.hdr <- "test_sub"
//' tgt.files <- mmutil_copy_selected_columns(
//'                          src.files$mtx,
//'                          src.files$row,
//'                          src.files$col,
//'                          sub.cols, tgt.hdr)
//' Y <- Matrix::readMM(tgt.files$mtx)
//' colnames(Y) <- read.table(tgt.files$col)$V1
//' print(Y)
//' unlink(list.files(pattern = src.hdr))
//' unlink(list.files(pattern = tgt.hdr))
//'
// [[Rcpp::export]]
Rcpp::List
mmutil_copy_selected_columns(const std::string mtx_file,
                             const std::string row_file,
                             const std::string col_file,
                             const Rcpp::StringVector &r_selected,
                             const std::string output,
                             const std::size_t MAX_COL_WORD = 100,
                             const char COL_WORD_SEP = '@')
{

    ASSERT_RETL(file_exists(mtx_file), "missing the MTX file");
    ASSERT_RETL(file_exists(row_file), "missing the ROW file");
    ASSERT_RETL(file_exists(col_file), "missing the COL file");

    std::vector<std::string> selected = rcpp::util::copy(r_selected);

    const std::string out_row_file = output + ".rows.gz";
    copy_file(row_file, out_row_file);
    copy_selected_columns(mtx_file,
                          col_file,
                          selected,
                          output,
                          MAX_COL_WORD,
                          COL_WORD_SEP);

    ASSERT_RETL(file_exists(out_row_file),
                "unable to create the row file: " << out_row_file)

    std::string out_col_file = output + ".cols.gz";
    ASSERT_RETL(file_exists(out_col_file),
                "unable to create the column file: " << out_col_file)

    std::string out_mtx_file = output + ".mtx.gz";
    ASSERT_RETL(file_exists(out_mtx_file),
                "unable to create the mtx file: " << out_mtx_file)

    ////////////////////////
    // index new mtx file //
    ////////////////////////

    const std::string out_idx_file = out_mtx_file + ".index";

    if (file_exists(out_idx_file))
        remove_file(out_idx_file);

    CHECK(mmutil::index::build_mmutil_index(out_mtx_file, out_idx_file));

    return Rcpp::List::create(Rcpp::_["mtx"] = out_mtx_file,
                              Rcpp::_["row"] = out_row_file,
                              Rcpp::_["col"] = out_col_file,
                              Rcpp::_["idx"] = out_idx_file);
}
