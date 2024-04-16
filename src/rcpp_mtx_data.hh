#include "mmutil.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_io.hh"

#include "RcppAnnoy.h"
#define ANNOYLIB_MULTITHREADED_BUILD 1

#ifndef RCPP_MTX_DATA_HH_
#define RCPP_MTX_DATA_HH_

using namespace mmutil::io;
using namespace mmutil::bgzf;

struct mtx_tuple_t {

    struct MTX {
        MTX(const std::string v)
            : val(v)
        {
        }
        const std::string val;
    };

    struct ROW {
        ROW(const std::string v)
            : val(v)
        {
        }
        const std::string val;
    };

    struct COL {
        COL(const std::string v)
            : val(v)
        {
        }
        const std::string val;
    };

    struct IDX {
        IDX(const std::string v)
            : val(v)
        {
        }
        const std::string val;
    };

    explicit mtx_tuple_t(const MTX &_mtx,
                         const ROW &_row,
                         const COL &_col,
                         const IDX &_idx)

        : mtx(_mtx.val)
        , row(_row.val)
        , col(_col.val)
        , idx(_idx.val)
    {
    }

    const MTX mtx;
    const ROW row;
    const COL col;
    const IDX idx;
};

struct mtx_data_t {

    using annoy_index_t = Annoy::AnnoyIndex<Index,
                                            Scalar,
                                            Annoy::Euclidean,
                                            Kiss64Random,
                                            RcppAnnoyIndexThreadPolicy>;

    explicit mtx_data_t(const mtx_tuple_t &mtx_tup,
                        const std::size_t MAX_ROW_WORD = 2,
                        const char ROW_WORD_SEP = '_',
                        const std::size_t MAX_COL_WORD = 100,
                        const char COL_WORD_SEP = '@')
        : mtx_file(mtx_tup.mtx.val)
        , row_file(mtx_tup.row.val)
        , col_file(mtx_tup.col.val)
        , idx_file(mtx_tup.idx.val)
    {
        CHECK(peek_bgzf_header(mtx_file, info));
        CHECK(read_mmutil_index(idx_file, mtx_idx));
        sub_rows.reserve(info.max_row);
        CHECK(read_line_file(row_file, sub_rows, MAX_ROW_WORD, ROW_WORD_SEP));
        CHECK(read_line_file(col_file, sub_cols, MAX_COL_WORD, COL_WORD_SEP));

        has_index = false;
        has_reloc = false;

        A.resize(info.max_row, info.max_row);
        A.setZero();
    }

    template <typename Derived>
    int build_index(const Eigen::MatrixBase<Derived> &Q_kn_,
                    const bool verbose = true)
    {

        ASSERT_RET(info.max_col == Q_kn_.cols(),
                   "Incompatible Q with the underlying mtx data");

        Q_kn = Q_kn_;
        normalize_columns_inplace(Q_kn);
        const std::size_t rank = Q_kn.rows();
        index_ptr = std::make_shared<annoy_index_t>(rank);
        annoy_index_t &index = *index_ptr.get();
        std::vector<Scalar> vec(rank);
        for (Index j = 0; j < Q_kn.cols(); ++j) {
            Eigen::Map<Mat>(vec.data(), rank, 1) = Q_kn.col(j);
            index.add_item(j, vec.data());
        }
        index.build(50);
        TLOG_(verbose, "Populated " << Q_kn.cols() << " items");
        has_index = true;

        return EXIT_SUCCESS;
    }

    SpMat read(const Index lb, const Index ub);
    SpMat read(const std::vector<Index> &index);
    SpMat read_reloc(const Index lb, const Index ub);
    SpMat read_reloc(const std::vector<Index> &index);

    SpMat read_matched(const std::vector<Scalar> &query,
                       const std::size_t knn,
                       std::vector<Index> &neigh_index,
                       std::vector<Scalar> &neigh_dist);

    SpMat read_matched_reloc(const std::vector<Scalar> &query,
                             const std::size_t knn,
                             std::vector<Index> &neigh_index,
                             std::vector<Scalar> &neigh_dist);

    void relocate_rows(const std::unordered_map<std::string, Index> &row2pos);

    void knn_search(const std::vector<Scalar> &query,
                    const std::size_t knn,
                    std::vector<Index> &neigh_index,
                    std::vector<Scalar> &neigh_dist);

    Index max_row() const;
    Index max_col() const;
    Index max_elem() const;

    const std::vector<std::string> &row_names() const;
    const std::vector<std::string> &col_names() const;

public:
    const std::string mtx_file;
    const std::string row_file;
    const std::string col_file;
    const std::string idx_file;

private:
    mm_info_reader_t info;
    std::vector<Index> mtx_idx;
    std::vector<std::string> sub_rows;
    std::vector<std::string> sub_cols;

private:
    SpMat A;

    std::shared_ptr<annoy_index_t> index_ptr;
    Mat Q_kn;
    bool has_index;
    bool has_reloc;
};

#endif
