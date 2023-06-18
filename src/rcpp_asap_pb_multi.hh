#include "rcpp_asap.hh"
#include "rcpp_asap_stat.hh"

#include "RcppAnnoy.h"
#define ANNOYLIB_MULTITHREADED_BUILD 1

#ifndef RCPP_ASAP_PB_MULTI_HH_
#define RCPP_ASAP_PB_MULTI_HH_

struct mtx_data_t {
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

    struct IDX {
        IDX(const std::string v)
            : val(v)
        {
        }
        const std::string val;
    };

    using annoy_index_t = AnnoyIndex<Index,
                                     Scalar,
                                     Euclidean,
                                     Kiss64Random,
                                     RcppAnnoyIndexThreadPolicy>;

    explicit mtx_data_t(const MTX &mtx, const ROW &row, const IDX &idx)
        : mtx_file(mtx.val)
        , row_file(row.val)
        , idx_file(idx.val)
    {
        CHECK(peek_bgzf_header(mtx_file, info));
        CHECK(read_mmutil_index(idx_file, mtx_idx));
        sub_rows.reserve(info.max_row);
        CHECK(read_vector_file(row_file, sub_rows));
        has_index = false;

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
    SpMat read(std::vector<Index> &index);
    SpMat read_reloc(const Index lb, const Index ub);
    SpMat read_reloc(std::vector<Index> &index);

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

public:
    const std::string mtx_file;
    const std::string row_file;
    const std::string idx_file;
    std::vector<Index> mtx_idx;
    std::vector<std::string> sub_rows;

    mm_info_reader_t info;

private:
    SpMat A;

    std::shared_ptr<annoy_index_t> index_ptr;
    Mat Q_kn;
    bool has_index;
};

SpMat
mtx_data_t::read(const Index lb, const Index ub)
{
    const Index Nb = info.max_col;
    const Index lb_mem = lb < Nb ? mtx_idx[lb] : 0;
    const Index ub_mem = ub < Nb ? mtx_idx[ub] : 0;
    return mmutil::io::read_eigen_sparse_subset_col(mtx_file,
                                                    lb,
                                                    ub,
                                                    lb_mem,
                                                    ub_mem);
}

SpMat
mtx_data_t::read_reloc(const Index lb, const Index ub)
{
    return A * read(lb, ub);
}

void
mtx_data_t::relocate_rows(const std::unordered_map<std::string, Index> &row2pos)
{
    const Index D = row2pos.size();
    const Index Dloc = info.max_row;
    A.resize(D, Dloc);
    A.setZero();

    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(Dloc);

    for (Index ri = 0; ri < sub_rows.size(); ++ri) {
        const std::string &r = sub_rows.at(ri);
        if (row2pos.count(r) > 0) {
            const Index d = row2pos.at(r);
            triplets.emplace_back(Eigen::Triplet<Scalar>(ri, d, 1));
        }
    }
    A.reserve(triplets.size());
    A.setFromTriplets(triplets.begin(), triplets.end());
}

SpMat
mtx_data_t::read_matched_reloc(const std::vector<Scalar> &query,
                               const std::size_t knn,
                               std::vector<Index> &neigh_index,
                               std::vector<Scalar> &neigh_dist)
{
    return A * read_matched(query, knn, neigh_index, neigh_dist);
}

SpMat
mtx_data_t::read_matched(const std::vector<Scalar> &query,
                         const std::size_t knn,
                         std::vector<Index> &neigh_index,
                         std::vector<Scalar> &neigh_dist)
{
    knn_search(query, knn, neigh_index, neigh_dist);

    return mmutil::io::read_eigen_sparse_subset_col(mtx_file,
                                                    mtx_idx,
                                                    neigh_index);
}

SpMat
mtx_data_t::read_reloc(std::vector<Index> &index)
{
    return A * read(index);
}

SpMat
mtx_data_t::read(std::vector<Index> &index)
{
    return mmutil::io::read_eigen_sparse_subset_col(mtx_file, mtx_idx, index);
}

void
mtx_data_t::knn_search(const std::vector<Scalar> &query,
                       const std::size_t knn,
                       std::vector<Index> &neigh_index,
                       std::vector<Scalar> &neigh_dist)
{
    annoy_index_t &index = *index_ptr.get();
    const std::size_t nn = Q_kn.cols();
    const std::size_t nsearch = std::min(knn, nn);

    index.get_nns_by_vector(query.data(),
                            nsearch,
                            -1,
                            &neigh_index,
                            &neigh_dist);
}

//////////////////////
// helper functions //
//////////////////////

template <typename T>
void
convert_r_index(const std::vector<T> &cvec, std::vector<T> &rvec)
{
    rvec.resize(cvec.size());
    auto r_index = [](const T x) -> T { return x + 1; };
    std::transform(cvec.begin(), cvec.end(), rvec.begin(), r_index);
}

#endif
