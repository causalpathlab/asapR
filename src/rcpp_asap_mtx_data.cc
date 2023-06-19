#include "rcpp_asap_mtx_data.hh"

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
