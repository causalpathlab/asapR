#include "rcpp_asap_eigenSparse_data.hh"

SpMat
eigenSparse_data_t::read(const Index lb, const Index ub)
{
    return data.middleCols(lb, ub - lb);
}

SpMat
eigenSparse_data_t::read_reloc(const Index lb, const Index ub)
{
    return A * read(lb, ub);
}

SpMat
eigenSparse_data_t::read(const std::vector<Index> &index)
{
    return col_sub(data, index);
}

SpMat
eigenSparse_data_t::read_reloc(const std::vector<Index> &index)
{
    return A * read(index);
}

void
eigenSparse_data_t::relocate_rows(
    const std::unordered_map<std::string, Index> &row2pos)
{
    const Index Dtot = row2pos.size();
    const Index Dloc = info.max_row;
    A.resize(Dtot, Dloc);
    A.setZero();

    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(Dloc);
    for (Index ri = 0; ri < sub_rows.size(); ++ri) {
        const std::string &r = sub_rows.at(ri);
        if (row2pos.count(r) > 0) {
            const Index d = row2pos.at(r);
            triplets.emplace_back(Eigen::Triplet<Scalar>(d, ri, 1));
        }
    }
    A.reserve(triplets.size());
    A.setFromTriplets(triplets.begin(), triplets.end());
    has_reloc = true;
}

SpMat
eigenSparse_data_t::read_matched(const std::vector<Scalar> &query,
                                 const std::size_t knn,
                                 std::vector<Index> &neigh_index,
                                 std::vector<Scalar> &neigh_dist)
{
    knn_search(query, knn, neigh_index, neigh_dist);

    return read(neigh_index);
}

SpMat
eigenSparse_data_t::read_matched_reloc(const std::vector<Scalar> &query,
                                       const std::size_t knn,
                                       std::vector<Index> &neigh_index,
                                       std::vector<Scalar> &neigh_dist)
{
    ASSERT(has_reloc, "should have assigned the relocation vector");

    return A * read_matched(query, knn, neigh_index, neigh_dist);
}

void
eigenSparse_data_t::knn_search(const std::vector<Scalar> &query,
                               const std::size_t knn,
                               std::vector<Index> &neigh_index,
                               std::vector<Scalar> &neigh_dist)
{
    ASSERT(has_index, "should have built the ANNOY index");

    annoy_index_t &index = *index_ptr.get();
    const std::size_t nn = Q_kn.cols();
    const std::size_t nsearch = std::min(knn, nn);

    index.get_nns_by_vector(query.data(),
                            nsearch,
                            -1,
                            &neigh_index,
                            &neigh_dist);
}
