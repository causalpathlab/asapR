#include "rcpp_util_struct.hh"
#include "rcpp_btree.hh"

// [[Rcpp::export]]
Eigen::MatrixXf
pbt_dependency_matrix(const std::size_t depth)
{

    if (depth < 2) {
        return Eigen::MatrixXf::Ones(1, 1);
    }

    return rcpp::util::pbt_dep_adj(depth);
}

namespace rcpp { namespace util {

SpMat
pbt_dep_adj(const std::size_t depth)
{
    const std::size_t K = pbt_num_depth_to_leaves(depth);
    const std::size_t N = pbt_num_depth_to_nodes(depth);

    // TLOG("K = " << K << ", N = " << N);

    // 1-based row index
    std::vector<std::size_t> rows(K);
    std::iota(rows.begin(), rows.end(), K);

    using ET = Eigen::Triplet<Scalar>;

    std::vector<ET> triplets;

    for (std::size_t k = 0; k < rows.size(); ++k) {
        triplets.emplace_back(ET(rows.at(k) - 1, k, 1));
    }

    for (std::size_t d = 1; d <= depth; ++d) {
        for (std::size_t k = 0; k < rows.size(); ++k) {
            rows[k] = std::floor(rows[k] / 2);
        }

        for (std::size_t k = 0; k < rows.size(); ++k) {
            triplets.emplace_back(ET(rows.at(k) - 1, k, 1));
        }
    }

    return build_eigen_sparse(triplets, N, K);
}

std::size_t
pbt_num_depth_to_leaves(const std::size_t depth)
{
    return 1 << depth;
}

std::size_t
pbt_num_depth_to_nodes(const std::size_t depth)
{
    return (1 << (depth + 1)) - 1;
}

std::size_t
pbt_num_leaves_to_nodes(const std::size_t num_leaves)
{
    const std::size_t depth = std::ceil(std::log2(num_leaves));
    ASSERT(depth < 1, "too shallow PBT");
    return pbt_num_depth_to_nodes(depth);
}

}} // end of namespace
