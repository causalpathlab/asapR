#ifndef MMUTIL_MATCH_HH_
#define MMUTIL_MATCH_HH_

#include "math.hh"
#include "mmutil.hh"
#include "tuple_util.hh"

namespace mmutil { namespace match {

struct KNN {
    explicit KNN(const std::size_t _val)
        : val(_val)
    {
    }
    const std::size_t val;
};

/**
 * @param deg_i number of elements
 * @param dist deg_i-vector for distance
 * @param weights deg_i-vector for weights

 Since the inner-product distance is d(x,y) = (1 - x'y),
 d = 0.5 * (x - y)'(x - y) = 0.5 * (x'x + y'y) - x'y,
 we have Gaussian weight w(x,y) = exp(-lambda * d(x,y))

 */
void normalize_weights(const Index deg_i,
                       std::vector<Scalar> &dist,
                       std::vector<Scalar> &weights);

//////////////////////
// reciprocal match //
//////////////////////

std::tuple<Index, Index, Scalar>
parse_triplet(const std::tuple<Index, Index, Scalar> &tt);

std::tuple<Index, Index, Scalar>
parse_triplet(const Eigen::Triplet<Scalar> &tt);

template <typename T>
inline std::vector<T>
keep_reciprocal_knn(const std::vector<T> &knn_index, bool undirected = false)
{
    // Make sure that we could only consider reciprocal kNN pairs
    std::unordered_map<std::tuple<Index, Index>,
                       short,
                       hash_tuple::hash<std::tuple<Index, Index>>>
        edge_count;

    auto _count = [&edge_count](const auto &tt) {
        Index i, j, temp;
        std::tie(i, j, std::ignore) = parse_triplet(tt);
        if (i == j)
            return;

        if (i > j) {
            temp = i;
            i = j;
            j = temp;
        }

        if (edge_count.count({ i, j }) < 1) {
            edge_count[{ i, j }] = 1;
        } else {
            edge_count[{ i, j }] += 1;
        }
    };

    std::for_each(knn_index.begin(), knn_index.end(), _count);

    auto is_mutual = [&edge_count, &undirected](const auto &tt) {
        Index i, j, temp;
        std::tie(i, j, std::ignore) = parse_triplet(tt);
        if (i == j)
            return false;
        if (i > j) {
            temp = i;
            i = j;
            j = temp;
        }
        if (undirected)
            return (edge_count[{ i, j }] > 1) && (i <= j);
        return (edge_count[{ i, j }] > 1);
    };

    std::vector<T> reciprocal_knn_index;
    reciprocal_knn_index.reserve(knn_index.size());
    std::copy_if(knn_index.begin(),
                 knn_index.end(),
                 std::back_inserter(reciprocal_knn_index),
                 is_mutual);

    return reciprocal_knn_index;
}

}} // namespace mmutil::match


#endif
