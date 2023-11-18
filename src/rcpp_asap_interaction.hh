#include "rcpp_asap.hh"

#ifndef RCPP_ASAP_INTERACTION_HH_
#define RCPP_ASAP_INTERACTION_HH_

template <typename I, typename I2, typename S, typename Derived>
int
build_knn_graph(const std::vector<I> &knn_src,
                const std::vector<I> &knn_tgt,
                const std::vector<S> &knn_weight,
                const I2 Ncell,
                Eigen::SparseMatrixBase<Derived> &_w)
{
    Derived W = _w.derived();
    using Scalar = typename Derived::Scalar;

    const Index Nedge = knn_src.size(); // number of pairs

    ASSERT_RET(
        Nedge == knn_tgt.size(),
        "source and target vectors should have the same number of elements");

    ASSERT_RET(
        Nedge == knn_weight.size(),
        "source and weight vectors should have the same number of elements");

    std::vector<Eigen::Triplet<Scalar>> knn_index;
    knn_index.reserve(Nedge);
    for (Index j = 0; j < Nedge; ++j) {
        // convert 1-based to 0-based
        const Index s = knn_src[j] - 1, t = knn_tgt[j] - 1;
        if (s < Ncell && t < Ncell && s >= 0 && t >= 0 && s != t) {
            knn_index.emplace_back(Eigen::Triplet<Scalar>(s, t, knn_weight[j]));
        }
    }

    W.resize(Ncell, Ncell);
    W.reserve(knn_index.size());
    W.setFromTriplets(knn_index.begin(), knn_index.end());

    return EXIT_SUCCESS;
}

#endif
