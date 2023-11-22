#include "rcpp_asap.hh"
#include "rcpp_util.hh"

#ifndef RCPP_ASAP_INTERACTION_HH_
#define RCPP_ASAP_INTERACTION_HH_

template <typename Derived>
Scalar
product_similarity(const Eigen::SparseMatrixBase<Derived> &y_d1,
                   const Eigen::SparseMatrixBase<Derived> &y_d2)
{
    const Scalar n1 = y_d1.sum();
    const Scalar n2 = y_d2.sum();
    const Scalar n12 = y_d1.cwiseProduct(y_d2).sum();
    const Scalar denom = std::sqrt(n1) * std::sqrt(n2);
    return (n1 < 1. || n2 < 1.) ? 0. : (n12 / denom);
}

template <typename I, typename I2, typename S, typename Derived>
int
build_knn_graph(const std::vector<I> &knn_src,
                const std::vector<I> &knn_tgt,
                const std::vector<S> &knn_weight,
                const I2 Ncell,
                Eigen::SparseMatrixBase<Derived> &_w)
{
    Derived &W = _w.derived();
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

    // TLOG("knn_index: " << knn_index.size());

    W.resize(Ncell, Ncell);
    W.reserve(knn_index.size());
    W.setFromTriplets(knn_index.begin(), knn_index.end());

    // TLOG("W: " << W.nonZeros());

    return EXIT_SUCCESS;
}

#endif
