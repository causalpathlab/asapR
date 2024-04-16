#include "rcpp_asap.hh"
#include "rcpp_asap_pb.hh"
#include "rcpp_asap_regression.hh"
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

#define PRODUCT_EDGE(A_dd, Y1, Y2) \
    (Y1.cwiseProduct(A_dd * Y2) + Y2.cwiseProduct(A_dd * Y1)) * 0.5

#define SUM_EDGE(A_dd, Y1, Y2) (Y1 + A_dd * Y2 + Y2 + A_dd * Y1) * 0.5

#endif
