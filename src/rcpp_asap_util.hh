#include "rcpp_asap.hh"

#ifndef RCPP_ASAP_UTIL_HH_
#define RCPP_ASAP_UTIL_HH_

namespace asap { namespace util {

template <typename T, typename Ret>
void
colsum_safe(const Eigen::MatrixBase<T> &X, Eigen::MatrixBase<Ret> &ret)
{
    inf_zero_op<Mat> op;
    ret.derived() = X.unaryExpr(op).colwise().sum();
}

template <typename T, typename Ret>
void
rowsum_safe(const Eigen::MatrixBase<T> &X, Eigen::MatrixBase<Ret> &ret)
{
    inf_zero_op<Mat> op;
    ret.derived() = X.unaryExpr(op).rowwise().sum();
}

template <typename T>
Scalar
sum_safe(const Eigen::MatrixBase<T> &X)
{
    inf_zero_op<Mat> op;
    return X.unaryExpr(op).sum();
}

template <typename T1, typename T2, typename Ret>
void
XY_safe(const Eigen::MatrixBase<T1> &X,
        const Eigen::MatrixBase<T2> &Y,
        Eigen::MatrixBase<Ret> &ret)
{
    inf_zero_op<Mat> op;
    ret.derived() = X.unaryExpr(op) * Y.unaryExpr(op);
}

template <typename T1, typename T2, typename Ret>
void
XtY_safe(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         Eigen::MatrixBase<Ret> &ret)
{
    XY_safe(X.transpose(), Y, ret);
}

template <typename T>
void
softmax_safe(const Eigen::MatrixBase<T> &logits, Eigen::MatrixBase<T> &prob)
{
    const Scalar maxlogits = logits.maxCoeff();
    prob.derived() = (logits.array() - maxlogits).exp().matrix();
    prob.derived() /= prob.sum();
}

template <typename Derived>
void
stretch_matrix_columns_inplace(Eigen::MatrixBase<Derived> &_y,
                               const typename Derived::Scalar qq_min = 0.01,
                               const typename Derived::Scalar qq_max = 0.99,
                               const bool verbose = false)
{

    Derived &Ystd = _y.derived();
    std::vector<typename Derived::Scalar> quantile(Ystd.size());
    Eigen::Map<Derived>(quantile.data(), Ystd.rows(), Ystd.cols()) = Ystd;

    //////////////////////////////
    // step 1. quntile clipping //
    //////////////////////////////

    TLOG_(verbose, "n = " << quantile.size() << " elements");

    const typename Derived::Scalar nn = quantile.size();
    const std::size_t q1 = std::ceil(qq_min * nn);
    const std::size_t q2 = std::floor(qq_max * nn);

    Scalar lb, ub;
    if (q1 < q2 && q2 < quantile.size()) {

        std::nth_element(quantile.begin(),
                         quantile.begin() + q1,
                         quantile.end());

        lb = quantile[q1];

        std::nth_element(quantile.begin() + q1 + 1,
                         quantile.begin() + q2,
                         quantile.end());

        ub = quantile[q2];
    } else {
        lb = *std::min_element(quantile.begin(), quantile.end());
        ub = *std::max_element(quantile.begin(), quantile.end());
    }

    TLOG_(verbose,
          "Shrink the raw values "
              << "between " << lb << " and " << ub << ".");

    if (lb < ub) {
        clamp_op<Derived> clamp_y(lb, ub);
        Ystd = Ystd.unaryExpr(clamp_y).eval();
    }

    /////////////////////////////
    // step 2. standardization //
    /////////////////////////////

    stdizer_t<Derived> std(Ystd);
    std.colwise();

    TLOG_(verbose,
          "After standardization we have: "
              << "[" << Ystd.minCoeff() << ", " << Ystd.maxCoeff() << "]");
}

}} // asap::util
#endif
