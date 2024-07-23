// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef EIGEN_UTIL_HH_
#define EIGEN_UTIL_HH_

#include <algorithm>
#include <functional>
#include <vector>
#include <limits>

#include "math.hh"
#include "util.hh"
#include "svd.hh"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_01.hpp>

template <typename T>
struct softmax_op_t {
    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    using RowVec = typename Eigen::internal::plain_row_type<T>::type;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;

    inline RowVec apply_row(Eigen::Ref<const RowVec> logits)
    {
        return log_row(logits).unaryExpr(exp_op);
    }

    inline ColVec apply_col(Eigen::Ref<const ColVec> logits)
    {
        return log_col(logits).unaryExpr(exp_op);
    }

    template <typename V>
    inline V _log_V(Eigen::Ref<const V> logits)
    {
        Index K = logits.size();
        Scalar log_denom = logits.coeff(0);
        for (Index k = 1; k < K; ++k) {
            log_denom = log_sum_exp(log_denom, logits.coeff(k));
        }
        return (logits.array() - log_denom).eval();
    }

    template <typename V>
    inline V _log_V_weighted(Eigen::Ref<const V> logits,
                             Eigen::Ref<const V> weights)
    {
        const Scalar max_logits = logits.maxCoeff();
        const Scalar wse = (logits.array() - max_logits)
                               .matrix()
                               .unaryExpr(exp_op)
                               .cwiseProduct(weights)
                               .sum();

        const Scalar log_denom = max_logits + fasterlog(wse);

        return (logits.array() - log_denom).eval();
    }

    inline RowVec log_row(Eigen::Ref<const RowVec> logits)
    {
        return _log_V<RowVec>(logits);
    }

    inline RowVec log_row_weighted(Eigen::Ref<const RowVec> logits,
                                   Eigen::Ref<const RowVec> weights)
    {
        return _log_V_weighted<RowVec>(logits, weights);
    }

    inline ColVec log_col(Eigen::Ref<const ColVec> logits)
    {
        return _log_V<ColVec>(logits);
    }

    inline ColVec log_col_weighted(Eigen::Ref<const ColVec> logits,
                                   Eigen::Ref<const ColVec> weights)
    {
        return _log_V_weighted<ColVec>(logits, weights);
    }

    struct log_sum_exp_t {
        Scalar operator()(const Scalar log_a, const Scalar log_b)
        {
            Scalar v;
            if (log_a < log_b) {
                v = log_b + fastlog(1. + fastexp(log_a - log_b));
            } else {
                v = log_a + fastlog(1. + fastexp(log_b - log_a));
            }
            return v;
        }
    } log_sum_exp;

    struct exp_op_t {
        const Scalar operator()(const Scalar x) const { return fasterexp(x); }
    } exp_op;

    struct log_op_t {
        const Scalar operator()(const Scalar x) const { return fasterlog(x); }
    } log_op;
};

template <typename T, typename RNG>
struct rowvec_sampler_t {
    using Scalar = typename T::Scalar;
    using Index = typename T::Index;

    using disc_distrib = boost::random::discrete_distribution<>;
    using disc_param = disc_distrib::param_type;
    using RowVec = typename Eigen::internal::plain_row_type<T>::type;

    explicit rowvec_sampler_t(RNG &_rng, const Index k)
        : rng(_rng)
        , K(k)
        , _prob(k)
    {
    }

    inline Index operator()(const RowVec &prob)
    {
        Eigen::Map<RowVec>(&_prob[0], 1, K) = prob;
        return _rdisc(rng, disc_param(_prob));
    }

    RNG &rng;
    const Index K;
    std::vector<Scalar> _prob;
    disc_distrib _rdisc;
};

template <typename T, typename RNG>
struct matrix_sampler_t {

    using disc_distrib = boost::random::discrete_distribution<>;
    using disc_param = disc_distrib::param_type;

    using Scalar = typename T::Scalar;
    using Index = typename T::Index;

    using IndexVec = std::vector<Index>;

    explicit matrix_sampler_t(RNG &_rng, const Index k)
        : rng(_rng)
        , K(k)
        , _weights(k)
        , _rdisc(_weights)
    {
    }

    template <typename Derived>
    const IndexVec &sample(const Eigen::MatrixBase<Derived> &W)
    {
        using ROW = typename Eigen::internal::plain_row_type<Derived>::type;
        check_size(W);

        for (Index g = 0; g < W.rows(); ++g) {
            Eigen::Map<ROW>(&_weights[0], 1, K) = W.row(g);
            _sampled[g] = _rdisc(rng, disc_param(_weights));
        }
        return _sampled;
    }

    template <typename Derived>
    const IndexVec &sample_logit(const Eigen::MatrixBase<Derived> &W)
    {
        using ROW = typename Eigen::internal::plain_row_type<Derived>::type;
        check_size(W);

        for (Index g = 0; g < W.rows(); ++g) {
            Eigen::Map<ROW>(&_weights[0], 1, K) = softmax.apply_row(W.row(g));
            _sampled[g] = _rdisc(rng, disc_param(_weights));
        }
        return _sampled;
    }

    template <typename Derived>
    const IndexVec &operator()(const Eigen::MatrixBase<Derived> &W)
    {
        return sample(W);
    }

    template <typename Derived>
    void check_size(const Eigen::MatrixBase<Derived> &W)
    {
        if (W.rows() != _sampled.size())
            _sampled.resize(W.rows());

        if (W.cols() != _weights.size())
            _weights.resize(W.cols());
    }

    const IndexVec &sampled() const { return _sampled; }

    void copy_to(IndexVec &dst) const
    {
        if (dst.size() != _sampled.size())
            dst.resize(_sampled.size());

        std::copy(std::begin(_sampled), std::end(_sampled), std::begin(dst));
    }

    RNG &rng;
    const Index K;
    std::vector<Scalar> _weights;
    disc_distrib _rdisc;
    IndexVec _sampled;

    softmax_op_t<T> softmax;
};

template <typename EigenVec>
inline auto
std_vector(const EigenVec eigen_vec)
{
    std::vector<typename EigenVec::Scalar> ret(eigen_vec.size());
    for (typename EigenVec::Index j = 0; j < eigen_vec.size(); ++j) {
        ret[j] = eigen_vec(j);
    }
    return ret;
}

template <typename T>
inline auto
eigen_vector(const std::vector<T> &std_vec)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> ret(std_vec.size());

    for (std::size_t j = 0; j < std_vec.size(); ++j) {
        ret(j) = std_vec.at(j);
    }

    return ret;
}

template <typename T, typename T2>
inline auto
eigen_vector_(const std::vector<T> &std_vec)
{
    Eigen::Matrix<T2, Eigen::Dynamic, 1> ret(std_vec.size());

    for (std::size_t j = 0; j < std_vec.size(); ++j) {
        ret(j) = std_vec.at(j);
    }

    return ret;
}

template <typename EigenVec, typename StdVec>
inline void
std_vector(const EigenVec eigen_vec, StdVec &ret)
{
    ret.resize(eigen_vec.size());
    using T = typename StdVec::value_type;
    for (typename EigenVec::Index j = 0; j < eigen_vec.size(); ++j) {
        ret[j] = static_cast<T>(eigen_vec(j));
    }
}

template <typename T>
inline std::vector<Eigen::Triplet<float>>
eigen_triplets(const std::vector<T> &Tvec, bool weighted = true)
{
    using Scalar = float;
    using _Triplet = Eigen::Triplet<Scalar>;
    using _TripletVec = std::vector<_Triplet>;

    _TripletVec ret;
    ret.reserve(Tvec.size());

    if (weighted) {
        for (auto tt : Tvec) {
            ret.emplace_back(
                _Triplet(std::get<0>(tt), std::get<1>(tt), std::get<2>(tt)));
        }
    } else {
        for (auto tt : Tvec) {
            ret.emplace_back(_Triplet(std::get<0>(tt), std::get<1>(tt), 1.0));
        }
    }
    return ret;
}

template <typename Scalar>
inline auto
eigen_triplets(const std::vector<Eigen::Triplet<Scalar>> &Tvec)
{
    return Tvec;
}

template <typename TVEC, typename INDEX>
inline Eigen::SparseMatrix<float, Eigen::ColMajor, std::ptrdiff_t> //
build_eigen_sparse(const TVEC &Tvec, const INDEX max_row, const INDEX max_col)
{
    auto _tvec = eigen_triplets(Tvec);
    using Scalar = float;
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, std::ptrdiff_t>;

    SpMat ret(max_row, max_col);
    ret.reserve(_tvec.size());
    ret.setFromTriplets(_tvec.begin(), _tvec.end());
    return ret;
}

////////////////////////////////////////////////////////////////

template <typename Derived>
void
build_diagonal(const typename Derived::Index d,
               const typename Derived::Scalar val,
               Eigen::SparseMatrixBase<Derived> &_x)
{
    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;
    Derived &x = _x.derived();
    x.setZero();
    x.resize(d, d);
    x.reserve(d);

    using ET = Eigen::Triplet<Scalar>;
    std::vector<ET> triples;
    triples.reserve(d);
    for (Index j = 0; j < d; ++j) {
        triples.emplace_back(ET(j, j, val));
    }

    x.setFromTriplets(triples.begin(), triples.end());
}

////////////////////////////////////////////////////////////////

template <typename Vec>
inline std::vector<typename Vec::Index>
eigen_argsort_descending(const Vec &data)
{
    using Index = typename Vec::Index;
    std::vector<Index> index(data.size());
    std::iota(std::begin(index), std::end(index), 0);
    std::sort(std::begin(index), std::end(index), [&](Index lhs, Index rhs) {
        return data(lhs) > data(rhs);
    });
    return index;
}

////////////////////////////////////////////////////////////////

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar,
                     Eigen::Dynamic,
                     Eigen::Dynamic,
                     Eigen::ColMajor>
row_score_degree(const Eigen::SparseMatrixBase<Derived> &_xx)
{
    const Derived &xx = _xx.derived();
    using Scalar = typename Derived::Scalar;
    using Mat =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    return xx.unaryExpr([](const Scalar x) { return std::abs(x); }) *
        Mat::Ones(xx.cols(), 1);
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar,
                     Eigen::Dynamic,
                     Eigen::Dynamic,
                     Eigen::ColMajor>
row_score_sd(const Eigen::SparseMatrixBase<Derived> &_xx)
{
    const Derived &xx = _xx.derived();
    using Scalar = typename Derived::Scalar;
    using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Mat =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    Vec s1 = xx * Mat::Ones(xx.cols(), 1);
    Vec s2 = xx.cwiseProduct(xx) * Mat::Ones(xx.cols(), 1);
    const Scalar n = xx.cols();
    Vec ret = s2 - s1.cwiseProduct(s1 / n);
    ret = ret / std::max(n - 1.0, 1.0);
    ret = ret.cwiseSqrt();

    return ret;
}

template <typename Derived, typename COLS>
inline Eigen::SparseMatrix<typename Derived::Scalar, //
                           Eigen::ColMajor>
col_sub(const Eigen::SparseMatrixBase<Derived> &_mat, const COLS &sub_cols)
{
    using SpMat =
        typename Eigen::SparseMatrix<typename Derived::Scalar, Eigen::ColMajor>;

    using Index = typename SpMat::Index;
    using Scalar = typename SpMat::Scalar;
    const SpMat &mat = _mat.derived();

    // ASSERT(!mat.IsRowMajor, "column major");

    SpMat ret(mat.rows(), sub_cols.size());

    using ET = Eigen::Triplet<Scalar>;

    std::vector<ET> triples;

    Index cc = 0;

    for (Index c : sub_cols) {
        if (c < 0 || c >= mat.cols())
            continue;

        for (typename SpMat::InnerIterator it(mat, c); it; ++it) {
            triples.push_back(ET(it.row(), cc, it.value()));
        }

        ++cc;
    }

    ret.reserve(triples.size());
    ret.setFromTriplets(triples.begin(), triples.end());

    return ret;
}

template <typename Derived, typename ROWS>
inline Eigen::Matrix<typename Derived::Scalar, //
                     Eigen::Dynamic,           //
                     Eigen::Dynamic,           //
                     Eigen::ColMajor>
row_sub(const Eigen::MatrixBase<Derived> &_mat, const ROWS &sub_rows)
{
    using Mat = typename Eigen::Matrix<typename Derived::Scalar, //
                                       Eigen::Dynamic,           //
                                       Eigen::Dynamic,           //
                                       Eigen::ColMajor>;

    using Index = typename Mat::Index;
    using Scalar = typename Mat::Scalar;
    const Mat &mat = _mat.derived();

    Mat ret(sub_rows.size(), mat.cols());
    ret.setZero();

    Index rr = 0;

    for (Index r : sub_rows) {
        if (r < 0 || r >= mat.rows())
            continue;

        ret.row(rr) += mat.row(r);

        ++rr;
    }

    return ret;
}

template <typename FUN, typename DATA>
inline void
visit_sparse_matrix(const DATA &data, FUN &fun)
{
    using Scalar = typename DATA::Scalar;
    using Index = typename DATA::Index;

    fun.eval_after_header(data.rows(), data.cols(), data.nonZeros());

    for (Index o = 0; o < data.outerSize(); ++o) {
        for (typename DATA::InnerIterator it(data, o); it; ++it) {
            fun.eval(it.row(), it.col(), it.value());
        }
    }

    fun.eval_end_of_data();
}

template <typename Derived>
inline Eigen::
    SparseMatrix<typename Derived::Scalar, Eigen::ColMajor, std::ptrdiff_t>
    vcat(const Eigen::SparseMatrixBase<Derived> &_upper,
         const Eigen::SparseMatrixBase<Derived> &_lower)
{
    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;
    const Derived &upper = _upper.derived();
    const Derived &lower = _lower.derived();

    ASSERT(upper.cols() == lower.cols(), "mismatching columns in vcat");

    using _Triplet = Eigen::Triplet<Scalar>;

    std::vector<_Triplet> triplets;
    triplets.reserve(upper.nonZeros() + lower.nonZeros());

    using SpMat = typename Eigen::
        SparseMatrix<typename Derived::Scalar, Eigen::ColMajor, std::ptrdiff_t>;

    for (Index k = 0; k < upper.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(upper, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }

    for (Index k = 0; k < lower.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(lower, k); it; ++it) {
            triplets.emplace_back(upper.rows() + it.row(),
                                  it.col(),
                                  it.value());
        }
    }

    SpMat result(lower.rows() + upper.rows(), upper.cols());
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

template <typename Derived>
inline Eigen::
    SparseMatrix<typename Derived::Scalar, Eigen::ColMajor, std::ptrdiff_t>
    hcat(const Eigen::SparseMatrixBase<Derived> &_left,
         const Eigen::SparseMatrixBase<Derived> &_right)
{
    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;
    const Derived &left = _left.derived();
    const Derived &right = _right.derived();

    ASSERT(left.rows() == right.rows(), "mismatching rows in hcat");

    using _Triplet = Eigen::Triplet<Scalar>;

    std::vector<_Triplet> triplets;
    triplets.reserve(left.nonZeros() + right.nonZeros());

    using SpMat = typename Eigen::
        SparseMatrix<typename Derived::Scalar, Eigen::ColMajor, std::ptrdiff_t>;

    for (Index k = 0; k < left.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(left, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }

    for (Index k = 0; k < right.outerSize(); ++k) {
        for (typename SpMat::InnerIterator it(right, k); it; ++it) {
            triplets.emplace_back(it.row(), left.cols() + it.col(), it.value());
        }
    }

    SpMat result(left.rows(), left.cols() + right.cols());
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

template <typename Derived, typename Derived2>
Eigen::Matrix<typename Derived::Scalar,
              Eigen::Dynamic,
              Eigen::Dynamic,
              Eigen::ColMajor>
hcat(const Eigen::MatrixBase<Derived> &_left,
     const Eigen::MatrixBase<Derived2> &_right)
{

    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;

    using Mat =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    const Derived &L = _left.derived();
    const Derived2 &R = _right.derived();

    ASSERT(L.rows() == R.rows(), "Must have the same number of rows");

    Mat ret(L.rows(), L.cols() + R.cols());

    for (Index j = 0; j < L.cols(); ++j) {
        ret.col(j) = L.col(j);
    }

    for (Index j = 0; j < R.cols(); ++j) {
        ret.col(j + L.cols()) = R.col(j);
    }

    return ret;
}

template <typename T>
struct stdizer_t {
    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    using RowVec = typename Eigen::internal::plain_row_type<T>::type;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;

    explicit stdizer_t(T &data_, const Scalar r_m = 1., const Scalar r_v = 1.)
        : X(data_)
        , D1(X.rows())
        , D2(X.cols())
        , rate_m(r_m)
        , rate_v(r_v)
        , rowMean(D2)
        , rowSqMean(D2)
        , rowMu(D2)
        , rowSd(D2)
        , rowNobs(D2)
        , temp(D2)
        , obs_val()
        , is_obs_val()
    {
        rowMean.setZero();
        rowMu.setZero();
        rowSqMean.setZero();
        rowSd.setOnes();
    }

private:
    T &X;

public:
    const Index D1, D2;
    const Scalar rate_m, rate_v;

    void colwise(const Scalar eps = 1e-8)
    {
        rowNobs = X.unaryExpr(is_obs_val).colwise().sum();
        rowNobs.array() += eps;

        rowMean = X.unaryExpr(obs_val).colwise().sum().cwiseQuotient(rowNobs);

        rowSqMean = (X.cwiseProduct(X))
                        .unaryExpr(obs_val)
                        .colwise()
                        .sum()
                        .cwiseQuotient(rowNobs);

        if (rate_m > 0) {
            rowMu *= (1. - rate_m);
            rowMu += rate_m * rowMean;
        }

        if (rate_v > 0) {
            temp = rowSqMean - rowMean.cwiseProduct(rowMean);
            temp.array() += eps;
            if (temp.minCoeff() < 0.) {
                temp.array() -= temp.minCoeff();
            }
            rowSd *= (1. - rate_v);
            rowSd += rate_v * temp.cwiseSqrt();
        }

        X.array().rowwise() -= rowMu.array();
        X.array().rowwise() /= (rowSd.array() + eps);
    }

    void colwise_scale(const Scalar eps = 1e-8)
    {
        rowNobs = X.unaryExpr(is_obs_val).colwise().sum();
        rowNobs.array() += eps;

        rowMean = X.unaryExpr(obs_val).colwise().sum().cwiseQuotient(rowNobs);

        rowSqMean = (X.cwiseProduct(X))
                        .unaryExpr(obs_val)
                        .colwise()
                        .sum()
                        .cwiseQuotient(rowNobs);

        if (rate_m == 1) {
            rowMu = rowMean;
        } else if (rate_m > 0) {
            rowMu *= (1. - rate_m);
            rowMu += rate_m * rowMean;
        }

        temp = rowSqMean - rowMean.cwiseProduct(rowMean);
        temp.array() += eps;
        if (temp.minCoeff() < 0.) {
            temp.array() -= temp.minCoeff();
        }

        if (rate_v > 0) {
            rowSd += rate_v * temp.cwiseSqrt();
        } else if (rate_v > 0) {
            rowSd *= (1. - rate_v);
            rowSd += rate_v * temp.cwiseSqrt();
        }

        X.array().rowwise() /= (rowSd.array() + eps);
    }

private:
    RowVec rowMean, rowSqMean, rowMu, rowSd, rowNobs, temp;

    struct obs_val_t {
        const Scalar operator()(const Scalar &x) const
        {
            return std::isfinite(x) ? x : 0.;
        }
    } obs_val;

    struct is_obs_val_t {
        const Scalar operator()(const Scalar &x) const
        {
            return std::isfinite(x) ? 1. : 0.;
        }
    } is_obs_val;
};

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar,
              Eigen::Dynamic,
              Eigen::Dynamic,
              Eigen::ColMajor>
standardize_columns(const Eigen::MatrixBase<Derived> &Xraw,
                    const typename Derived::Scalar EPS = 1e-8)
{
    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;
    using mat_t =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    mat_t X(Xraw.rows(), Xraw.cols());
    X = Xraw;
    stdizer_t<mat_t> std_op(X);
    std_op.colwise(EPS);
    return X;
}

template <typename Derived>
void
standardize_columns_inplace(Eigen::MatrixBase<Derived> &X_,
                            const typename Derived::Scalar EPS = 1e-8)
{
    using Scalar = typename Derived::Scalar;

    Derived &X = X_.derived();
    stdizer_t<Derived> std_op(X);
    std_op.colwise(EPS);
}

template <typename Derived>
void
scale_columns_inplace(Eigen::MatrixBase<Derived> &X_,
                      const typename Derived::Scalar EPS = 1e-8)
{
    using Scalar = typename Derived::Scalar;

    Derived &X = X_.derived();
    stdizer_t<Derived> std_op(X);
    std_op.colwise_scale(EPS);
}

template <typename Derived, typename Derived2>
void
residual_columns_inplace(Eigen::MatrixBase<Derived> &_yy,
                         const Eigen::MatrixBase<Derived2> &_xx,
                         const typename Derived::Scalar eps = 1e-8)
{
    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;

    using Mat =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    const Derived Yraw = _yy.derived(); // copy to avoid aliasing issues
    Derived &Yout = _yy.derived();
    const Derived2 &X = _xx.derived();
    using ColVec = typename Eigen::internal::plain_col_type<Derived>::type;

    ASSERT(X.rows() == Yraw.rows(), "incompatible Y and X");

    if (X.cols() < 1) {
        // nothing to do
    } else if (X.cols() == 1) {

        const Scalar denom = X.cwiseProduct(X).sum();
        for (Index k = 0; k < Yraw.cols(); ++k) {
            const Scalar b =
                (X.transpose() * Yraw.col(k)).sum() / (denom + eps);
            Yout.col(k) = Yraw.col(k) - b * X.col(0);
        }

    } else {
        const std::size_t r =
            std::min(X.rows(), std::min(X.cols(), Yraw.cols()));

        // if (X.rows() < 1000) {
        //     Eigen::BDCSVD<Derived> svd_x;
        //     svd_x.compute(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
        //     d = svd_x.singularValues();
        //     u = svd_x.matrixU();
        // } else {
        // }
        const std::size_t lu_iter = 5;
        RandomizedSVD<Derived> svd_x(r, lu_iter);
        svd_x.compute(X);
        ColVec d = svd_x.singularValues();
        Mat u = svd_x.matrixU();

        for (Index k = 0; k < r; ++k) {
            if (d(k) < eps)
                u.col(k).setZero();
        }

        // X theta = X inv(X'X) X' Y
        //         = U D V' V inv(D^2) V' (U D V')' Y
        //         = U inv(D) V' V D U' Y
        //         = U U' Y

        Yout = Yraw - u * u.transpose() * Yraw;
    }
}

template <typename Derived>
void
normalize_columns_inplace(Eigen::MatrixBase<Derived> &_mat)
{
    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;

    Derived &mat = _mat.derived();
    const Scalar eps = 1e-8;

    for (Index c = 0; c < mat.cols(); ++c) {
        const Scalar denom = std::max(mat.col(c).norm(), eps);
        mat.col(c) /= denom;
    }
}

template <typename Derived>
void
normalize_columns_inplace(Eigen::SparseMatrixBase<Derived> &_mat)
{
    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;

    Derived &mat = _mat.derived();
    const Scalar eps = 1e-8;

    std::vector<Scalar> col_norm(mat.cols());
    std::fill(col_norm.begin(), col_norm.end(), 0.0);

    for (Index k = 0; k < mat.outerSize(); ++k) {
        for (typename Derived::InnerIterator it(mat, k); it; ++it) {
            const Scalar x = it.value();
            col_norm[it.col()] += x * x;
        }
    }

    for (Index k = 0; k < mat.outerSize(); ++k) {
        for (typename Derived::InnerIterator it(mat, k); it; ++it) {
            const Scalar x = it.value();
            const Scalar denom = std::sqrt(col_norm[it.col()]);
            it.valueRef() = x / std::max(denom, eps);
        }
    }
}

////////////////////////////////////////////////////////////////
template <typename Derived>
void
setConstant(Eigen::SparseMatrixBase<Derived> &mat,
            const typename Derived::Scalar val)
{
    using Scalar = typename Derived::Scalar;
    auto fill_const = [val](const Scalar &x) { return val; };
    Derived &Mat = mat.derived();
    Mat = Mat.unaryExpr(fill_const);
}

template <typename Derived>
void
setConstant(Eigen::MatrixBase<Derived> &mat, const typename Derived::Scalar val)
{
    Derived &Mat = mat.derived();
    Mat.setConstant(val);
}

template <typename T>
struct running_stat_t {
    using Scalar = typename T::Scalar;
    using Index = typename T::Index;

    explicit running_stat_t(const Index _d1, const Index _d2)
        : d1 { _d1 }
        , d2 { _d2 }
    {
        Cum.resize(d1, d2);
        SqCum.resize(d1, d2);
        Mean.resize(d1, d2);
        Var.resize(d1, d2);
        reset();
    }

    void reset()
    {
        setConstant(SqCum, 0.0);
        setConstant(Cum, 0.0);
        setConstant(Mean, 0.0);
        setConstant(Var, 0.0);
        n = 0.0;
    }

    template <typename Derived>
    void operator()(const Eigen::MatrixBase<Derived> &X)
    {
        Cum += X;
        SqCum += X.cwiseProduct(X);
        n += 1.0;
    }

    template <typename Derived>
    void operator()(const Eigen::SparseMatrixBase<Derived> &X)
    {
        Cum += X;
        SqCum += X.cwiseProduct(X);
        n += 1.0;
    }

    const T mean()
    {
        if (n > 0) {
            Mean = Cum / n;
        }
        return Mean;
    }

    const T var()
    {
        if (n > 1.) {
            Mean = Cum / n;
            Var = SqCum / (n - 1.) - Mean.cwiseProduct(Mean) * n / (n - 1.);
            Var = Var.unaryExpr(clamp_zero_op);
        } else {
            Var.setZero();
        }
        return Var;
    }

    const T sd() { return var().cwiseSqrt(); }

    const Index d1;
    const Index d2;

    struct clamp_zero_op_t {
        const Scalar operator()(const Scalar &x) const
        {
            return x < .0 ? 0. : x;
        }
    } clamp_zero_op;

    T Cum;
    T SqCum;
    T Mean;
    T Var;
    Scalar n;
};

template <typename T>
struct inf_zero_op {
    using Scalar = typename T::Scalar;
    const Scalar operator()(const Scalar &x) const
    {
        return std::isfinite(x) ? x : zero_val;
    }
    static constexpr Scalar zero_val = 0.0;
};

template <typename T>
struct is_obs_op {
    using Scalar = typename T::Scalar;
    const Scalar operator()(const Scalar &x) const
    {
        return std::isfinite(x) ? one_val : zero_val;
    }
    static constexpr Scalar one_val = 1.0;
    static constexpr Scalar zero_val = 0.0;
};

template <typename T>
struct is_positive_op {
    using Scalar = typename T::Scalar;
    const Scalar operator()(const Scalar &x) const
    {
        return (std::isfinite(x) && (x > zero_val)) ? one_val : zero_val;
    }
    static constexpr Scalar one_val = 1.0;
    static constexpr Scalar zero_val = 0.0;
};

template <typename T>
struct exp_op {
    using Scalar = typename T::Scalar;
    const Scalar operator()(const Scalar &x) const { return fasterexp(x); }
};

template <typename T>
struct log1p_op {
    using Scalar = typename T::Scalar;
    const Scalar operator()(const Scalar &x) const { return fasterlog(1. + x); }
};

template <typename T>
struct safe_log_op {
    using Scalar = typename T::Scalar;

    explicit safe_log_op(const Scalar _min_val)
        : min_val(_min_val)
        , min_log_val(fasterlog(min_val))
    {
    }

    const Scalar operator()(const Scalar &x) const
    {
        return x <= min_val ? min_log_val : fasterlog(x);
    }

    const Scalar min_val;
    const Scalar min_log_val;
};

template <typename T>
struct safe_sqrt_op {
    using Scalar = typename T::Scalar;
    const Scalar operator()(const Scalar &x) const
    {
        return x <= 0. ? 0. : std::sqrt(x);
    }
};

template <typename T>
struct safe_div_op {
    using Scalar = typename T::Scalar;
    explicit safe_div_op(const Scalar _a0, const Scalar _b0)
        : a0(_a0)
        , b0(_b0)
    {
    }
    const Scalar operator()(const Scalar &a, const Scalar &b) const
    {
        return (a + a0) / (b + b0);
    }
    const Scalar a0;
    const Scalar b0;
};

template <typename T>
struct at_least_one_op {
    using Scalar = typename T::Scalar;
    const Scalar operator()(const Scalar &x) const { return (x < 1.) ? 1. : x; }
};

template <typename T>
struct at_least_zero_op {
    using Scalar = typename T::Scalar;
    const Scalar operator()(const Scalar &x) const { return (x < 0.) ? 0. : x; }
};

template <typename T>
struct at_least_val_op {
    using Scalar = typename T::Scalar;
    explicit at_least_val_op(const Scalar _lb)
        : lb(_lb)
    {
    }
    const Scalar operator()(const Scalar &x) const { return (x < lb) ? lb : x; }
    const Scalar lb;
};

template <typename T>
struct clamp_op {
    using Scalar = typename T::Scalar;
    explicit clamp_op(const Scalar _lb, const Scalar _ub)
        : lb(_lb)
        , ub(_ub)
    {
        ASSERT(lb < ub, "LB < UB");
    }
    const Scalar operator()(const Scalar &x) const
    {
        if (x > ub)
            return ub;
        if (x < lb)
            return lb;
        return x;
    }
    const Scalar lb;
    const Scalar ub;
};

template <typename T>
struct add_pseudo_op {
    using Scalar = typename T::Scalar;
    explicit add_pseudo_op(const Scalar pseudo_val)
        : val(pseudo_val)
    {
    }
    const Scalar operator()(const Scalar &x) const { return x + val; }
    const Scalar val;
};

template <typename T1, typename T2, typename Ret>
void
XY_nobs(const Eigen::MatrixBase<T1> &X,
        const Eigen::MatrixBase<T2> &Y,
        Eigen::MatrixBase<Ret> &ret,
        const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    ret.derived() = (X.unaryExpr(op1) * Y.unaryExpr(op2)).array() + pseudo;
}

template <typename T1, typename T2, typename Ret>
void
XY_nobs(const Eigen::MatrixBase<T1> &X,
        const Eigen::MatrixBase<T2> &Y,
        Eigen::SparseMatrixBase<Ret> &ret,
        const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    add_pseudo_op<Ret> op_add(pseudo);

    times_set(X.unaryExpr(op1), Y.unaryExpr(op2), ret);
    ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         const Eigen::MatrixBase<T3> &Z,
         Eigen::MatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;

    ret.derived() =
        (X.unaryExpr(op1) * Y.unaryExpr(op2) * Z.unaryExpr(op3)).array() +
        pseudo;
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         const Eigen::MatrixBase<T3> &Z,
         Eigen::SparseMatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;
    add_pseudo_op<Ret> op_add(pseudo);

    auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
    times_set(X.unaryExpr(op1), YZ, ret);
    ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         const Eigen::SparseMatrixBase<T3> &Z,
         Eigen::MatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;
    add_pseudo_op<Ret> op_add(pseudo);

    auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
    times_set(X.unaryExpr(op1), YZ, ret);
    ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         const Eigen::SparseMatrixBase<T3> &Z,
         Eigen::SparseMatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;
    add_pseudo_op<Ret> op_add(pseudo);

    auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
    times_set(X.unaryExpr(op1), YZ, ret);
    ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::SparseMatrixBase<T1> &X,
         const Eigen::SparseMatrixBase<T2> &Y,
         const Eigen::MatrixBase<T3> &Z,
         Eigen::SparseMatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;
    add_pseudo_op<Ret> op_add(pseudo);

    auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
    times_set(X.unaryExpr(op1), YZ, ret);
    ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void
XYZ_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::SparseMatrixBase<T2> &Y,
         const Eigen::MatrixBase<T3> &Z,
         Eigen::SparseMatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    is_obs_op<T1> op1;
    is_obs_op<T2> op2;
    is_obs_op<T3> op3;
    add_pseudo_op<Ret> op_add(pseudo);

    auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
    times_set(X.unaryExpr(op1), YZ, ret);
    ret.derived() = ret.unaryExpr(op_add);
}

////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename Ret>
void
XtY_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         Eigen::MatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    XY_nobs(X.transpose(), Y, ret, pseudo);
}

template <typename T1, typename T2, typename Ret>
void
XtY_nobs(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         Eigen::SparseMatrixBase<Ret> &ret,
         const typename Ret::Scalar pseudo = 1.0)
{
    XY_nobs(X.transpose(), Y, ret, pseudo);
}

#endif
