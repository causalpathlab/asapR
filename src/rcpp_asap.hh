#ifndef _RCPP_ASAP_HH
#define _RCPP_ASAP_HH

#include "mmutil.hh"
#include "mmutil_stat.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "mmutil_io.hh"
#include "tuple_util.hh"

#include "svd.hh"

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>

// [[Rcpp::depends(dqrng, sitmo, BH)]]
#include <dqrng.h>
#include <dqrng_distribution.h>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <xoshiro.h>

using namespace mmutil::io;
using namespace mmutil::bgzf;

template <typename RNG>
struct discrete_sampler_t {

    using disc_distrib = boost::random::discrete_distribution<>;

    explicit discrete_sampler_t(RNG &_rng, const Index k)
        : rng(_rng)
        , K(k)
        , _weights(k)
        , _rdisc(_weights)
    {
    }

    template <typename Derived>
    const std::vector<Index> &operator()(const Eigen::MatrixBase<Derived> &W)
    {
        using RowVec = typename Eigen::internal::plain_row_type<Derived>::type;

        using param = disc_distrib::param_type;

        if (W.rows() != _sampled.size())
            _sampled.resize(W.rows());

        for (Index g = 0; g < W.rows(); ++g) {
            Eigen::Map<RowVec>(&_weights[0], 1, K) = W.row(g);
            _sampled[g] = _rdisc(rng, param(_weights));
        }
        return _sampled;
    }

    const std::vector<Index> &sampled() const { return _sampled; }

    RNG &rng;
    const Index K;
    std::vector<Scalar> _weights;
    disc_distrib _rdisc;
    std::vector<Index> _sampled;
};

template <typename RNG>
struct latent_matrix_t {
    explicit latent_matrix_t(const Index r,
                             const Index c,
                             RNG &_rng,
                             const Index k)
        : nrows(r)
        , ncols(c)
        , ruint_op(_rng, k)
        , Z(r, c)
    {
        randomize();
    }

    const Index rows() const { return nrows; }
    const Index cols() const { return ncols; }
    const Index nrows, ncols;

    template <typename Derived>
    Derived slice_k(const Eigen::MatrixBase<Derived> &Y, const Index k) const
    {
        return Y.cwiseProduct(Z.unaryExpr(is_k_t(k)));
    }

    inline void randomize() { Z = IntMat::NullaryExpr(nrows, ncols, ruint_op); }

    inline const Index coeff(const Index i, const Index j) const
    {
        return Z.coeff(i, j);
    }

    inline void set(const Index i, const Index j, const Index v)
    {
        Z(i, j) = v;
    }

    struct ruint_op_t {

        explicit ruint_op_t(RNG &_rng, const Index k)
            : rng(_rng)
            , K(k)
            , _rK(0, k - 1)
        {
        }

        using uint_distrib = boost::random::uniform_int_distribution<Index>;

        const Index operator()() const { return _rK(rng); }

        RNG &rng;
        const Index K;
        uint_distrib _rK;
    };

    ruint_op_t ruint_op;

    IntMat Z;

    struct is_k_t {

        explicit is_k_t(const Index _k)
            : k(_k)
        {
        }

        const Scalar operator()(const Index &z) const
        {
            return z == k ? 1. : 0.;
        }

        const Index k;
    };
};

template <typename RNG>
struct gamma_param_t {
    explicit gamma_param_t(const Index r,
                           const Index c,
                           const Scalar _a0,
                           const Scalar _b0,
                           RNG &_rng)
        : nrows(r)
        , ncols(c)
        , a0(_a0)
        , b0(_b0)
        , a_stat(r, c)
        , b_stat(r, c)
        , rgamma_op(_rng)
    {
        a_stat.setConstant(a0);
        b_stat.setConstant(b0);
        calibrate();
    }

    void calibrate()
    {
        estimate_mean = a_stat.cwiseQuotient(b_stat);
        estimate_log = a_stat.binaryExpr(b_stat, estimate_log_op);
    }

    const Mat &mean() const { return estimate_mean; }
    const Mat &log_mean() const { return estimate_log; }

    template <typename Derived1, typename Derived2>
    void update(const Eigen::MatrixBase<Derived1> &update_a,
                const Eigen::MatrixBase<Derived2> &update_b)
    {
        a_stat.setConstant(a0);
        b_stat.setConstant(b0);
        a_stat += update_a;
        b_stat += update_b;
    }

    template <typename Derived1, typename Derived2>
    void update_col(const Eigen::MatrixBase<Derived1> &update_a,
                    const Eigen::MatrixBase<Derived2> &update_b,
                    const Index k)
    {
        a_stat.col(k).setConstant(a0);
        b_stat.col(k).setConstant(b0);
        a_stat.col(k) += update_a;
        b_stat.col(k) += update_b;
    }

    Mat sample() { return a_stat.binaryExpr(b_stat, rgamma_op); }

    const Index rows() const { return nrows; }
    const Index cols() const { return ncols; }

    struct rgamma_op_t {

        explicit rgamma_op_t(RNG &_rng)
            : rng(_rng)
        {
        }

        using gamma_distrib = boost::random::gamma_distribution<Scalar>;

        const Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            return _rgamma(rng, gamma_distrib::param_type(a, b));
        }

        RNG &rng;
        gamma_distrib _rgamma;
    };

    struct estimate_log_op_t {
        const Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            return fasterdigamma(a) - fasterlog(b);
        }
    };

    const Index nrows, ncols;
    const Scalar a0, b0;

    Mat a_stat;
    Mat b_stat;
    Mat estimate_mean; // E[lambda]
    Mat estimate_log;  // E[log lambda]

    rgamma_op_t rgamma_op;
    estimate_log_op_t estimate_log_op;
};

#endif
