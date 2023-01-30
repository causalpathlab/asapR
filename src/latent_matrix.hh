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

#ifndef LATENT_MATRIX_HH_
#define LATENT_MATRIX_HH_

template <typename RNG>
struct latent_matrix_t {

    using IntegerMatrix = typename Eigen::
        Matrix<std::ptrdiff_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    explicit latent_matrix_t(const Index r,
                             const Index c,
                             const Index k,
                             RNG &_rng)
        : nrows(r)
        , ncols(c)
        , K(k)
        , ruint_op(_rng, k)
        , Z(r, c)
        , rng(_rng)
    {
        Z.setZero();
        randomize();
    }

    const Index rows() const { return nrows; }
    const Index cols() const { return ncols; }

    // Y .* (Z == k)
    template <typename Derived>
    Derived slice_k(const Eigen::MatrixBase<Derived> &Y, const Index k) const
    {
        return Y.cwiseProduct(Z.unaryExpr(is_k_t(k)));
    }

    template <typename Derived>
    void sample_mh(const std::vector<Index> &rowwise_proposal,
                   const Eigen::MatrixBase<Derived> &colwise_logit,
                   const std::size_t NUM_THREADS = 1)
    {
        constexpr Scalar zero = 0;

        boost::random::uniform_01<Scalar> runif;

        // const Index D = rowwise_logit.rows();
        // const Index N = colwise_logit.rows();

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
        {
            RNG lrng(rng);
            lrng.long_jump(omp_get_thread_num() + 1);
#pragma omp for
#else
        RNG &lrng = rng;
#endif
            for (Index jj = 0; jj < Z.cols(); ++jj) {
                for (Index ii = 0; ii < Z.rows(); ++ii) {
                    const Index k_old = Z(ii, jj);
                    const Index k_new = rowwise_proposal.at(ii);
                    if (k_old != k_new) {
                        const Scalar l_new = colwise_logit(jj, k_new);
                        const Scalar l_old = colwise_logit(jj, k_old);
                        const Scalar log_mh_ratio =
                            std::min(zero, l_new - l_old);
                        const Scalar u = runif(lrng);
                        if (u <= 0 || fasterlog(u) < log_mh_ratio) {
                            Z(ii, jj) = k_new;
                        }
                    }
                }
            }
#if defined(_OPENMP)
        }
#endif
    }

    inline void randomize()
    {
        Z = IntegerMatrix::NullaryExpr(nrows, ncols, ruint_op);
    }

    const Index nrows, ncols, K;

private:
    struct ruint_op_t {

        explicit ruint_op_t(RNG &_rng, const Index k)
            : rng(_rng)
            , K(k)
            , _rK { 0, K - 1 }
        {
        }

        using distrib = boost::random::uniform_int_distribution<Index>;

        const Index operator()() const { return _rK(rng); }

        RNG &rng;
        const Index K;
        distrib _rK;
    };

    struct is_k_t {

        explicit is_k_t(const Index _k)
            : k_target(_k)
        {
        }

        const Scalar operator()(const Index &z) const
        {
            return z == k_target ? 1. : 0.;
        }

        const Index k_target;
    };

    ruint_op_t ruint_op;
    IntegerMatrix Z;
    RNG &rng;

    // inline const Index coeff(const Index i, const Index j) const
    // {
    //     return Z.coeff(i, j);
    // }

    // inline void set(const Index i, const Index j, const Index v)
    // {
    //     Z(i, j) = v;
    // }
};

#endif
