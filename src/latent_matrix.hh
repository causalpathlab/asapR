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

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;

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

    // Gibbs sampling combining the log-probabilities of the rows and
    // columns
    template <typename Derived>
    void gibbs_sample_row_col(const Eigen::MatrixBase<Derived> &rowLogit,
                              const Eigen::MatrixBase<Derived> &colLogit,
                              const std::size_t NUM_THREADS = 1)
    {

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
        {
            RNG lrng(rng);
            lrng.long_jump(omp_get_thread_num() + 1);
#pragma omp for
#else
        RNG &lrng = rng;
#endif
            for (Index rr = 0; rr < Z.rows(); ++rr) {
                rowvec_sampler_t<Mat, RNG> sampler(lrng, K);
                for (Index cc = 0; cc < Z.cols(); ++cc) {
                    Z(rr, cc) =
                        sampler(softmax(rowLogit.row(rr) + colLogit.row(cc)));
                }
            }
#if defined(_OPENMP)
        }
#endif
    }

    // MH sampling proposal by rows, then test by columns
    template <typename Derived>
    void mh_sample_row_col(const std::vector<Index> &rowwise_proposal,
                           const Eigen::MatrixBase<Derived> &colwise_logit,
                           const std::size_t NUM_THREADS = 1)
    {
        constexpr Scalar zero = 0;
        boost::random::uniform_01<Scalar> runif;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
        {
            RNG lrng(rng);
            lrng.long_jump(omp_get_thread_num() + 1);
#pragma omp for
#else
        RNG &lrng = rng;
#endif
            for (Index cc = 0; cc < Z.cols(); ++cc) {
                for (Index rr = 0; rr < Z.rows(); ++rr) {
                    const Index k_old = Z(rr, cc);
                    const Index k_new = rowwise_proposal.at(rr);
                    if (k_old != k_new) {
                        const Scalar l_new = colwise_logit(cc, k_new);
                        const Scalar l_old = colwise_logit(cc, k_old);
                        const Scalar log_mh_ratio =
                            std::min(zero, l_new - l_old);
                        const Scalar u = runif(lrng);
                        if (u <= 0 || fasterlog(u) < log_mh_ratio) {
                            Z(rr, cc) = k_new;
                        }
                    }
                }
            }
#if defined(_OPENMP)
        }
#endif
    }

    // MH sampling proposal by columns, then test by rows
    template <typename Derived>
    void mh_sample_col_row(const std::vector<Index> &colwise_proposal,
                           const Eigen::MatrixBase<Derived> &rowwise_logit,
                           const std::size_t NUM_THREADS = 1)
    {
        constexpr Scalar zero = 0;
        boost::random::uniform_01<Scalar> runif;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
        {
            RNG lrng(rng);
            lrng.long_jump(omp_get_thread_num() + 1);
#pragma omp for
#else
        RNG &lrng = rng;
#endif
            for (Index rr = 0; rr < Z.rows(); ++rr) {
                for (Index cc = 0; cc < Z.cols(); ++cc) {
                    const Index k_old = Z(rr, cc);
                    const Index k_new = colwise_proposal.at(cc);
                    if (k_old != k_new) {
                        const Scalar l_new = rowwise_logit(rr, k_new);
                        const Scalar l_old = rowwise_logit(rr, k_old);
                        const Scalar log_mh_ratio =
                            std::min(zero, l_new - l_old);
                        const Scalar u = runif(lrng);
                        if (u <= 0 || fasterlog(u) < log_mh_ratio) {
                            Z(rr, cc) = k_new;
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
    softmax_op_t<Mat> softmax;
};

#endif
