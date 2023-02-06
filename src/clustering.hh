#include "mmutil.hh"
#include "eigen_util.hh"

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>

// [[Rcpp::depends(dqrng, sitmo, BH)]]
#include <dqrng.h>
#include <dqrng_distribution.h>
#include <xoshiro.h>
#include "clustering_util.hh"

#ifndef CLUSTERING_HH_
#define CLUSTERING_HH_

template <typename T>
struct softmax_op_t {
    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    using RowVec = typename Eigen::internal::plain_row_type<T>::type;

    RowVec operator()(const RowVec &logits)
    {
        const Scalar log_denom = std::accumulate(logits.data(),
                                                 logits.data() + logits.size(),
                                                 log_eps,
                                                 log_sum_exp);

        return (logits.array() - log_denom).matrix().unaryExpr(exp_op);
    }

    struct log_sum_exp_t {
        Scalar operator()(const Scalar log_a, const Scalar log_b)
        {
            return _log_sum_exp(log_a, log_b);
        }
    } log_sum_exp;

    struct exp_op_t {
        const Scalar operator()(const Scalar &x) const { return fasterexp(x); }
    } exp_op;

    static constexpr Scalar log_eps = -200; // log(eps) = -200
};

template <typename T, typename RNG>
struct row_sampler_t {

    using disc_distrib = boost::random::discrete_distribution<>;
    using disc_param = disc_distrib::param_type;
    using RowVec = typename Eigen::internal::plain_row_type<T>::type;

    explicit row_sampler_t(RNG &_rng, const Index k)
        : rng(_rng)
        , K(k)
        , _prob(k)
    {
    }

    Index operator()(const RowVec &prob)
    {
        Eigen::Map<RowVec>(&_prob[0], 1, K) = prob;
        return _rdisc(rng, disc_param(_prob));
    }

    RNG &rng;
    const Index K;
    std::vector<Scalar> _prob;
    disc_distrib _rdisc;
};

template <typename T>
struct clustering_status_t {
    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;
    using RowVec = typename Eigen::internal::plain_row_type<T>::type;

    explicit clustering_status_t(const Index n, const Index d, const Index k)
        : N(n)
        , D(d)
        , K(k)
        , latent(N, K)
        , parameter(D, K)
        , log_parameter(D, K)
    {
    }

    template <typename Derived>
    void record_latent(const Eigen::MatrixBase<Derived> &prob)
    {
        latent(prob);
    }

    template <typename Derived>
    void record_parameter(const Eigen::MatrixBase<Derived> &param)
    {
        parameter(param);
    }

    template <typename Derived>
    void record_log_parameter(const Eigen::MatrixBase<Derived> &log_param)
    {
        log_parameter(log_param);
    }

    const Index N, D, K;
    running_stat_t<T> latent;
    running_stat_t<T> parameter;
    running_stat_t<T> log_parameter;
    std::vector<Scalar> elbo;

    struct exp_op_t {
        const Scalar operator()(const Scalar &x) const { return fasterexp(x); }
    } exp_op;
};

template <typename F, typename F0, typename S, typename Derived>
void
clustering_by_lcvi(clustering_status_t<S> &status,
                   const Eigen::MatrixBase<Derived> &X,
                   const std::size_t Ltrunc,
                   const double alpha,
                   const double a0,
                   const double b0,
                   const std::size_t rseed = 42,
                   const std::size_t mcmc = 100,
                   const std::size_t burnin = 10,
                   const bool verbose = true)
{

    const Index N = X.rows();
    const Index D = X.cols();

    if (verbose) {
        TLOG("Truncation level = " << Ltrunc);
    }

    using RowVec = typename Eigen::internal::plain_row_type<Derived>::type;
    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);

    softmax_op_t<Mat> softmax;
    row_sampler_t<Mat, RNG> sampler(rng, Ltrunc);
    RowVec row_prob(Ltrunc);
    row_prob.setOnes();
    row_prob /= row_prob.sum();

    F0 prior(alpha, Ltrunc);

    std::vector<Index> cindex(Ltrunc);
    std::iota(cindex.begin(), cindex.end(), 0);
    std::vector<F> components;

    auto _add_c = [&D, &a0, &b0](const auto &) { return F(D, a0, b0); };

    std::transform(cindex.begin(),
                   cindex.end(),
                   std::back_inserter(components),
                   _add_c);

    if (verbose) {
        TLOG("Clustering the rows of " << N << " x " << D << " matrix");
    }

    Mat log_prob(N, Ltrunc);
    Mat prob_01(N, Ltrunc);
    Mat param(D, Ltrunc);
    Mat log_param(D, Ltrunc);

    std::vector<Index> membership(N);

    for (Index r = 0; r < N; ++r) {
        const Index k = sampler(row_prob);
        components[k].add_point(X.row(r));
        prior.add_to(k);
        membership[r] = k;
    }

    auto &elbo_vec = status.elbo;
    elbo_vec.reserve(mcmc + burnin);

    for (Index t = 0; t < (mcmc + burnin); ++t) {

        Scalar elbo = 0.;

        for (Index r = 0; r < N; ++r) {
            const Index k_old = membership.at(r);
            components[k_old].remove_point(X.row(r));
            prior.subtract_from(k_old);
            for (Index k = 0; k < Ltrunc; ++k) {
                log_prob(r, k) = components[k].log_predictive(X.row(r));
            }
            row_prob = softmax(log_prob.row(r));
            const Index k_new = sampler(row_prob);
            components[k_new].add_point(X.row(r));
            prior.add_to(k_new);
            membership[r] = k_new;

            elbo += log_prob(r, k_new);

            prob_01.row(r).setZero();
            prob_01(r, k_new) = 1.;
        }

        elbo_vec.emplace_back(elbo);
        if (verbose) {
            TLOG("Clustering by MCMC t = " << t << ", " << elbo);
        } else {
            if (t % 10 == 0)
                Rcpp::Rcerr << ". " << std::flush;
            if (t > 0 && t % 100 == 0)
                Rcpp::Rcerr << std::endl;
        }
        if (t > burnin) {
            status.record_latent(prob_01);
            for (Index k = 0; k < Ltrunc; ++k) {
                param.col(k) = components[k].posterior_mean().transpose();
                log_param.col(k) =
                    components[k].posterior_log_mean().transpose();
            }
            status.record_parameter(param);
            status.record_log_parameter(log_param);
        }
        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            WLOG("Interruption by a user at t=" << t);
            break;
        }
    }

    Rcpp::Rcerr << std::endl;

    if (verbose) {
        TLOG("Done -- lcvi");
    }
}

#endif
