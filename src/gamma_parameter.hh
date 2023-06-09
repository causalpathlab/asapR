#include "mmutil.hh"

#ifndef GAMMA_PARAMETER_HH_
#define GAMMA_PARAMETER_HH_

template <typename T, typename RNG>
struct gamma_param_t {

    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    using Type = T;

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
        , estimate_mean(r, c)
        , estimate_sd(r, c)
        , estimate_log(r, c)
        , estimate_log_sd(r, c)
        , rgamma_op(_rng)
    {
        a_stat.setConstant(a0);
        b_stat.setConstant(b0);
        calibrate();
    }

    void calibrate()
    {
        estimate_mean = a_stat.cwiseQuotient(b_stat);
        estimate_sd = a_stat.binaryExpr(b_stat, estimate_sd_op);
        estimate_log = a_stat.binaryExpr(b_stat, estimate_log_op);
        estimate_log_sd = a_stat.unaryExpr(estimate_sd_log_op);
    }

    const T &mean() const { return estimate_mean; }
    const T &sd() const { return estimate_sd; }
    const T &log_mean() const { return estimate_log; }
    const T &log_sd() const { return estimate_log_sd; }

    void reset_stat_only()
    {
        a_stat.setConstant(a0);
        b_stat.setConstant(b0);
    }

    template <typename Derived1, typename Derived2>
    void add(const Eigen::MatrixBase<Derived1> &add_a,
             const Eigen::MatrixBase<Derived2> &add_b)
    {
        a_stat += add_a;
        b_stat += add_b;
    }

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

    T sample() { return a_stat.binaryExpr(b_stat, rgamma_op); }

    const T &sample_log_mean()
    {
        estimate_mean = a_stat.binaryExpr(b_stat, rgamma_op);
        estimate_log = estimate_mean.unaryExpr(log_op);
        return estimate_log;
    }

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
            return _rgamma(rng, typename gamma_distrib::param_type(a, b));
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

    // Delta method
    // sqrt V[ln(mu)] = sqrt (V[mu] / mu)
    //                = 1/sqrt(a -1 )
    // approximated at the mode = (a - 1)/b
    struct estimate_sd_log_op_t {
        Scalar operator()(const Scalar &a) const
        {
            const Scalar one = 1.0;
            const Scalar zero = 0.0;

            if (a > one)
                return std::max(one / std::sqrt(a - one), zero);

            return std::max(one / std::sqrt(a), zero);
        }
    };

    // sqrt(a) / b
    struct estimate_sd_op_t {
        Scalar operator()(const Scalar &a, const Scalar &b) const
        {
            return std::max(std::sqrt(a) / (b), static_cast<Scalar>(0.));
        }
    };

    struct log_op_t {
        const Scalar operator()(const Scalar &x) const { return fasterlog(x); }
    };

    const Index nrows, ncols;
    const Scalar a0, b0;

    T a_stat;
    T b_stat;
    T estimate_mean;   // E[lambda]
    T estimate_sd;     // SD[lambda]
    T estimate_log;    // E[log lambda]
    T estimate_log_sd; // SD[log lambda]

    rgamma_op_t rgamma_op;
    estimate_log_op_t estimate_log_op;
    estimate_sd_log_op_t estimate_sd_log_op;
    estimate_sd_op_t estimate_sd_op;

    log_op_t log_op;
};

#endif
