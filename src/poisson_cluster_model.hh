#include "mmutil.hh"

#ifndef POISSON_CLUSTER_MODEL_HH_
#define POISSON_CLUSTER_MODEL_HH_

template <typename T>
struct poisson_component_t {

    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;
    using RowVec = typename Eigen::internal::plain_row_type<T>::type;

    explicit poisson_component_t(const Index d,
                                 const Scalar _a0,
                                 const Scalar _b0)
        : dim(d)
        , dd(d)
        , a0(_a0)
        , b0(_b0)
        , Freq_stat(dim)
        , N_stat(0)
        , lgamma_op(_a0)
        , digamma_op(_a0)
    {
        clear();
    }

    void clear()
    {
        Freq_stat.setZero();
        N_stat = 0;
    }

    // X: sample x dimension
    template <typename Derived>
    void add_point(const Eigen::MatrixBase<Derived> &X)
    {
        Freq_stat += X.colwise().sum();
        N_stat += X.rows();
    }

    template <typename Derived>
    void remove_point(const Eigen::MatrixBase<Derived> &X)
    {
        Freq_stat -= X.colwise().sum();
        N_stat -= X.rows();
    }

    // sum_g lgamma(F[g] + a0 + x[r, g]) - lgamma(F[g] + a0)
    // - sum_g (x[r, g] + F[g] + a0) * log(b0 + N + 1)
    // + sum_g (F[g] + a0) * log(b0 + N)
    template <typename Derived>
    Derived log_predictive_row(const Eigen::MatrixBase<Derived> &X) const
    {
        return ((X.rowwise() + Freq_stat).unaryExpr(lgamma_op).rowwise() -
                Freq_stat.unaryExpr(lgamma_op))
                   .rowwise()
                   .sum() +
            ((((X.rowwise() + Freq_stat).array() + a0) *
              (-fasterlog(b0 + N_stat + 1.)))
                 .matrix()
                 .rowwise() +
             ((Freq_stat.array() + a0) * fasterlog(N_stat + b0)).matrix())
                .rowwise()
                .sum();
    }

    template <typename Derived>
    const Scalar log_predictive(const Eigen::MatrixBase<Derived> &xx) const
    {
        const Scalar term1 = ((xx + Freq_stat).unaryExpr(lgamma_op) -
                              Freq_stat.unaryExpr(lgamma_op))
                                 .sum();

        const Scalar term2 =
            ((xx + Freq_stat).array() + a0).sum() * fasterlog(b0 + N_stat + 1.);

        const Scalar term3 =
            (Freq_stat.array() + a0).sum() * fasterlog(N_stat + b0);

        return term1 - term2 + term3;
    }

    const Index dim;
    const Scalar dd, a0, b0;

    RowVec posterior_mean() const
    {
        return (Freq_stat.array() + a0).matrix() / (N_stat + b0);
    }

    RowVec posterior_log_mean() const
    {
        return (Freq_stat.unaryExpr(digamma_op).array() -
                fasterlog(N_stat + b0))
            .matrix();
    }

private:
    RowVec Freq_stat;
    Scalar N_stat;

    struct lgamma_op_t {
        explicit lgamma_op_t(const Scalar _a0)
            : a0(_a0)
        {
        }
        const Scalar operator()(const Scalar &xx) const
        {
            return fasterlgamma(a0 + xx);
        }
        const Scalar a0;
    };

    lgamma_op_t lgamma_op;

    struct digamma_op_t {
        explicit digamma_op_t(const Scalar _a0)
            : a0(_a0)
        {
        }
        const Scalar operator()(const Scalar &xx) const
        {
            return fasterdigamma(a0 + xx);
        }
        const Scalar a0;
    };

    digamma_op_t digamma_op;
};

#endif
