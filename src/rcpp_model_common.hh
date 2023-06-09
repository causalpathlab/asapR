#include "mmutil.hh"

#ifndef RCPP_MODEL_COMMON_HH_
#define RCPP_MODEL_COMMON_HH_

struct STOCH {
    explicit STOCH(const bool v)
        : val(v)
    {
    }
    const bool val;
};

struct STD {
    explicit STD(const bool v)
        : val(v)
    {
    }
    const bool val;
};

struct DO_SVD {
    explicit DO_SVD(const bool v)
        : val(v)
    {
    }
    const bool val;
};

struct RSEED : check_positive_t<std::size_t> {
    explicit RSEED(const std::size_t c)
        : check_positive_t<std::size_t>(c)
    {
    }
};

struct NumFact : check_positive_t<std::size_t> {
    explicit NumFact(const std::size_t k)
        : check_positive_t<std::size_t>(k)
    {
    }
};

template <typename MODEL, typename Derived>
typename MODEL::Scalar
log_likelihood(MODEL &model, const Eigen::MatrixBase<Derived> &Y_dn)
{
    return log_likelihood(typename MODEL::tag(), model, Y_dn);
}

template <typename MODEL, typename Derived>
void
initialize_stat(MODEL &model,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_SVD &do_svd)
{
    initialize_stat(typename MODEL::tag(), model, Y_dn, do_svd);
}

template <typename MODEL, typename Derived>
void
add_stat_by_col(MODEL &model,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const STOCH &stoch_,
                const STD &std_)
{
    add_stat_by_col(typename MODEL::tag(), model, Y_dn, stoch_, std_);
}

template <typename MODEL, typename Derived>
void
add_stat_by_row(MODEL &model,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const STOCH &stoch_,
                const STD &std_)
{
    add_stat_by_row(typename MODEL::tag(), model, Y_dn, stoch_, std_);
}

#endif
