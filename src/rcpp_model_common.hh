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

struct DO_AUX_STD {
    explicit DO_AUX_STD(const bool v)
        : val(v)
    {
    }
    const bool val;
};

struct DO_DEGREE_CORRECTION {
    explicit DO_DEGREE_CORRECTION(const bool v)
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

struct NThreads : check_positive_t<std::size_t> {
    explicit NThreads(const std::size_t k)
        : check_positive_t<std::size_t>(k)
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
log_likelihood(const MODEL &model, const Eigen::MatrixBase<Derived> &Y_dn)
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
add_stat_to_col(MODEL &model,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_AUX_STD &std_,
                const DO_DEGREE_CORRECTION &dc_)
{
    add_stat_to_col(typename MODEL::tag(), model, Y_dn, std_, dc_);
}

template <typename MODEL, typename Derived>
void
add_stat_to_row(MODEL &model,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_AUX_STD &std_,
                const DO_DEGREE_CORRECTION &dc_)
{
    add_stat_to_row(typename MODEL::tag(), model, Y_dn, std_, dc_);
}

template <typename MODEL, typename Derived>
void
add_stat_to_mid(MODEL &model,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_AUX_STD &std_,
                const DO_DEGREE_CORRECTION &dc_)
{
    add_stat_to_mid(typename MODEL::tag(), model, Y_dn, std_, dc_);
}

template <typename MODEL, typename Derived>
void
add_stat_to_col(MODEL &model,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_AUX_STD &std_)
{
    add_stat_to_col(typename MODEL::tag(),
                    model,
                    Y_dn,
                    std_,
                    DO_DEGREE_CORRECTION(true));
}

template <typename MODEL, typename Derived>
void
add_stat_to_row(MODEL &model,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_AUX_STD &std_)
{
    add_stat_to_row(typename MODEL::tag(),
                    model,
                    Y_dn,
                    std_,
                    DO_DEGREE_CORRECTION(true));
}

template <typename MODEL, typename Derived>
void
add_stat_to_mid(MODEL &model,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_AUX_STD &std_)
{
    add_stat_to_mid(typename MODEL::tag(),
                    model,
                    Y_dn,
                    std_,
                    DO_DEGREE_CORRECTION(true));
}

#endif
