#include "mmutil.hh"
#include "svd.hh"

#ifndef POISSON_NMF_MODEL_HH_
#define POISSON_NMF_MODEL_HH_

template <typename T, typename RNG, typename PARAM>
struct poisson_nmf_t {

    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;

    explicit poisson_nmf_t(const Index d,
                           const Index n,
                           const Index k,
                           const Scalar a0,
                           const Scalar b0,
                           std::size_t rseed = 42)
        : D(d)
        , N(n)
        , K(k)
        , onesD(d, 1)
        , onesN(n, 1)
        , tempK(K)
        , rng(rseed)
        , row_degree(d, 1, a0, b0, rng)
        , column_degree(n, 1, a0, b0, rng)
        , row_topic(d, k, a0, b0, rng)
        , column_topic(n, k, a0, b0, rng)
        , topic_loading(k, 1, a0, b0, rng)
    {
        onesD.setOnes();
        onesN.setOnes();
    }

    void randomize_topics()
    {
        Mat row_a = T::Ones(D, K);
        Mat row_b = row_topic.sample();
        row_topic.update(row_a, row_b);

        Mat column_a = T::Ones(N, K);
        Mat column_b = column_topic.sample();
        column_topic.update(column_a, column_b);
    }

    template <typename Derived>
    void initialize_by_svd(const Eigen::MatrixBase<Derived> &Y)
    {
        const std::size_t lu_iter = 5; // this should be good

        RandomizedSVD<T> svd(K, lu_iter); //
        const Mat yy = standardize(Y.unaryExpr(log1p_op));
        svd.compute(yy);

        Mat row_a = standardize(svd.matrixU()).unaryExpr(exp_op) /
            static_cast<Scalar>(D);
        Mat row_b = T::Ones(D, K);
        row_topic.update(row_a, row_b);

        Mat column_a = standardize(svd.matrixV()).unaryExpr(exp_op) /
            static_cast<Scalar>(N);
        Mat column_b = T::Ones(N, K);
        column_topic.update(column_a, column_b);
    }

    template <typename Derived>
    void initialize_degree(const Eigen::MatrixBase<Derived> &Y)
    {
        column_degree.update(Y.transpose() * onesD, onesN);
        column_degree.calibrate();
        row_degree.update(Y * onesN, onesD * column_degree.mean().sum());
        row_degree.calibrate();
    }

    template <typename Derived>
    void update_degree(const Eigen::MatrixBase<Derived> &Y)
    {

        // const Scalar row_sum = row_degree.mean().sum();
        column_degree.update(Y.transpose() * onesD,
                             column_topic.mean() *
                                 (row_topic.mean().transpose() *
                                  row_degree.mean()));
        column_degree.calibrate();

        // const Scalar col_sum = column_degree.mean().sum();
        row_degree.update(Y * onesN,
                          row_topic.mean() *
                              (column_topic.mean().transpose() *
                               column_degree.mean()));
        row_degree.calibrate();
    }

    template <typename Derived, typename Latent>
    const Scalar log_likelihood(const Eigen::MatrixBase<Derived> &Y,
                                const Latent &latent)
    {
        double term0 = 0., term1 = 0., term2 = 0.;

        term0 += (Y.array().colwise() * take_row_log_degree().array()).sum();

        term0 +=
            (Y.array().rowwise() * take_column_log_degree().transpose().array())
                .sum();

        for (Index k = 0; k < K; ++k) {

            term1 += latent.slice_k(Y, k).sum() * take_topic_log_loading(k);

            term1 += (latent.slice_k(Y, k).array().colwise() *
                      row_topic.log_mean().col(k).array())
                         .sum();
            term1 += (latent.slice_k(Y, k).array().rowwise() *
                      column_topic.log_mean().col(k).transpose().array())
                         .sum();
        }

        term2 =
            (((row_topic.mean().array().colwise() * take_row_degree().array())
                  .rowwise() *
              take_topic_loading().transpose().array())
                 .matrix() *
             (column_topic.mean().transpose().array().rowwise() *
              take_column_degree().transpose().array())
                 .matrix())
                .sum();

        return term0 + term1 - term2;
    }

    inline const auto take_row_degree() const
    {
        return row_degree.mean().col(0);
    }

    inline const auto take_topic_loading() const
    {
        return topic_loading.mean().col(0);
    }

    inline const auto take_topic_log_loading() const
    {
        return topic_loading.log_mean().col(0);
    }

    inline Scalar take_topic_loading(const Index k) const
    {
        return topic_loading.mean().coeff(k, 0);
    }

    inline Scalar take_topic_log_loading(const Index k) const
    {
        return topic_loading.log_mean().coeff(k, 0);
    }

    inline const auto take_row_log_degree() const
    {
        return row_degree.log_mean().col(0);
    }

    inline const auto take_column_degree() const
    {
        return column_degree.mean().col(0);
    }

    inline const auto take_column_log_degree() const
    {
        return column_degree.log_mean().col(0);
    }

    template <typename Derived, typename Latent>
    void update_topic_loading(const Eigen::MatrixBase<Derived> &Y,
                              const Latent &latent)
    {
        for (Index k = 0; k < K; ++k) {
            tempK(k) = latent.slice_k(Y, k).sum();
        }
        tempK2 = (row_topic.mean().transpose() * take_row_degree())
                     .cwiseProduct(column_topic.mean().transpose() *
                                   take_column_degree());

        topic_loading.update(tempK, tempK2);
        topic_loading.calibrate();
    }

    template <typename Derived, typename Latent>
    void update_column_topic(const Eigen::MatrixBase<Derived> &Y,
                             const Latent &latent)
    {
        // a[j, k] = sum_i Y[i,j] * (Z[i,j] == k)
        // b[j, k] = d_j * sum_i row[i, k] * topic[k]
        tempK = (row_topic.mean().transpose() * take_row_degree())
                    .cwiseProduct(take_topic_loading());

        for (Index k = 0; k < K; ++k) {
            column_topic.update_col(latent.slice_k(Y, k).transpose() * onesD,
                                    column_degree.mean() * tempK(k),
                                    k);
        }
        column_topic.calibrate();
    }

    template <typename Derived, typename Latent>
    void update_row_topic(const Eigen::MatrixBase<Derived> &Y,
                          const Latent &latent)
    {
        tempK = (column_topic.mean().transpose() * take_column_degree())
                    .cwiseProduct(take_topic_loading());

        for (Index k = 0; k < K; ++k) {
            row_topic.update_col(latent.slice_k(Y, k) * onesN,
                                 row_degree.mean() * tempK(k),
                                 k);
        }
        row_topic.calibrate();
    }

    const Index D, N, K;

    T onesD;
    T onesN;
    ColVec tempK;
    ColVec tempK2;
    RNG rng;

    PARAM row_degree;
    PARAM column_degree;

    PARAM row_topic;
    PARAM column_topic;
    PARAM topic_loading;

    struct log_op_t {
        const Scalar operator()(const Scalar &x) const { return std::log(x); }
    } log_op;

    struct log1p_op_t {
        const Scalar operator()(const Scalar &x) const
        {
            return std::log(1. + x);
        }
    } log1p_op;

    struct exp_op_t {
        const Scalar operator()(const Scalar &x) const { return std::exp(x); }
    } exp_op;
};

#endif
