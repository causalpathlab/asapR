#include "mmutil.hh"
#include "svd.hh"

#ifndef POISSON_MODULAR_NMF_MODEL_HH_
#define POISSON_MODULAR_NMF_MODEL_HH_

template <typename T, typename RNG, typename PARAM>
struct poisson_modular_nmf_t {

    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;

    explicit poisson_modular_nmf_t(const Index d,
                                   const Index n,
                                   const Index k,
                                   const Index l,
                                   const Scalar a0,
                                   const Scalar b0,
                                   std::size_t rseed = 42)
        : D(d)
        , N(n)
        , K(k)
        , L(l)
        , onesD(D, 1)
        , onesN(N, 1)
        , onesL(L, 1)
        , tempK(K)
        , tempK2(K)
        , tempL(L)
        , tempL2(L)
        , tempDL(D, L)
        , tempDK(D, K)
        , tempNL(N, L)
        , tempNK(N, K)
        , tempLK(L, K)
        , rng(rseed)
        , row_D(D, 1, a0, b0, rng)
        , column_N(N, 1, a0, b0, rng)
        , row_DL(D, L, a0, b0, rng)
        , middle_LK(L, K, a0, b0, rng)
        , column_NK(N, K, a0, b0, rng)
        , loading_K(K, 1, a0, b0, rng)
    {
        onesD.setOnes();
        onesN.setOnes();
        onesL.setOnes();
    }

    void randomize_topics()
    {
        Mat row_a = T::Ones(D, L);
        Mat row_b = row_DL.sample();
        row_DL.update(row_a, row_b);

        Mat middle_a = T::Ones(L, K);
        Mat middle_b = middle_LK.sample();
        middle_LK.update(middle_a, middle_b);

        Mat column_a = T::Ones(N, K);
        Mat column_b = column_NK.sample();
        column_NK.update(column_a, column_b);
    }

    template <typename Derived>
    void initialize_by_svd(const Eigen::MatrixBase<Derived> &Y)
    {
        const std::size_t lu_iter = 5; // this should be good

        RandomizedSVD<T> svd(K, lu_iter); //
        const Mat yy = standardize(Y.unaryExpr(log1p_op));
        svd.compute(Y);

        Mat row_a(D, L);
        row_a.setZero();

        for (Index j = 0; j < std::min(L, K); ++j) {
            row_a.col(j) = svd.matrixU().col(j);
        }

        row_DL.update(standardize(row_a).unaryExpr(exp_op) /
                          static_cast<Scalar>(D),
                      T::Ones(D, L));

        middle_LK.update(T::Ones(L, K) / static_cast<Scalar>(L * K),
                         T::Ones(L, K));

        column_NK.update(standardize(svd.matrixV()).unaryExpr(exp_op) /
                             static_cast<Scalar>(N),
                         T::Ones(N, K));
    }

    template <typename Derived>
    void initialize_degree(const Eigen::MatrixBase<Derived> &Y)
    {
        column_N.update(Y.transpose() * onesD, onesN);
        column_N.calibrate();
        row_D.update(Y * onesN, onesD * column_N.mean().sum());
        row_D.calibrate();
    }

    template <typename Derived>
    void update_degree(const Eigen::MatrixBase<Derived> &Y)
    {
        // const Scalar row_sum = row_D.mean().sum();
        column_N.update(Y.transpose() * onesD,
                        column_NK.mean() * middle_LK.mean().transpose() *
                            row_DL.mean().transpose() * row_D.mean());
        column_N.calibrate();

        // const Scalar col_sum = column_N.mean().sum();
        row_D.update(Y * onesN,
                     row_DL.mean() * middle_LK.mean() *
                         column_NK.mean().transpose() * column_N.mean());
        row_D.calibrate();
    }

    template <typename Derived1, typename Derived2, typename Latent>
    Scalar log_likelihood(const Eigen::MatrixBase<Derived1> &Y_DN,
                          const Eigen::MatrixBase<Derived2> &C_DL,
                          const Latent &latent_NL)
    {
        Scalar term0 = 0., term1 = 0., term2 = 0.;

        term0 += (Y_DN.array().colwise() * get_row_logD().array()).sum();

        term0 +=
            (Y_DN.array().rowwise() * get_col_logN().transpose().array()).sum();

        tempNL = Y_DN.transpose() * C_DL;

        for (Index kk = 0; kk < K; ++kk) {

            term1 += latent_NL.slice_k(tempNL, kk).sum() * get_loading_K(kk);

            term1 +=
                (latent_NL.slice_k(tempNL, kk) * middle_LK.log_mean().col(kk))
                    .sum();

            term1 += (column_NK.log_mean().col(kk).transpose() *
                      latent_NL.slice_k(tempNL, kk))
                         .sum();
        }

        tempDL = row_DL.mean().array().colwise() * get_row_D().array();
        tempNK = column_NK.mean().array().colwise() * get_col_N().array();
        tempLK = middle_LK.mean().array().rowwise() *
            get_loading_K().transpose().array();

        term2 += (tempDL * tempLK * tempNK.transpose()).sum();

        return term0 + term1 - term2;
    }

    template <typename Derived1, typename Derived2, typename Latent>
    void update_loading_K(const Eigen::MatrixBase<Derived1> &Y_DN,
                          const Eigen::MatrixBase<Derived2> &C_DL,
                          const Latent &latent_NL)
    {
        tempNL = Y_DN.transpose() * C_DL;
        for (Index k = 0; k < K; ++k) {
            tempK(k) = latent_NL.slice_k(tempNL, k).sum();
        }

        tempDL =
            (row_DL.mean().array().colwise() * get_row_D().array()).matrix();

        tempNK =
            (column_NK.mean().array().colwise() * get_col_N().array()).matrix();

        tempK2 = (tempDL * middle_LK.mean())
                     .colwise()
                     .sum()
                     .transpose()
                     .cwiseProduct(tempNK.colwise().sum().transpose());

        loading_K.update(tempK, tempK2);
        loading_K.calibrate();
    }

    // row ~ Gamma(a, b)
    //  D x L
    // a0 +   C   * sum_k Y  %*% (Z == k)
    //      D x L        D x N     N x L
    // b0 + row.deg %*% t(E[middle] %*% E[column] %*% col.deg)
    //       D x 1         L x K         K x N       N x 1
    template <typename Derived1, typename Derived2, typename Latent>
    void update_row_topic(const Eigen::MatrixBase<Derived1> &Y_DN,
                          const Eigen::MatrixBase<Derived2> &C_DL,
                          const Latent &latent_NL)
    {
        tempK = column_NK.mean().transpose() * get_col_N();
        tempL = middle_LK.mean() * tempK.cwiseProduct(get_loading_K());

        tempDL.setZero();
        for (Index k = 0; k < K; ++k) {
            tempDL += C_DL.cwiseProduct(latent_NL.mult_slice_k(Y_DN, k));
        }

        for (Index l = 0; l < L; ++l) {
            row_DL.update_col(tempDL.col(l), row_D.mean() * tempL(l), l);
        }
        row_DL.calibrate();
    }

    // middle ~ Gamma(a, b)
    //  L x K
    // a0 +  t(C)  %*%  Y  %*% (Z == k) * 1
    //        L x D    D x N     N x L    L x 1
    // a0 + t((t(Y) %*% C) * (Z == k)) * 1
    //          N x L         N x L       N x 1
    // b0  + E[row] %*% row.deg * t(E[column_k]) %*% E[column.deg]
    //        L x D      D x 1        1 x N            N x 1
    template <typename Derived1, typename Derived2, typename Latent>
    void update_middle_topic(const Eigen::MatrixBase<Derived1> &Y_DN,
                             const Eigen::MatrixBase<Derived2> &C_DL,
                             const Latent &latent_NL)
    {
        tempK = (column_NK.mean().transpose() * get_col_N())
                    .cwiseProduct(get_loading_K());
        tempNL = Y_DN.transpose() * C_DL;
        tempL = row_DL.mean().transpose() * get_row_D();

        for (Index kk = 0; kk < K; ++kk) {
            middle_LK.update_col(latent_NL.slice_k(tempNL, kk).transpose() *
                                     onesN,
                                 tempL * tempK(kk),
                                 kk);
        }
        middle_LK.calibrate();
    }

    // column ~ Gamma(a, b)
    //  N x K
    // a0 + ( t(Y) %*%  C )  * (Z == k) %*% 1
    //       N x D    D x L      N x L     L x 1
    // b0 + column.degree * t(t(E[middle]) %*% t(E[row]) %*% row.deg)
    //        N x 1         t(  K x L           L x D          D x 1)
    template <typename Derived1, typename Derived2, typename Latent>
    void update_column_topic(const Eigen::MatrixBase<Derived1> &Y_DN,
                             const Eigen::MatrixBase<Derived2> &C_DL,
                             const Latent &latent_NL)
    {
        tempNL = Y_DN.transpose() * C_DL;
        tempK = (get_row_D().transpose() * row_DL.mean() * middle_LK.mean())
                    .transpose()
                    .cwiseProduct(get_loading_K());

        for (Index kk = 0; kk < K; ++kk) {
            column_NK.update_col(latent_NL.slice_k(tempNL, kk) * onesL,
                                 column_N.mean() * tempK(kk),
                                 kk);
        }
        column_NK.calibrate();
    }

    inline const auto get_loading_K() const { return loading_K.mean().col(0); }

    inline Scalar get_loading_K(const Index k) const
    {
        return loading_K.mean().coeff(k, 0);
    }

    inline const auto get_loading_logK() const
    {
        return loading_K.log_mean().col(0);
    }

    inline Scalar get_loading_logK(const Index k) const
    {
        return loading_K.log_mean().coeff(k, 0);
    }

    const auto get_row_D() const { return row_D.mean().col(0); }

    const auto get_row_logD() const { return row_D.log_mean().col(0); }

    const auto get_col_N() const { return column_N.mean().col(0); }

    const auto get_col_logN() const { return column_N.log_mean().col(0); }

    const Index D, N, K, L;

private:
    T onesD;
    T onesN;
    T onesL;
    ColVec tempK;
    ColVec tempK2;
    ColVec tempL;
    ColVec tempL2;
    Mat tempDL;
    Mat tempDK;
    Mat tempNL;
    Mat tempNK;
    Mat tempLK;
    RNG rng;

public:
    PARAM row_D;
    PARAM column_N;

    PARAM row_DL;
    PARAM middle_LK;
    PARAM column_NK;
    PARAM loading_K;

private:
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
