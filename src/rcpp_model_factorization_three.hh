#ifndef RCPP_MODEL_FACTORIZATION_THREE_HH_
#define RCPP_MODEL_FACTORIZATION_THREE_HH_

struct factorization_three_tag { };

template <typename ROW, typename MID, typename COL, typename RNG>
struct factorization_three_t {

    using Type = typename ROW::Type;
    using Index = typename Type::Index;
    using Scalar = typename Type::Scalar;
    using RowVec = typename Eigen::internal::plain_row_type<Type>::type;

    using RNG_TYPE = RNG;
    using tag = factorization_three_tag;

    template <typename Derived>
    explicit factorization_three_t(ROW &_row_dl,
                                   MID &_mid_lk,
                                   COL &_col_nk,
                                   const RSEED &rseed)
        : beta_dl(_row_dl)
        , A_lk(_mid_lk)
        , theta_nk(_col_nk)
        , D(beta_dl.rows())
        , N(theta_nk.rows())
        , L(beta_dl.cols())
        , K(theta_nk.cols())
        , logRow_aux_dl(D, L)
        , row_aux_dl(D, L)
        , logMid_aux_lk(D, L)
        , mid_aux_lk(D, L)
        , logCol_aux_nk(N, K)
        , col_aux_nk(N, K)
        , std_log_row_aux_dl(logRow_aux_dl, 1, 1)
        , std_log_mid_aux_lk(logMid_aux_lk, 1, 1)
        , std_log_col_aux_nk(logCol_aux_nk, 1, 1)
        , rng(rseed.val)
    {
        ASSERT(theta_nk.cols() == A_lk.cols(),
               "col factors and A_lk must have the same number of columns");
        ASSERT(beta_dl.cols() == A_lk.rows(),
               "row factors and A_lk must be compatible");

        randomize_auxiliaries();
    }

    ROW &beta_dl;
    MID &A_lk;
    COL &theta_nk;

    const Index D;
    const Index N;
    const Index L;
    const Index K;

    Type logRow_aux_dl;
    Type row_aux_dl;

    Type logMid_aux_lk;
    Type mid_aux_lk;

    Type logCol_aux_nk;
    Type col_aux_nk;

    stdizer_t<Type> std_log_row_aux_dl;
    stdizer_t<Type> std_log_mid_aux_lk;
    stdizer_t<Type> std_log_col_aux_nk;

private:
    RNG rng;
    softmax_op_t<Type> softmax;

    // template <typename Derived>
    // void _normalize_aux_cols(stdizer_t<Derived> &std_log_aux,
    //                          Eigen::MatrixBase<Derived> &log_aux,
    //                          Eigen::MatrixBase<Derived> &aux,
    //                          const bool stoch,
    //                          const bool do_stdize)
    // {
    //     ///////////////////////////////////
    //     // this helps spread the columns //
    //     ///////////////////////////////////

    //     if (do_stdize) {
    //         std_log_aux.colwise();
    //     }

    //     for (Index ii = 0; ii < aux.rows(); ++ii) {
    //         log_aux.row(ii) = softmax.log_row(log_aux.row(ii));
    //     }

    //     //////////////////////////////////////
    //     // stochastic sampling or normalize //
    //     //////////////////////////////////////

    //     if (stoch) {
    //         aux.setZero();
    //         rowvec_sampler_t<Type, RNG> sampler(rng, aux.cols());
    //         for (Index ii = 0; ii < aux.rows(); ++ii) {
    //             auto kk = sampler(log_aux.unaryExpr(exp));
    //             aux(ii, kk) = 1;
    //         }
    //     } else {
    //         aux = log_aux.unaryExpr(exp);
    //     }
    // }

public:
    // void _row_factor_aux(const bool stoch, const bool do_stdize)
    // {
    //     _normalize_aux_cols(std_log_row_aux_dl,
    //                         logRow_aux_dl,
    //                         row_aux_dl,
    //                         stoch,
    //                         do_stdize);
    // }

    // void _col_factor_aux(const bool stoch, const bool do_stdize)
    // {
    //     _normalize_aux_cols(std_log_col_aux_nk,
    //                         logCol_aux_nk,
    //                         col_aux_nk,
    //                         stoch,
    //                         do_stdize);
    // }

private:
    void randomize_auxiliaries()
    {
        logRow_aux_dl = Type::Random(D, L);
        for (Index ii = 0; ii < D; ++ii) {
            row_aux_dl.row(ii) = softmax.apply_row(logRow_aux_dl.row(ii));
        }

        logMid_aux_lk = Type::Random(L, K);
        for (Index ll = 0; ll < L; ++ll) {
            mid_aux_lk.row(ll) = softmax.apply_row(logMid_aux_lk.row(ll));
        }

        logCol_aux_nk = Type::Random(N, K);
        for (Index jj = 0; jj < N; ++jj) {
            col_aux_nk.row(jj) = softmax.apply_row(logCol_aux_nk.row(jj));
        }
    }

private:
    exp_op<Type> exp;
    log1p_op<Type> log1p;
};

template <typename MODEL, typename Derived>
typename MODEL::Scalar
log_likelihood(const factorization_three_tag,
               const MODEL &fact,
               const Eigen::MatrixBase<Derived> &Y_dn)

{
    const auto D = fact.D;
    const auto N = fact.N;
    const auto K = fact.K;
    typename Mat::Scalar llik = 0;
    typename Mat::Scalar denom = N * D;

    // llik += (Y_dn.transpose() *
    //          fact.row_aux_dl.cwiseProduct(fact.beta_dl.log_mean()))
    //             .sum() /
    //     denom;

    // llik += ((fact.row_aux_dl * fact.A_lk)
    //              .cwiseProduct(
    //                  (Y_dn *
    //                   fact.col_aux_nk.cwiseProduct(fact.theta_nk.log_mean()))))
    //             .sum() /
    //     denom;

    // llik -= (fact.beta_dl.mean() * fact.A_lk * fact.theta_nk.mean().transpose())
    //             .sum() /
    //     denom;

    return llik;
}

template <typename MODEL, typename Derived>
void initialize_stat(const factorization_three_tag,
                     MODEL &fact,
                     const Eigen::MatrixBase<Derived> &Y_dn,
                     const DO_SVD &do_svd);

template <typename MODEL, typename Derived>
void
_initialize_stat_random(const factorization_three_tag,
                        MODEL &fact,
                        const Eigen::MatrixBase<Derived> &Y_dn)
{
    using Mat = typename MODEL::Type;
    using Index = typename Mat::Index;
    using Scalar = typename Mat::Scalar;

    const Index D = fact.D;
    const Index N = fact.N;
    const Index K = fact.K;
    const Index L = fact.L;

    Mat temp_dl = fact.beta_dl.sample();
    fact.beta_dl.update(temp_dl, Mat::Ones(D, L));

    Mat temp_lk = fact.A_lk.sample();
    fact.A_lk.update(temp_lk, Mat::Ones(L, K));

    Mat temp_nk = fact.theta_nk.sample();
    fact.theta_nk.update(temp_nk, Mat::Ones(N, K));

    fact.beta_dl.calibrate();
    fact.theta_nk.calibrate();
}

template <typename MODEL, typename Derived>
void
_initialize_stat_svd(const factorization_three_tag,
                     MODEL &fact,
                     const Eigen::MatrixBase<Derived> &Y_dn)
{
    using T = typename MODEL::Type;
    using Index = typename T::Index;
    using Scalar = typename T::Scalar;

    const Index D = fact.D;
    const Index N = fact.N;
    const Index K = fact.K;
    const Index L = fact.L;

    const Scalar lb = -8, ub = 8;
    clamp_op<T> clamp_(lb, ub);
    at_least_zero_op<T> at_least_zero;
    log1p_op<T> log1p;

    T yy = standardize_columns(Y_dn.unaryExpr(at_least_zero).unaryExpr(log1p))
               .unaryExpr(clamp_);

    const std::size_t lu_iter = 5;
    RandomizedSVD<Derived> svd(K, lu_iter);
    svd.compute(yy);

    {
        T temp_dl = fact.beta_dl.sample();
        fact.beta_dl.update(temp_dl, Mat::Ones(D, L));
    }
    {
        T temp_lk = fact.A_lk.sample();
        fact.A_lk.update(temp_lk, Mat::Ones(L, K));
    }
    {
        T a = svd.matrixV().unaryExpr(at_least_zero);
        T b = T::Ones(N, K) / static_cast<Scalar>(N);
        fact.theta_nk.update(a, b);
    }

    fact.beta_dl.calibrate();
    fact.theta_nk.calibrate();
}

template <typename MODEL, typename Derived>
void
initialize_stat(const factorization_three_tag,
                MODEL &fact,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_SVD &do_svd)
{
    if (do_svd.val) {
        _initialize_stat_svd(factorization_three_tag(), fact, Y_dn);
    } else {
        _initialize_stat_random(factorization_three_tag(), fact, Y_dn);
    }
}

template <typename MODEL, typename Derived>
void
add_stat_by_row(const factorization_three_tag,
                MODEL &fact,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const STD &std_)
{

    const bool do_stdize = std_.val;

    using T = typename MODEL::Type;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;

    at_least_one_op<T> at_least_one;

    //////////////////////////////////////////////
    // Estimation of auxiliary variables (i,l)  //
    //////////////////////////////////////////////

    // fact.logRow_aux_dl =
    //     Y_dn * fact.theta_nk.log_mean() * fact.A_lk.transpose();

    // fact.logRow_aux_dl.array().colwise() /=
    //     Y_dn.rowwise().sum().unaryExpr(at_least_one).array();

    // fact.logRow_aux_dl += fact.beta_dl.log_mean();

    // fact._row_factor_aux(stoch, do_stdize);

    ///////////////////////////
    // Accumulate statistics //
    ///////////////////////////

    // fact.row_aux_dl.cwiseProduct(Y_dn * fact.col_aux_nk *
    //                              fact.A_lk.transpose());

    // fact.beta_dl.add((fact.row_aux_dl.array().colwise() *
    //                   Y_dn.rowwise().sum().array())
    //                      .matrix(),
    //                  ColVec::Ones(fact.D) *
    //                      (fact.theta_nk.mean() * fact.A_lk.transpose())
    //                          .colwise()
    //                          .sum());
}

template <typename MODEL, typename Derived>
void
add_stat_by_col(const factorization_three_tag,
                MODEL &fact,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const STD &std_)
{
    const bool do_stdize = std_.val;

    using T = typename MODEL::Type;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;

    at_least_one_op<T> at_least_one;

    //////////////////////////////////////////////
    // Estimation of auxiliary variables (i,l)  //
    //////////////////////////////////////////////

    // fact.logRow_aux_dl =
    //     Y_dn * fact.theta_nk.log_mean() * fact.A_lk.transpose();

    // fact.logRow_aux_dl.array().colwise() /=
    //     Y_dn.rowwise().sum().unaryExpr(at_least_one).array();

    // fact.logRow_aux_dl += fact.beta_dl.log_mean();

    // fact._row_factor_aux(stoch, false);

    //////////////////////////////////////////////
    // Estimation of auxiliary variables (j,k)  //
    //////////////////////////////////////////////

    // fact.logCol_aux_nk = Y_dn.transpose() *
    //     fact.row_aux_dl.cwiseProduct(fact.beta_dl.log_mean()) * fact.A_lk;

    // fact.logCol_aux_nk.array().colwise() /=
    //     Y_dn.colwise().sum().transpose().unaryExpr(at_least_one).array();

    // fact.logCol_aux_nk += fact.theta_nk.log_mean();

    // fact._col_factor_aux(stoch, do_stdize);

    ///////////////////////////
    // Accumulate statistics //
    ///////////////////////////

    // fact.theta_nk.add(fact.col_aux_nk.cwiseProduct(Y_dn.transpose() *
    //                                                fact.row_aux_dl * fact.A_lk),
    //                   ColVec::Ones(fact.N) *
    //                       (fact.beta_dl.mean() * fact.A_lk).colwise().sum());
}

#endif
