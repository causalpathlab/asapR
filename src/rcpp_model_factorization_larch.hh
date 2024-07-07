#ifndef RCPP_MODEL_FACTORIZATION_LARCH_HH_
#define RCPP_MODEL_FACTORIZATION_LARCH_HH_

struct factorization_larch_tag { };

template <typename ROW, typename COL, typename RNG>
struct factorization_larch_t {

    using Type = typename ROW::Type;
    using Index = typename Type::Index;
    using Scalar = typename Type::Scalar;
    using RowVec = typename Eigen::internal::plain_row_type<Type>::type;

    using RNG_TYPE = RNG;
    using tag = factorization_larch_tag;

    template <typename Derived>
    explicit factorization_larch_t(ROW &_row_dl,
                                   COL &_col_nk,
                                   const Eigen::MatrixBase<Derived> &_A_lk,
                                   const RSEED &rseed)
        : beta_dl(_row_dl)
        , theta_nk(_col_nk)
        , A_lk(_A_lk)
        , W_l(A_lk.transpose().colwise().sum())
        , V_k(A_lk.colwise().sum())
        , D(beta_dl.rows())
        , N(theta_nk.rows())
        , L(beta_dl.cols())
        , K(theta_nk.cols())
        // , logRow_aux_dl(D, L)
        // , row_aux_dl(D, L)
        , logRow_aux_dk(D, K)
        , row_aux_dk(D, K)
        , logCol_aux_nk(N, K)
        , col_aux_nk(N, K)
        // , std_log_row_aux_dl(logRow_aux_dl, 1, 1)
        , std_log_row_aux_dk(logRow_aux_dk, 1, 1)
        , std_log_col_aux_nk(logCol_aux_nk, 1, 1)
        , rng(rseed.val)
        , num_threads(1)
    {
        ASSERT(theta_nk.cols() == A_lk.cols(),
               "col factors and A_lk must have the same number of columns");
        ASSERT(beta_dl.cols() == A_lk.rows(),
               "row factors and A_lk must be compatible");

        randomize_auxiliaries();
    }

    template <typename Derived>
    explicit factorization_larch_t(ROW &_row_dl,
                                   COL &_col_nk,
                                   const Eigen::MatrixBase<Derived> &_A_lk,
                                   const RSEED &rseed,
                                   const NThreads &nthr)
        : beta_dl(_row_dl)
        , theta_nk(_col_nk)
        , A_lk(_A_lk)
        , W_l(A_lk.transpose().colwise().sum())
        , V_k(A_lk.colwise().sum())
        , D(beta_dl.rows())
        , N(theta_nk.rows())
        , L(beta_dl.cols())
        , K(theta_nk.cols())
        // , logRow_aux_dl(D, L)
        // , row_aux_dl(D, L)
        , logRow_aux_dk(D, K)
        , row_aux_dk(D, K)
        , logCol_aux_nk(N, K)
        , col_aux_nk(N, K)
        // , std_log_row_aux_dl(logRow_aux_dl, 1, 1)
        , std_log_row_aux_dk(logRow_aux_dk, 1, 1)
        , std_log_col_aux_nk(logCol_aux_nk, 1, 1)
        , rng(rseed.val)
        , num_threads(nthr.val)
    {
        ASSERT(theta_nk.cols() == A_lk.cols(),
               "col factors and A_lk must have the same number of columns");
        ASSERT(beta_dl.cols() == A_lk.rows(),
               "row factors and A_lk must be compatible");

        randomize_auxiliaries();
    }

public:
    ROW &beta_dl;
    COL &theta_nk;

    const Mat A_lk;
    const RowVec W_l;
    const RowVec V_k;

    const Index D;
    const Index N;
    const Index L;
    const Index K;

public:
    // Type logRow_aux_dl;
    // Type row_aux_dl;
    Type logRow_aux_dk;
    Type row_aux_dk;
    Type logCol_aux_nk;
    Type col_aux_nk;

    // stdizer_t<Type> std_log_row_aux_dl;
    stdizer_t<Type> std_log_row_aux_dk;
    stdizer_t<Type> std_log_col_aux_nk;

private:
    RNG rng;

public:
    const std::size_t num_threads;

private:
    softmax_op_t<Type> softmax;

public:
    void _row_factor_aux(const bool do_stdize)
    {

        if (do_stdize) {
            // std_log_row_aux_dl.colwise();
            std_log_row_aux_dk.colwise();
        }

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
        for (Index ii = 0; ii < D; ++ii) {
            if (do_stdize) {
                // logRow_aux_dl.row(ii) = softmax.log_row(logRow_aux_dl.row(ii));
                logRow_aux_dk.row(ii) = softmax.log_row(logRow_aux_dk.row(ii));
            } else {
                // logRow_aux_dl.row(ii) =
                //     softmax.log_row_weighted(logRow_aux_dl.row(ii), W_l);
                logRow_aux_dk.row(ii) =
                    softmax.log_row_weighted(logRow_aux_dk.row(ii), V_k);
            }
        }
        // row_aux_dl = logRow_aux_dl.unaryExpr(exp);
        row_aux_dk = logRow_aux_dk.unaryExpr(exp);
    }

    void _col_factor_aux(const bool do_stdize)
    {
        if (do_stdize) {
            std_log_col_aux_nk.colwise();
        }

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
        for (Index jj = 0; jj < N; ++jj) {
            if (do_stdize) {
                logCol_aux_nk.row(jj) =
                    softmax.log_row_weighted(logCol_aux_nk.row(jj), V_k);
            } else {
                logCol_aux_nk.row(jj) = softmax.log_row(logCol_aux_nk.row(jj));
            }
        }

        col_aux_nk = logCol_aux_nk.unaryExpr(exp);
    }

private:
    void randomize_auxiliaries()
    {
        // logRow_aux_dl = Type::Random(D, L);
        // std_log_row_aux_dl.colwise();

        // for (Index ii = 0; ii < D; ++ii) {
        //     row_aux_dl.row(ii) = softmax.apply_row(logRow_aux_dl.row(ii));
        // }

        logRow_aux_dk = Type::Random(D, K);
        std_log_row_aux_dk.colwise();

        for (Index ii = 0; ii < D; ++ii) {
            row_aux_dk.row(ii) = softmax.apply_row(logRow_aux_dk.row(ii));
        }

        logCol_aux_nk = Type::Random(N, K);
        std_log_col_aux_nk.colwise();

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
log_likelihood(const factorization_larch_tag,
               const MODEL &fact,
               const Eigen::MatrixBase<Derived> &Y_dn)

{
    const auto D = fact.D;
    const auto N = fact.N;
    const auto K = fact.K;
    typename MODEL::Type::Scalar llik = 0;
    typename MODEL::Type::Scalar denom = N * D;

    safe_log_op<typename MODEL::Type> safe_log(1e-8);

    llik += Y_dn.cwiseProduct(fact.beta_dl.mean() * fact.A_lk *
                              fact.theta_nk.mean().transpose())
                .unaryExpr(safe_log)
                .sum() /
        denom;

    llik -= (fact.beta_dl.mean() * fact.A_lk * fact.theta_nk.mean().transpose())
                .sum() /
        denom;

    return llik;
}

template <typename MODEL, typename Derived>
void initialize_stat(const factorization_larch_tag,
                     MODEL &fact,
                     const Eigen::MatrixBase<Derived> &Y_dn,
                     const DO_SVD &do_svd);

template <typename MODEL, typename Derived>
void
_initialize_stat_random(const factorization_larch_tag,
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
    fact.beta_dl.calibrate();

    fact.theta_nk.reset_stat_only();
    fact.theta_nk.calibrate();
}

template <typename MODEL, typename Derived>
void
_initialize_stat_svd(const factorization_larch_tag,
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
    RandomizedSVD<Derived> svd(L, lu_iter);
    svd.compute(yy);

    {
        T a = svd.matrixU().unaryExpr(at_least_zero);
        T b = T::Ones(D, L);
        fact.beta_dl.update(a, b);
    }

    fact.beta_dl.calibrate();

    fact.theta_nk.reset_stat_only();
    fact.theta_nk.calibrate();
}

template <typename MODEL, typename Derived>
void
initialize_stat(const factorization_larch_tag,
                MODEL &fact,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_SVD &do_svd)
{
    if (do_svd.val) {
        _initialize_stat_svd(factorization_larch_tag(), fact, Y_dn);
    } else {
        _initialize_stat_random(factorization_larch_tag(), fact, Y_dn);
    }
}

template <typename MODEL, typename Derived>
void
add_stat_to_row(const factorization_larch_tag,
                MODEL &fact,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_AUX_STD &std_,
                const DO_DEGREE_CORRECTION &dc_)
{

    const bool do_stdize = std_.val;
    const bool do_dc = dc_.val;

    using T = typename MODEL::Type;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;
    using RowVec = typename Eigen::internal::plain_row_type<T>::type;

    at_least_one_op<T> at_least_one;

    //////////////////////////////////////////////
    // Estimation of auxiliary variables (i,l)  //
    //////////////////////////////////////////////

    // fact.logRow_aux_dl =
    //     Y_dn * fact.theta_nk.log_mean() * fact.A_lk.transpose();

    // fact.logRow_aux_dl.array().colwise() /=
    //     Y_dn.rowwise().sum().unaryExpr(at_least_one).array();

    // fact.logRow_aux_dl.array().rowwise() /= fact.W_l.array();

    // fact.logRow_aux_dl += fact.beta_dl.log_mean();

    //////////////////////////////////////////////
    // Estimation of auxiliary variables (i,k)  //
    //////////////////////////////////////////////

    fact.logRow_aux_dk =
        ((Y_dn * fact.theta_nk.log_mean()).array().colwise() /
         (Y_dn.rowwise().sum().unaryExpr(at_least_one).array()))
            .matrix();

    fact.logRow_aux_dk +=
        ((fact.beta_dl.log_mean() * fact.A_lk).array().rowwise() /
         fact.V_k.array())
            .matrix();

    fact._row_factor_aux(do_stdize);

    ///////////////////////////
    // Accumulate statistics //
    ///////////////////////////

    if (do_dc) {

        // ((fact.row_aux_dl.array().rowwise() * fact.W_l.array()).colwise() *
        //  Y_dn.rowwise().sum().array())
        //     .matrix();

        fact.beta_dl
            .add(((fact.row_aux_dk * fact.A_lk.transpose()).array().colwise() *
                  Y_dn.rowwise().sum().array())
                     .matrix(),
                 Y_dn.rowwise().mean() *
                     (Y_dn.colwise().mean() * fact.theta_nk.mean() *
                      fact.A_lk.transpose()));

    } else {

        fact.beta_dl
            .add(((fact.row_aux_dk * fact.A_lk.transpose()).array().colwise() *
                  Y_dn.rowwise().sum().array())
                     .matrix(),
                 ColVec::Ones(fact.D) *
                     (fact.theta_nk.mean() * fact.A_lk.transpose())
                         .colwise()
                         .sum());
    }
}

template <typename MODEL, typename Derived>
void
add_stat_to_col(const factorization_larch_tag,
                MODEL &fact,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_AUX_STD &std_,
                const DO_DEGREE_CORRECTION &dc_)
{
    const bool do_stdize = std_.val;
    const bool do_dc = dc_.val;

    using T = typename MODEL::Type;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;

    at_least_one_op<T> at_least_one;

    //////////////////////////////////////////////
    // Estimation of auxiliary variables (j,k)  //
    //////////////////////////////////////////////

    fact.logCol_aux_nk = Y_dn.transpose() * fact.beta_dl.log_mean() * fact.A_lk;

    fact.logCol_aux_nk.array().colwise() /=
        Y_dn.colwise().sum().transpose().unaryExpr(at_least_one).array();

    fact.logCol_aux_nk.array().rowwise() /= fact.V_k.array();

    fact.logCol_aux_nk += fact.theta_nk.log_mean();

    fact._col_factor_aux(do_stdize);

    ///////////////////////////
    // Accumulate statistics //
    ///////////////////////////

    if (do_dc) {

        fact.theta_nk.add(((fact.col_aux_nk.array().rowwise() *
                            fact.V_k.array())
                               .colwise() *
                           Y_dn.transpose().rowwise().sum().array())
                              .matrix(),
                          Y_dn.transpose().rowwise().mean() *
                              (Y_dn.transpose().colwise().mean() *
                               fact.beta_dl.mean() * fact.A_lk));

    } else {

        fact.theta_nk
            .add(((fact.col_aux_nk.array().rowwise() * fact.V_k.array())
                      .colwise() *
                  Y_dn.transpose().rowwise().sum().array())
                     .matrix(),
                 ColVec::Ones(fact.N) *
                     (fact.beta_dl.mean() * fact.A_lk).colwise().sum());
    }
}

#endif
