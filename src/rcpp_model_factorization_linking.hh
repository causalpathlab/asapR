#ifndef RCPP_MODEL_FACTORIZATION_LINKING_HH_
#define RCPP_MODEL_FACTORIZATION_LINKING_HH_

struct factorization_linking_tag { };

template <typename LINKING, typename ROW, typename COL>
struct factorization_linking_t {

    using Type = typename ROW::Type;
    using Index = typename Type::Index;
    using Scalar = typename Type::Scalar;
    using RowVec = typename Eigen::internal::plain_row_type<Type>::type;

    using tag = factorization_linking_tag;

    explicit factorization_linking_t(LINKING &_linking_dl,
                                     ROW &_row_lk,
                                     COL &_col_nk,
                                     const NThreads &nthr)
        : alpha_dl(_linking_dl)
        , beta_lk(_row_lk)
        , theta_nk(_col_nk)
        , D(alpha_dl.rows())
        , L(beta_lk.rows())
        , N(theta_nk.rows())
        , K(beta_lk.cols())
        , logRow_aux_dk(D, K)
        , row_aux_dk(D, K)
        , logRow_linking_aux_dk(D, K)
        , row_linking_aux_dk(D, K)
        , logCol_aux_nk(N, K)
        , col_aux_nk(N, K)
        , temp_k(K)
        , temp_l(L)
        , std_logRow_aux_dk(logRow_aux_dk, 1, 1)
        , std_logRow_linking_aux_dk(logRow_linking_aux_dk, 1, 1)
        , std_logCol_aux_nk(logCol_aux_nk, 1, 1)
        , num_threads(nthr.val)
    {
        ASSERT(theta_nk.cols() == beta_lk.cols(),
               "row and col factors must have the same # columns");
        ASSERT(beta_lk.rows() == alpha_dl.cols(),
               "beta and linking factors must be compatible");

        randomize_auxiliaries();
    }

    explicit factorization_linking_t(LINKING &_linking_dl,
                                     ROW &_row_lk,
                                     COL &_col_nk)
        : factorization_linking_t(_linking_dl, _row_lk, _col_nk, NThreads(1))
    {
    }

    ROW &alpha_dl;
    ROW &beta_lk;
    COL &theta_nk;

    const Index D;
    const Index L;
    const Index N;
    const Index K;

    Type logRow_aux_dk;
    Type row_aux_dk;
    Type logRow_linking_aux_dk;
    Type row_linking_aux_dk;
    Type logCol_aux_nk;
    Type col_aux_nk;

    RowVec temp_k;
    RowVec temp_l;

    stdizer_t<Type> std_logRow_aux_dk;
    stdizer_t<Type> std_logRow_linking_aux_dk;
    stdizer_t<Type> std_logCol_aux_nk;

    const std::size_t num_threads;

    softmax_op_t<Type> softmax;

    // normalize sum_k aux(i, k) sum_l beta(l, k) = 1
    void _row_linking_factor_aux(const bool do_stdize)
    {
        if (do_stdize) {
            std_logRow_linking_aux_dk.colwise();
        }

        temp_k = beta_lk.mean().colwise().sum();

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
        for (Index ii = 0; ii < row_linking_aux_dk.rows(); ++ii) {
            logRow_linking_aux_dk.row(ii) =
                softmax.log_row_weighted(logRow_linking_aux_dk.row(ii), temp_k);
        }
        row_linking_aux_dk = logRow_linking_aux_dk.unaryExpr(exp);
    }

    // normalize sum_k aux(i, k) sum_l alpha(i, l) = 1
    void _row_factor_aux(const bool do_stdize)
    {
        if (do_stdize) {
            std_logRow_aux_dk.colwise();
        }

        // log_aux is accurate only up to a constant factor
#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
        for (Index ii = 0; ii < row_aux_dk.rows(); ++ii) {
            logRow_aux_dk.row(ii) = softmax.log_row(logRow_aux_dk.row(ii));
        }

        row_aux_dk = (logRow_aux_dk.unaryExpr(exp).array().colwise() /
                      alpha_dl.mean().rowwise().sum().array())
                         .matrix();
    }

    // normalize sum aux(j,k) = 1
    void _col_factor_aux(const bool do_stdize)
    {
        if (do_stdize) {
            std_logCol_aux_nk.colwise();
        }

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
        for (Index jj = 0; jj < col_aux_nk.rows(); ++jj) {
            logCol_aux_nk.row(jj) = softmax.log_row(logCol_aux_nk.row(jj));
        }
        col_aux_nk = logCol_aux_nk.unaryExpr(exp);
    }

    void randomize_auxiliaries()
    {
        logRow_aux_dk = Type::Random(D, K);
        for (Index ii = 0; ii < D; ++ii) {
            row_aux_dk.row(ii) = softmax.apply_row(logRow_aux_dk.row(ii));
        }

        logRow_linking_aux_dk = Type::Random(D, K);
        for (Index ii = 0; ii < D; ++ii) {
            row_linking_aux_dk.row(ii) =
                softmax.apply_row(logRow_linking_aux_dk.row(ii));
        }

        logCol_aux_nk = Type::Random(N, K);
        for (Index jj = 0; jj < N; ++jj) {
            col_aux_nk.row(jj) = softmax.apply_row(logCol_aux_nk.row(jj));
        }
    }

    exp_op<Type> exp;
    log1p_op<Type> log1p;
};

template <typename MODEL, typename Derived>
typename MODEL::Scalar
log_likelihood(const factorization_linking_tag,
               const MODEL &fact,
               const Eigen::MatrixBase<Derived> &Y_dn)

{
    const auto D = fact.D;
    const auto N = fact.N;
    const auto K = fact.K;
    typename MODEL::Type::Scalar llik = 0;
    typename MODEL::Type::Scalar denom = N * D;

    safe_log_op<typename MODEL::Type> safe_log(1e-8);

    llik += (Y_dn.cwiseProduct((fact.alpha_dl.mean() * fact.beta_lk.mean() *
                                fact.theta_nk.mean().transpose())
                                   .unaryExpr(safe_log))
                 .sum() /
             denom);

    llik -= ((fact.alpha_dl.mean() * fact.beta_lk.mean() *
              fact.theta_nk.mean().transpose())
                 .sum() /
             denom);

    return llik;
}

template <typename MODEL, typename Derived>
void initialize_stat(const factorization_linking_tag,
                     MODEL &fact,
                     const Eigen::MatrixBase<Derived> &Y_dn,
                     const DO_SVD &do_svd,
                     const typename MODEL::Scalar jitter);

template <typename MODEL, typename Derived>
void
_initialize_stat_random(const factorization_linking_tag,
                        MODEL &fact,
                        const Eigen::MatrixBase<Derived> &Y_dn,
                        const typename MODEL::Scalar jitter)
{
    using Mat = typename MODEL::Type;
    using Index = typename Mat::Index;
    using Scalar = typename Mat::Scalar;

    const Index L = fact.L;
    const Index K = fact.K;

    Mat temp_lk = fact.beta_lk.sample() * jitter;
    fact.beta_lk.update(temp_lk, Mat::Ones(L, K));
    fact.beta_lk.calibrate();

    fact.alpha_dl.reset_stat_only();
    fact.alpha_dl.calibrate();

    fact.theta_nk.reset_stat_only();
    fact.theta_nk.calibrate();
}

template <typename MODEL, typename Derived>
void
_initialize_stat_svd(const factorization_linking_tag,
                     MODEL &fact,
                     const Eigen::MatrixBase<Derived> &Y_dn,
                     const typename MODEL::Scalar jitter)
{
    using T = typename MODEL::Type;
    using Index = typename T::Index;
    using Scalar = typename T::Scalar;

    const Index N = fact.N;
    const Index K = fact.K;

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
        T a = svd.matrixV().unaryExpr(at_least_zero) * jitter;
        T b = T::Ones(N, K) / static_cast<Scalar>(N);
        fact.theta_nk.update(a, b);
    }
    fact.theta_nk.calibrate();

    fact.alpha_dl.reset_stat_only();
    fact.alpha_dl.calibrate();

    fact.beta_lk.reset_stat_only();
    fact.beta_lk.calibrate();
}

template <typename MODEL, typename Derived>
void
initialize_stat(const factorization_linking_tag,
                MODEL &fact,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_SVD &do_svd,
                const typename MODEL::Scalar jitter)
{
    if (do_svd.val) {
        _initialize_stat_svd(factorization_linking_tag(), fact, Y_dn, jitter);
    } else {
        _initialize_stat_random(factorization_linking_tag(),
                                fact,
                                Y_dn,
                                jitter);
    }
}

/////////////////////
// 1. update alpha //
/////////////////////

template <typename MODEL, typename Derived>
void
add_stat_to_row(const factorization_linking_tag,
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

    fact.logRow_linking_aux_dk =
        ((Y_dn * fact.theta_nk.log_mean()).array().colwise() /
         Y_dn.rowwise().sum().unaryExpr(at_least_one).array())
            .matrix();

    fact.logRow_linking_aux_dk +=
        ((fact.alpha_dl.log_mean() * fact.beta_lk.mean()).array().rowwise() /
         (fact.beta_lk.mean().colwise().sum()).array())
            .matrix();

    fact._row_linking_factor_aux(do_stdize);

    ///////////////////////////
    // Accumulate statistics //
    ///////////////////////////

    if (do_dc) {
        fact.alpha_dl.add(((fact.row_linking_aux_dk *
                            fact.beta_lk.mean().transpose())
                               .array()
                               .colwise() *
                           Y_dn.rowwise().sum().array())
                              .matrix(),
                          Y_dn.rowwise().mean() *
                              (Y_dn.colwise().mean() * fact.theta_nk.mean() *
                               fact.beta_lk.mean().transpose()));

    } else {

        fact.alpha_dl
            .add(((fact.row_linking_aux_dk * fact.beta_lk.mean().transpose())
                      .array()
                      .colwise() *
                  Y_dn.rowwise().sum().array())
                     .matrix(),
                 ColVec::Ones(fact.D) *
                     (fact.theta_nk.mean() * fact.beta_lk.mean().transpose())
                         .colwise()
                         .sum());
    }
}

////////////////////
// 2. update beta //
////////////////////

template <typename MODEL, typename Derived>
void
add_stat_to_mid(const factorization_linking_tag,
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

    fact.logRow_aux_dk = ((Y_dn * fact.theta_nk.log_mean()).array().colwise() /
                          Y_dn.rowwise().sum().unaryExpr(at_least_one).array())
                             .matrix();

    fact.logRow_aux_dk +=
        ((fact.alpha_dl.mean() * fact.beta_lk.log_mean()).array().colwise() /
         (fact.alpha_dl.mean().rowwise().sum()).array())
            .matrix();

    fact._row_factor_aux(do_stdize);

    ///////////////////////////
    // Accumulate statistics //
    ///////////////////////////

    if (do_dc) {

        fact.beta_lk.add((fact.alpha_dl.mean().array().colwise() *
                          Y_dn.rowwise().sum().array())
                                 .matrix()
                                 .transpose() *
                             fact.row_aux_dk,
                         (fact.alpha_dl.mean().transpose() *
                          (Y_dn.rowwise().mean())) *
                             (Y_dn.colwise().mean() * fact.theta_nk.mean()));

    } else {

        fact.beta_lk.add((fact.alpha_dl.mean().array().colwise() *
                          Y_dn.rowwise().sum().array())
                                 .matrix()
                                 .transpose() *
                             fact.row_aux_dk,
                         (fact.alpha_dl.mean().colwise().sum().transpose()) *
                             (fact.theta_nk.mean().colwise().sum()));
    }
}

template <typename MODEL, typename Derived>
void
add_stat_to_col(const factorization_linking_tag,
                MODEL &fact,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_AUX_STD &std_,
                const DO_DEGREE_CORRECTION &dc_)
{
    const bool do_stdize = std_.val;

    using T = typename MODEL::Type;
    using ColVec = typename Eigen::internal::plain_col_type<T>::type;

    at_least_one_op<T> at_least_one;

    //////////////////////////////////////////////
    // Estimation of auxiliary variables (j,k)  //
    //////////////////////////////////////////////

    fact.logCol_aux_nk =
        Y_dn.transpose() * fact.alpha_dl.log_mean() * fact.beta_lk.mean();

    fact.logCol_aux_nk.array().colwise() /=
        Y_dn.colwise().sum().transpose().unaryExpr(at_least_one).array();

    fact.logCol_aux_nk.array().rowwise() /=
        fact.beta_lk.mean().colwise().sum().array();

    fact.logCol_aux_nk += fact.theta_nk.log_mean();

    // normalize the auxiliary variables
    fact._col_factor_aux(do_stdize);

    ///////////////////////////
    // Accumulate statistics //
    ///////////////////////////

    if (dc_.val) {
        fact.theta_nk.add((fact.col_aux_nk.array().colwise() *
                           Y_dn.transpose().rowwise().sum().array())
                              .matrix(),
                          Y_dn.transpose().rowwise().mean() *
                              (Y_dn.rowwise().mean().transpose() *
                               fact.alpha_dl.mean() * fact.beta_lk.mean()));

    } else {

        fact.theta_nk.add((fact.col_aux_nk.array().colwise() *
                           Y_dn.transpose().rowwise().sum().array())
                              .matrix(),
                          ColVec::Ones(fact.N) *
                              (fact.alpha_dl.mean() * fact.beta_lk.mean())
                                  .colwise()
                                  .sum());
    }
}

#endif
