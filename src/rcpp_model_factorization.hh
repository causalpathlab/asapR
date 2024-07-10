#ifndef RCPP_MODEL_FACTORIZATION_HH_
#define RCPP_MODEL_FACTORIZATION_HH_

struct factorization_tag { };

template <typename ROW, typename COL, typename RNG>
struct factorization_t {

    using Type = typename ROW::Type;
    using Index = typename Type::Index;
    using Scalar = typename Type::Scalar;
    using RowVec = typename Eigen::internal::plain_row_type<Type>::type;

    using RNG_TYPE = RNG;
    using tag = factorization_tag;

    explicit factorization_t(ROW &_row_dk, COL &_col_nk, const RSEED &rseed)
        : beta_dk(_row_dk)
        , theta_nk(_col_nk)
        , D(beta_dk.rows())
        , N(theta_nk.rows())
        , K(beta_dk.cols())
        , logRow_aux_dk(D, K)
        , row_aux_dk(D, K)
        , logCol_aux_nk(N, K)
        , col_aux_nk(N, K)
        , std_log_row_aux_dk(logRow_aux_dk, 1, 1)
        , std_log_col_aux_nk(logCol_aux_nk, 1, 1)
        , rng(rseed.val)
        , num_threads(1)
    {
        ASSERT(theta_nk.cols() == beta_dk.cols(),
               "row and col factors must have the same # columns");
        randomize_auxiliaries();
    }

    explicit factorization_t(ROW &_row_dk,
                             COL &_col_nk,
                             const RSEED &rseed,
                             const NThreads &nthr)
        : beta_dk(_row_dk)
        , theta_nk(_col_nk)
        , D(beta_dk.rows())
        , N(theta_nk.rows())
        , K(beta_dk.cols())
        , logRow_aux_dk(D, K)
        , row_aux_dk(D, K)
        , logCol_aux_nk(N, K)
        , col_aux_nk(N, K)
        , std_log_row_aux_dk(logRow_aux_dk, 1, 1)
        , std_log_col_aux_nk(logCol_aux_nk, 1, 1)
        , rng(rseed.val)
        , num_threads(nthr.val)
    {
        ASSERT(theta_nk.cols() == beta_dk.cols(),
               "row and col factors must have the same # columns");
        randomize_auxiliaries();
    }

    ROW &beta_dk;
    COL &theta_nk;

    const Index D;
    const Index N;
    const Index K;

    Type logRow_aux_dk;
    Type row_aux_dk;
    Type logCol_aux_nk;
    Type col_aux_nk;

    stdizer_t<Type> std_log_row_aux_dk;
    stdizer_t<Type> std_log_col_aux_nk;

    RNG rng;

    const std::size_t num_threads;

    softmax_op_t<Type> softmax;

    template <typename Derived>
    void _normalize_aux_cols(stdizer_t<Derived> &std_log_aux,
                             Eigen::MatrixBase<Derived> &log_aux,
                             Eigen::MatrixBase<Derived> &aux,
                             const bool do_stdize)
    {
        ///////////////////////////////////
        // this helps spread the columns //
        ///////////////////////////////////

        if (do_stdize) {
            std_log_aux.colwise();
        }

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
        for (Index ii = 0; ii < aux.rows(); ++ii) {
            log_aux.row(ii) = softmax.log_row(log_aux.row(ii));
        }
        aux = log_aux.unaryExpr(exp);
        aux.array().colwise() /= aux.array().rowwise().sum();
    }

    void _row_factor_aux(const bool do_stdize)
    {
        _normalize_aux_cols(std_log_row_aux_dk,
                            logRow_aux_dk,
                            row_aux_dk,
                            do_stdize);
    }

    void _col_factor_aux(const bool do_stdize)
    {
        _normalize_aux_cols(std_log_col_aux_nk,
                            logCol_aux_nk,
                            col_aux_nk,
                            do_stdize);
    }

    void randomize_auxiliaries()
    {
        logRow_aux_dk = Type::Random(D, K);
        for (Index ii = 0; ii < D; ++ii) {
            row_aux_dk.row(ii) = softmax.apply_row(logRow_aux_dk.row(ii));
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
log_likelihood(const factorization_tag,
               const MODEL &fact,
               const Eigen::MatrixBase<Derived> &Y_dn)

{
    const auto D = fact.D;
    const auto N = fact.N;
    const auto K = fact.K;
    typename MODEL::Type::Scalar llik = 0;
    typename MODEL::Type::Scalar denom = N * D;

    safe_log_op<typename MODEL::Type> safe_log(1e-8);

    llik += Y_dn.cwiseProduct(fact.beta_dk.mean() *
                              fact.theta_nk.mean().transpose())
                .unaryExpr(safe_log)
                .sum() /
        denom;

    llik -=
        (fact.beta_dk.mean() * fact.theta_nk.mean().transpose()).sum() / denom;

    return llik;
}

template <typename MODEL, typename Derived>
void initialize_stat(const factorization_tag,
                     MODEL &fact,
                     const Eigen::MatrixBase<Derived> &Y_dn,
                     const DO_SVD &do_svd,
                     const typename MODEL::Scalar jitter);

template <typename MODEL, typename Derived>
void
_initialize_stat_random(const factorization_tag,
                        MODEL &fact,
                        const Eigen::MatrixBase<Derived> &Y_dn,
                        const typename MODEL::Scalar jitter)
{
    using Mat = typename MODEL::Type;
    using Index = typename Mat::Index;
    using Scalar = typename Mat::Scalar;

    const Index D = fact.D;
    const Index N = fact.N;
    const Index K = fact.K;

    // auto &rng = fact.rng;
    // auto &exp = fact.exp;
    // using norm_dist_t = boost::random::normal_distribution<Scalar>;
    // norm_dist_t norm_dist(0., 1.);
    // auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };
    // Mat temp_dk = Mat::NullaryExpr(D, K, rnorm).unaryExpr(exp) * jitter;

    Mat temp_dk = fact.beta_dk.sample() * jitter;
    fact.beta_dk.update(temp_dk, Mat::Ones(D, K));
    fact.beta_dk.calibrate();

    // Mat temp_nk = fact.theta_nk.sample();
    // fact.theta_nk.update(temp_nk, Mat::Ones(N, K));

    fact.theta_nk.reset_stat_only();
    fact.theta_nk.calibrate();
}

template <typename MODEL, typename Derived>
void
_initialize_stat_svd(const factorization_tag,
                     MODEL &fact,
                     const Eigen::MatrixBase<Derived> &Y_dn,
                     const typename MODEL::Scalar jitter)
{
    using T = typename MODEL::Type;
    using Index = typename T::Index;
    using Scalar = typename T::Scalar;

    const Index D = fact.D;
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
        T a = svd.matrixU().unaryExpr(at_least_zero) * jitter;
        T b = T::Ones(D, K) / static_cast<Scalar>(D);
        fact.beta_dk.update(a, b);
    }
    {
        T a = svd.matrixV().unaryExpr(at_least_zero) * jitter;
        T b = T::Ones(N, K) / static_cast<Scalar>(N);
        fact.theta_nk.update(a, b);
    }

    fact.beta_dk.calibrate();
    fact.theta_nk.calibrate();
}

template <typename MODEL, typename Derived>
void
initialize_stat(const factorization_tag,
                MODEL &fact,
                const Eigen::MatrixBase<Derived> &Y_dn,
                const DO_SVD &do_svd,
                const typename MODEL::Scalar jitter)
{
    if (do_svd.val) {
        _initialize_stat_svd(factorization_tag(), fact, Y_dn, jitter);
    } else {
        _initialize_stat_random(factorization_tag(), fact, Y_dn, jitter);
    }
}

template <typename MODEL, typename Derived>
void
add_stat_to_col(const factorization_tag,
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

    fact.logCol_aux_nk = Y_dn.transpose() * fact.beta_dk.log_mean();
    fact.logCol_aux_nk.array().colwise() /=
        Y_dn.colwise().sum().transpose().unaryExpr(at_least_one).array();
    fact.logCol_aux_nk += fact.theta_nk.log_mean();

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
                               fact.beta_dk.mean()));
    } else {
        fact.theta_nk.add((fact.col_aux_nk.array().colwise() *
                           Y_dn.transpose().rowwise().sum().array())
                              .matrix(),
                          ColVec::Ones(fact.N) *
                              fact.beta_dk.mean().colwise().sum());
    }
}

template <typename MODEL, typename Derived>
void
add_stat_to_row(const factorization_tag,
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
    // Estimation of auxiliary variables (i,k)  //
    //////////////////////////////////////////////

    fact.logRow_aux_dk = Y_dn * fact.theta_nk.log_mean();
    fact.logRow_aux_dk.array().colwise() /=
        Y_dn.rowwise().sum().unaryExpr(at_least_one).array();
    fact.logRow_aux_dk += fact.beta_dk.log_mean();

    fact._row_factor_aux(do_stdize);

    ///////////////////////////
    // Accumulate statistics //
    ///////////////////////////

    if (dc_.val) {
        fact.beta_dk.add((fact.row_aux_dk.array().colwise() *
                          Y_dn.rowwise().sum().array())
                             .matrix(),
                         Y_dn.rowwise().mean() *
                             (Y_dn.colwise().mean() * fact.theta_nk.mean()));
    } else {
        fact.beta_dk.add((fact.row_aux_dk.array().colwise() *
                          Y_dn.rowwise().sum().array())
                             .matrix(),
                         ColVec::Ones(fact.D) *
                             fact.theta_nk.mean().colwise().sum());
    }
}

#endif
