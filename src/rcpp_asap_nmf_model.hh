#include "mmutil.hh"
#include "gamma_parameter.hh"

#ifndef ASAP_NMF_MODEL_HH_
#define ASAP_NMF_MODEL_HH_

template <typename RNG>
struct asap_nmf_model_t {

    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using gamma_t = gamma_param_t<Mat, RNG>;

    struct ROW : check_positive_t<Index> {
        explicit ROW(const Index r)
            : check_positive_t<Index>(r)
        {
        }
    };

    struct COL : check_positive_t<Index> {
        explicit COL(const Index c)
            : check_positive_t<Index>(c)
        {
        }
    };

    struct RSEED : check_positive_t<std::size_t> {
        explicit RSEED(const std::size_t c)
            : check_positive_t<std::size_t>(c)
        {
        }
    };

    struct FACT : check_positive_t<Index> {
        explicit FACT(const Index k)
            : check_positive_t<Index>(k)
        {
        }
    };

    struct A0 : check_positive_t<Scalar> {
        explicit A0(const Scalar v)
            : check_positive_t<Scalar>(v)
        {
        }
    };

    struct B0 : check_positive_t<Scalar> {
        explicit B0(const Scalar v)
            : check_positive_t<Scalar>(v)
        {
        }
    };

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

    struct NULL_DATA {
    };

    explicit asap_nmf_model_t(const ROW &row,
                              const COL &col,
                              const FACT &fact,
                              const RSEED &rseed,
                              const A0 &a0_,
                              const B0 &b0_)
        : D(row.val)
        , N(col.val)
        , K(std::min(fact.val, N))
        , rng(rseed.val)
        , a0(a0_.val)
        , b0(b0_.val)
        , tempK(K)
        , sampler(rng, K)
        , beta_dk(D, K, a0, b0, rng)
        , theta_nk(N, K, a0, b0, rng)
        , logRow_aux_dk(D, K)
        , row_aux_dk(D, K)
        , logCol_aux_nk(N, K)
        , col_aux_nk(N, K)
        , logNet_row_aux_dk(D, K)
        , net_row_aux_dk(D, K)
        , logNet_col_aux_nk(N, K)
        , net_col_aux_nk(N, K)
        , std_log_row_aux_dk(logRow_aux_dk, 1, 1)
        , std_log_col_aux_nk(logCol_aux_nk, 1, 1)
        , std_log_net_row_aux_dk(logNet_row_aux_dk, 1, 1)
        , std_log_net_col_aux_nk(logNet_col_aux_nk, 1, 1)
    {
        ones_n = ColVec::Ones(N);
        ones_d = ColVec::Ones(D);
        has_row_net = false;
        has_col_net = false;
    }

public:
    const Index D, N, K;
    RNG rng;
    const Scalar a0, b0;

private:
    RowVec tempK;

    template <typename Derived>
    void _normalize_aux_cols(stdizer_t<Derived> &std_log_aux,
                             Eigen::MatrixBase<Derived> &log_aux,
                             Eigen::MatrixBase<Derived> &aux,
                             const bool stoch,
                             const bool do_stdize)
    {
        ///////////////////////////////////
        // this helps spread the columns //
        ///////////////////////////////////

        if (do_stdize) {
            std_log_aux.colwise();
        }

        for (Index ii = 0; ii < aux.rows(); ++ii) {
            tempK = log_aux.row(ii);
            log_aux.row(ii) = softmax.log_row(tempK);
        }

        //////////////////////////////////////
        // stochastic sampling or normalize //
        //////////////////////////////////////

        if (stoch) {
            aux.setZero();
            for (Index ii = 0; ii < aux.rows(); ++ii) {
                Index kk = sampler(log_aux.unaryExpr(exp));
                aux(ii, kk) = 1;
            }
        } else {
            aux = log_aux.unaryExpr(exp);
        }
    }

    void _row_factor_aux(const bool stoch, const bool do_stdize)
    {
        _normalize_aux_cols(std_log_row_aux_dk,
                            logRow_aux_dk,
                            row_aux_dk,
                            stoch,
                            do_stdize);
    }

    void _net_row_factor_aux(const bool stoch, const bool do_stdize)
    {
        _normalize_aux_cols(std_log_net_row_aux_dk,
                            logNet_row_aux_dk,
                            net_row_aux_dk,
                            stoch,
                            do_stdize);
    }

    void _col_factor_aux(const bool stoch, const bool do_stdize)
    {
        _normalize_aux_cols(std_log_col_aux_nk,
                            logCol_aux_nk,
                            col_aux_nk,
                            stoch,
                            do_stdize);
    }

    void _net_col_factor_aux(const bool stoch, const bool do_stdize)
    {
        _normalize_aux_cols(std_log_net_col_aux_nk,
                            logNet_col_aux_nk,
                            net_col_aux_nk,
                            stoch,
                            do_stdize);
    }

public:
    template <typename Derived>
    void update_by_row(const Eigen::MatrixBase<Derived> &Y_dn,
                       const STOCH &stoch_,
                       const STD &std_)
    {

        const bool stoch = stoch_.val;
        const bool do_stdize = std_.val;

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (i,k)  //
        //////////////////////////////////////////////

        logRow_aux_dk = Y_dn * theta_nk.log_mean();
        logRow_aux_dk.array().colwise() /= Y_d1.array();
        logRow_aux_dk += beta_dk.log_mean();

        _row_factor_aux(stoch, do_stdize);

        // Additionally update column topic factors, theta(j, k)
        // based on this new row_aux and the previous col_aux
        if (!has_col_net) {
            theta_nk.update(col_aux_nk.cwiseProduct(Y_dn.transpose() *
                                                    row_aux_dk),
                            ones_n * beta_dk.mean().colwise().sum());
            theta_nk.calibrate();
        }

        // Update row topic factors based on this new row_aux
        beta_dk.update((row_aux_dk.array().colwise() * Y_d.array()).matrix(),
                       ones_d * theta_nk.mean().colwise().sum());
        beta_dk.calibrate();
    }

    template <typename Derived>
    void update_by_row(const Eigen::MatrixBase<Derived> &Y_dn,
                       const NULL_DATA &_null_counterfactual,
                       const NULL_DATA &_null_network,
                       const STOCH &stoch_,
                       const STD &std_)
    {
        update_by_row(Y_dn, stoch_, std_);
    }

    template <typename Derived, typename Derived2>
    void update_by_row(const Eigen::MatrixBase<Derived> &Y_dn,
                       const Eigen::MatrixBase<Derived> &Y0_dn,
                       const NULL_DATA &_null_network,
                       const STOCH &stoch_,
                       const STD &std_)
    {
        WLOG("Not implemented; will not use Y0");
        update_by_row(Y_dn, stoch_, std_);
    }

    template <typename Derived, typename Derived2>
    void update_by_row(const Eigen::MatrixBase<Derived> &Y_dn,
                       const NULL_DATA &_null_counterfactual,
                       const Eigen::SparseMatrixBase<Derived2> &A_dd,
                       const STOCH &stoch_,
                       const STD &std_)
    {

        const bool stoch = stoch_.val;
        const bool do_stdize = std_.val;

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (i,k)  //
        //////////////////////////////////////////////

        logNet_row_aux_dk = A_dd * beta_dk.log_mean();
        logNet_row_aux_dk.array().colwise() /= A_d1.array();
        logNet_row_aux_dk += beta_dk.log_mean();

        _net_row_factor_aux(stoch, true);

        logRow_aux_dk = Y_dn * theta_nk.log_mean() + A_dd * beta_dk.log_mean();
        logRow_aux_dk.array().colwise() /= (Y_d1.array() + A_d.array());
        logRow_aux_dk += beta_dk.log_mean();

        _row_factor_aux(stoch, do_stdize);

        // Additionally update column topic factors, theta(j, k)
        // based on this new row_aux and the previous col_aux
        if (!has_col_net) {
            theta_nk.update(col_aux_nk.cwiseProduct(Y_dn.transpose() *
                                                    row_aux_dk),
                            ones_n * beta_dk.mean().colwise().sum());
            theta_nk.calibrate();
        }

        // Update row topic factors based on this new row_aux
        beta_dk.update((row_aux_dk.array().colwise() * Y_d.array() +
                        net_row_aux_dk.array().colwise() * A_d.array())
                           .matrix(),
                       ones_d * theta_nk.mean().colwise().sum() +
                           (ones_d * beta_dk.mean().colwise().sum() -
                            beta_dk.mean())
                               .unaryExpr(at_least_zero));
        beta_dk.calibrate();
    }

public:
    template <typename Derived>
    void update_by_col(const Eigen::MatrixBase<Derived> &Y_dn,
                       const STOCH &stoch_,
                       const STD &std_)
    {

        const bool stoch = stoch_.val;
        const bool do_stdize = std_.val;

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (j,k)  //
        //////////////////////////////////////////////

        logCol_aux_nk = Y_dn.transpose() * beta_dk.log_mean();
        logCol_aux_nk.array().colwise() /= Y_n1.array();
        logCol_aux_nk += theta_nk.log_mean();

        _col_factor_aux(stoch, do_stdize);

        ///////////////////////
        // update parameters //
        ///////////////////////

        // Update row topic factors
        if (!has_row_net) {
            beta_dk.update(row_aux_dk.cwiseProduct(Y_dn * col_aux_nk), //
                           ones_d * theta_nk.mean().colwise().sum());  //
            beta_dk.calibrate();
        }

        // Update column topic factors
        theta_nk
            .update((col_aux_nk.array().colwise() * Y_n.array()).matrix(), //
                    ones_n * beta_dk.mean().colwise().sum());              //
        theta_nk.calibrate();
    }

    template <typename Derived>
    void update_by_col(const Eigen::MatrixBase<Derived> &Y_dn,
                       const NULL_DATA &_null_counterfactual,
                       const NULL_DATA &_null_network,
                       const STOCH &stoch_,
                       const STD &std_)
    {
        update_by_col(Y_dn, stoch_, std_);
    }

    template <typename Derived, typename Derived2>
    void update_by_col(const Eigen::MatrixBase<Derived> &Y_dn,
                       const Eigen::MatrixBase<Derived> &Y0_dn,
                       const NULL_DATA &_null_network,
                       const STOCH &stoch_,
                       const STD &std_)
    {
        WLOG("Not implemented; will not use Y0");
        update_by_col(Y_dn, stoch_, std_);
    }

    template <typename Derived, typename Derived2>
    void update_by_col(const Eigen::MatrixBase<Derived> &Y_dn,
                       const NULL_DATA &_null_counterfactual,
                       const Eigen::SparseMatrixBase<Derived2> &A_nn,
                       const STOCH &stoch_,
                       const STD &std_)
    {

        const bool stoch = stoch_.val;
        const bool do_stdize = std_.val;

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (j,k)  //
        //////////////////////////////////////////////

        logNet_col_aux_nk = A_nn * theta_nk.log_mean();
        logNet_col_aux_nk.array().colwise() /= A_n1.array();
        logNet_col_aux_nk += theta_nk.log_mean();

        _net_col_factor_aux(stoch, true);

        logCol_aux_nk =
            Y_dn.transpose() * beta_dk.log_mean() + A_nn * theta_nk.log_mean();
        logCol_aux_nk.array().colwise() /= (Y_n1.array() + A_n.array());
        logCol_aux_nk += theta_nk.log_mean();

        _col_factor_aux(stoch, do_stdize);

        ///////////////////////
        // update parameters //
        ///////////////////////

        // Update row topic factors
        if (!has_row_net) {
            beta_dk.update(row_aux_dk.cwiseProduct(Y_dn * col_aux_nk),
                           ones_d * theta_nk.mean().colwise().sum());
            beta_dk.calibrate();
        }

        // Update column topic factors
        theta_nk.update((col_aux_nk.array().colwise() * Y_n.array() +
                         net_col_aux_nk.array() * A_n.array())
                            .matrix(),
                        ones_n * beta_dk.mean().colwise().sum() +
                            (ones_n * theta_nk.mean().colwise().sum() -
                             theta_nk.mean())

        );
        theta_nk.calibrate();
    }

public:
    template <typename Derived>
    void initialize_by_svd(const Eigen::MatrixBase<Derived> &Y_dn,
                           const Scalar lb = -8,
                           const Scalar ub = 8)
    {
        clamp_op<Mat> clamp_(lb, ub);
        Mat yy = standardize_columns(Y_dn.unaryExpr(log1p));
        yy = yy.unaryExpr(clamp_).eval();
        // const std::size_t lu_iter = 5;      // this should be good
        // RandomizedSVD<Mat> svd(K, lu_iter); //
        // svd.compute(yy);
        Eigen::BDCSVD<Mat> svd;
        svd.compute(yy, Eigen::ComputeThinU | Eigen::ComputeThinV);

        {
            Mat a = svd.matrixU().unaryExpr(at_least_zero);
            Mat b = Mat::Ones(D, K) / static_cast<Scalar>(D);
            beta_dk.update(a, b);
        }
        {
            Mat a = svd.matrixV().unaryExpr(at_least_zero);
            Mat b = Mat::Ones(N, K) / static_cast<Scalar>(N);
            theta_nk.update(a, b);
        }

        beta_dk.calibrate();
        theta_nk.calibrate();
        randomize_auxiliaries();
    }

    void initialize_random()
    {
        {
            Mat a = beta_dk.sample();
            Mat b = Mat::Ones(D, K);
            beta_dk.update(a / static_cast<Scalar>(D), b);
        }
        {
            Mat a = theta_nk.sample();
            Mat b = Mat::Ones(N, K);
            theta_nk.update(a / static_cast<Scalar>(N), b);
        }
        beta_dk.calibrate();
        theta_nk.calibrate();
        randomize_auxiliaries();
    }

public:
    template <typename Derived>
    void precompute(const Eigen::MatrixBase<Derived> &Y_dn,
                    const NULL_DATA &,
                    const NULL_DATA &)
    {
        precompute_Y(Y_dn);
    }

    template <typename Derived, typename Derived2>
    void precompute(const Eigen::MatrixBase<Derived> &Y_dn,
                    const NULL_DATA &,
                    const Eigen::SparseMatrixBase<Derived2> &A_nn)
    {
        precompute_Y(Y_dn);
        precompute_A_col(A_nn);
    }

    template <typename Derived, typename Derived2>
    void precompute(const Eigen::MatrixBase<Derived> &Y_dn,
                    const Eigen::SparseMatrixBase<Derived2> &A_dd,
                    const NULL_DATA &)
    {
        precompute_Y(Y_dn);
        precompute_A_row(A_dd);
    }

    template <typename Derived, typename Derived2, typename Derived3>
    void precompute(const Eigen::MatrixBase<Derived> &Y_dn,
                    const Eigen::SparseMatrixBase<Derived2> &A_dd,
                    const Eigen::SparseMatrixBase<Derived3> &A_nn)
    {
        precompute_Y(Y_dn);
        precompute_A_row(A_dd);
        precompute_A_col(A_nn);
    }

private:
    template <typename Derived>
    void precompute_Y(const Eigen::MatrixBase<Derived> &Y_dn)
    {
        Y_n = Y_dn.colwise().sum().transpose();
        Y_d = Y_dn.rowwise().sum();
        Y_n1 = Y_n.unaryExpr(at_least_one);
        Y_d1 = Y_d.unaryExpr(at_least_one);
    }

    template <typename Derived>
    void precompute_A_row(const Eigen::SparseMatrixBase<Derived> &A_dd)
    {
        A_d = A_dd.transpose() * ColVec::Ones(A_dd.rows());
        A_d1 = A_d.unaryExpr(at_least_one);
        has_row_net = true;
    }

    template <typename Derived>
    void precompute_A_col(const Eigen::SparseMatrixBase<Derived> &A_nn)
    {
        A_n = A_nn.transpose() * ColVec::Ones(A_nn.rows());
        A_n1 = A_n.unaryExpr(at_least_one);
        has_col_net = true;
    }

public:
    template <typename Derived>
    Scalar log_likelihood(const Eigen::MatrixBase<Derived> &Y_dn)
    {
        Scalar llik = 0;
        const Scalar denom = N * D;

        llik += (row_aux_dk.cwiseProduct(beta_dk.log_mean()).transpose() * Y_dn)
                    .sum() /
            denom;

        llik +=
            (Y_dn * col_aux_nk.cwiseProduct(theta_nk.log_mean())).sum() / denom;

        llik -=
            ((ones_d.transpose() * row_aux_dk.cwiseProduct(beta_dk.mean())) *
             (col_aux_nk.cwiseProduct(theta_nk.mean()).transpose() * ones_n))
                .sum() /
            denom;

        return llik;
    }

    template <typename Derived>
    std::tuple<Mat, Mat>
    log_topic_correlation(const Eigen::MatrixBase<Derived> &Y_dn)
    {
        Mat log_x = beta_dk.log_mean();
        standardize_columns_inplace(log_x);
        Mat R_nk = (Y_dn.transpose() * log_x).array().colwise() / Y_n1.array();
        return std::make_tuple(log_x, R_nk);
    }

private:
    void randomize_auxiliaries()
    {
        logRow_aux_dk = Mat::Random(D, K);
        for (Index ii = 0; ii < D; ++ii) {
            row_aux_dk.row(ii) = softmax.apply_row(logRow_aux_dk.row(ii));
        }

        logCol_aux_nk = Mat::Random(N, K);
        for (Index jj = 0; jj < N; ++jj) {
            col_aux_nk.row(jj) = softmax.apply_row(logCol_aux_nk.row(jj));
        }
    }

private:
    rowvec_sampler_t<Mat, RNG> sampler;

public:
    gamma_t beta_dk;  // dictionary
    gamma_t theta_nk; // scaling for all the factor loading

private:
    Mat logRow_aux_dk, row_aux_dk;         // row to topic latent assignment
    Mat logCol_aux_nk, col_aux_nk;         // column to topic latent assignment
    Mat logNet_row_aux_dk, net_row_aux_dk; // based on row's network
    Mat logNet_col_aux_nk, net_col_aux_nk; // based on col's network

    stdizer_t<Mat> std_log_row_aux_dk;
    stdizer_t<Mat> std_log_col_aux_nk;
    stdizer_t<Mat> std_log_net_row_aux_dk;
    stdizer_t<Mat> std_log_net_col_aux_nk;

    ColVec ones_n;
    ColVec ones_d;
    ColVec Y_n;
    ColVec Y_d;
    ColVec Y_n1;
    ColVec Y_d1;
    ColVec A_d;
    ColVec A_d1;
    ColVec A_n;
    ColVec A_n1;

private: // functors
    exp_op<Mat> exp;
    log1p_op<Mat> log1p;
    at_least_one_op<Mat> at_least_one;
    at_least_zero_op<Mat> at_least_zero;
    softmax_op_t<Mat> softmax;

    bool has_row_net;
    bool has_col_net;
};

//////////////////////
// helper functions //
//////////////////////

template <typename RNG>
Rcpp::List
rcpp_list_out(const asap_nmf_model_t<RNG> &model)
{
    return Rcpp::List::create(Rcpp::_["beta"] = model.beta_dk.mean(),
                              Rcpp::_["log.beta"] = model.beta_dk.log_mean(),
                              Rcpp::_["log.beta.sd"] = model.beta_dk.log_sd(),
                              Rcpp::_["theta"] = model.theta_nk.mean(),
                              Rcpp::_["log.theta"] = model.theta_nk.log_mean(),
                              Rcpp::_["log.theta.sd"] =
                                  model.theta_nk.log_sd());
}

#endif
