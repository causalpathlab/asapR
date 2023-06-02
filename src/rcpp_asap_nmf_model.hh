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
        , logPhi_dk(D, K)
        , phi_dk(D, K)
        , logRho_nk(N, K)
        , rho_nk(N, K)
        , std_ln_rho_nk(logRho_nk, 1, 1)
    {

        ones_n = ColVec::Ones(N);
        ones_d = ColVec::Ones(D);
    }

public:
    const Index D, N, K;
    RNG rng;
    const Scalar a0, b0;

private:
    RowVec tempK;

public:
    template <typename Derived>
    void update_by_row(const Eigen::MatrixBase<Derived> &Y_dn,
                       const bool stoch = false)
    {
        //////////////////////////////////////////////
        // Estimation of auxiliary variables (i,k)  //
        //////////////////////////////////////////////

        logPhi_dk = Y_dn * theta_nk.log_mean();
        logPhi_dk.array().colwise() /= Y_d1.array();
        logPhi_dk += beta_dk.log_mean();
        for (Index ii = 0; ii < D; ++ii) {
            tempK = logPhi_dk.row(ii);
            logPhi_dk.row(ii) = softmax.log_row(tempK);
        }

        if (stoch) {
            phi_dk.setZero();
            for (Index ii = 0; ii < D; ++ii) {
                Index k = sampler(logPhi_dk.row(ii).unaryExpr(exp));
                phi_dk(ii, k) = 1.;
            }
        } else {
            phi_dk = logPhi_dk.unaryExpr(exp);
        }

        // Update column topic factors, theta(j, k)
        theta_nk.update(rho_nk.cwiseProduct(Y_dn.transpose() * phi_dk), //
                        ones_n * beta_dk.mean().colwise().sum());       //
        theta_nk.calibrate();

        // Update row topic factors
        beta_dk.update((phi_dk.array().colwise() * Y_d.array()).matrix(), //
                       ones_d * theta_nk.mean().colwise().sum());         //
        beta_dk.calibrate();
    }

    template <typename Derived>
    void update_by_col(const Eigen::MatrixBase<Derived> &Y_dn,
                       const Scalar eps = 1e-4)
    {
        //////////////////////////////////////////////
        // Estimation of auxiliary variables (j,k)  //
        //////////////////////////////////////////////

        logRho_nk = Y_dn.transpose() * beta_dk.log_mean();
        logRho_nk.array().colwise() /= Y_n1.array();
        logRho_nk += theta_nk.log_mean();

        ///////////////////////////////////
        // this helps spread the columns //
        ///////////////////////////////////

        std_ln_rho_nk.colwise(eps);

        for (Index jj = 0; jj < N; ++jj) {
            tempK = logRho_nk.row(jj);
            logRho_nk.row(jj) = softmax.log_row(tempK);
        }

        rho_nk = logRho_nk.unaryExpr(exp);

        ///////////////////////
        // update parameters //
        ///////////////////////

        // Update row topic factors
        beta_dk.update(phi_dk.cwiseProduct(Y_dn * rho_nk),        //
                       ones_d * theta_nk.mean().colwise().sum()); //
        beta_dk.calibrate();

        // Update column topic factors
        theta_nk.update((rho_nk.array().colwise() * Y_n.array()).matrix(), //
                        ones_n * beta_dk.mean().colwise().sum());          //
        theta_nk.calibrate();
    }

public:
    template <typename Derived>
    void initialize_by_svd(const Eigen::MatrixBase<Derived> &Y_dn)
    {
        const std::size_t lu_iter = 5;      // this should be good
        RandomizedSVD<Mat> svd(K, lu_iter); //
        const Mat yy = standardize_columns(Y_dn.unaryExpr(log1p));
        svd.compute(yy);
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
        randomize_auxiliaries();
    }

    template <typename Derived>
    void precompute(const Eigen::MatrixBase<Derived> &Y_dn)
    {
        Y_n = Y_dn.colwise().sum().transpose();
        Y_d = Y_dn.transpose().colwise().sum();
        Y_n1 = Y_n.unaryExpr(at_least_one);
        Y_d1 = Y_d.unaryExpr(at_least_one);
    }

    template <typename Derived>
    Scalar log_likelihood(const Eigen::MatrixBase<Derived> &Y_dn)
    {
        Scalar llik = 0;
        const Scalar denom = N * D;

        llik +=
            (phi_dk.cwiseProduct(beta_dk.log_mean()).transpose() * Y_dn).sum() /
            denom;

        llik += (phi_dk.cwiseProduct(Y_dn * theta_nk.log_mean())).sum() / denom;

        llik -=
            (phi_dk.cwiseProduct(logPhi_dk).transpose() * Y_dn).sum() / denom;

        llik -= (ones_d.transpose() * beta_dk.mean() *
                 theta_nk.mean().transpose() * ones_n)
                    .sum() /
            denom;

        return llik;
    }

    template <typename Derived>
    std::tuple<Mat, Mat>
    log_topic_correlation(const Eigen::MatrixBase<Derived> &Y_dn)
    {
        Mat log_x = standardize_columns(beta_dk.log_mean());
        Mat R_nk = (Y_dn.transpose() * log_x).array().colwise() / Y_n1.array();
        return std::make_tuple(log_x, R_nk);
    }

private:
    void randomize_auxiliaries()
    {
        logPhi_dk = Mat::Random(D, K);
        for (Index ii = 0; ii < D; ++ii) {
            phi_dk.row(ii) = softmax.apply_row(logPhi_dk.row(ii));
        }

        logRho_nk = Mat::Random(N, K);
        for (Index jj = 0; jj < N; ++jj) {
            rho_nk.row(jj) = softmax.apply_row(logRho_nk.row(jj));
        }
    }

private:
    rowvec_sampler_t<Mat, RNG> sampler;

public:
    gamma_t beta_dk;       // dictionary
    gamma_t theta_nk;      // scaling for all the factor loading
    Mat logPhi_dk, phi_dk; // row to topic latent assignment
    Mat logRho_nk, rho_nk; // column to topic latent assignment

private:
    stdizer_t<Mat> std_ln_rho_nk;

    ColVec ones_n;
    ColVec ones_d;
    ColVec Y_n;
    ColVec Y_d;
    ColVec Y_n1;
    ColVec Y_d1;

private: // functors
    exp_op<Mat> exp;
    log1p_op<Mat> log1p;
    at_least_one_op<Mat> at_least_one;
    softmax_op_t<Mat> softmax;
};

#endif
