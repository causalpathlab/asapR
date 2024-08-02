#include "rcpp_asap.hh"
#include "rcpp_asap_util.hh"
#include "gamma_parameter.hh"
#include "rcpp_asap_batch.hh"
#include "rcpp_mtx_data.hh"
#include "rcpp_eigenSparse_data.hh"

#ifndef RCPP_ASAP_REGRESSION_HH_
#define RCPP_ASAP_REGRESSION_HH_

namespace asap { namespace regression {

struct stat_options_t {
    stat_options_t()
        : do_log1p(false)
        , verbose(true)
        , NUM_THREADS(0)
        , BLOCK_SIZE(1000)
        , MAX_ROW_WORD(2)
        , ROW_WORD_SEP('_')
        , MAX_COL_WORD(100)
        , COL_WORD_SEP('@')
        , do_stdize_x(true)
        , do_stdize_r(true)
        , a0(1.)
        , b0(1.)
        , CELL_NORM(0)
        , max_iter(10)
    {
    }

    bool do_log1p;
    bool verbose;
    std::size_t NUM_THREADS;
    std::size_t BLOCK_SIZE;
    std::size_t MAX_ROW_WORD;
    char ROW_WORD_SEP;
    std::size_t MAX_COL_WORD;
    char COL_WORD_SEP;
    bool do_stdize_x;
    bool do_stdize_r;
    double a0;
    double b0;
    double CELL_NORM;
    std::size_t max_iter;
};

template <typename Data, typename Derived, typename Derived2>
Rcpp::List run_pmf_regression(Data &data,
                              const Eigen::MatrixBase<Derived> &log_beta,
                              const Eigen::MatrixBase<Derived2> &log_delta,
                              const std::vector<std::string> &row_names,
                              const std::vector<std::string> &col_names,
                              const stat_options_t &options);

template <typename Data,
          typename Derived1,
          typename Derived2,
          typename Derived3>
int run_pmf_stat(Data &data,
                 const Eigen::MatrixBase<Derived1> &_log_x,
                 const std::vector<std::string> &pos2row,
                 const stat_options_t &options,
                 Eigen::MatrixBase<Derived2> &_r_nk,
                 Eigen::MatrixBase<Derived3> &_y_n);

template <typename Derived1, typename Derived2, typename Derived3>
int
pmf_stat_mtx(const std::string mtx_file,
             const std::string row_file,
             const std::string col_file,
             const std::string idx_file,
             const Eigen::MatrixBase<Derived1> &_log_x,
             const std::vector<std::string> &pos2row,
             const stat_options_t &options,
             Eigen::MatrixBase<Derived2> &_r_nk,
             Eigen::MatrixBase<Derived3> &_y_n)
{
    mtx_data_t data(mtx_tuple_t(mtx_tuple_t::MTX(mtx_file),
                                mtx_tuple_t::ROW(row_file),
                                mtx_tuple_t::COL(col_file),
                                mtx_tuple_t::IDX(idx_file)),
                    options.MAX_ROW_WORD,
                    options.ROW_WORD_SEP);
    return run_pmf_stat(data, _log_x, pos2row, options, _r_nk, _y_n);
}

template <typename Derived1, typename Derived2, typename Derived3>
int
pmf_stat(const Eigen::SparseMatrix<float> &y_dn,
         const Eigen::MatrixBase<Derived1> &_log_x,
         const std::vector<std::string> &pos2row,
         const stat_options_t &options,
         Eigen::MatrixBase<Derived2> &_r_nk,
         Eigen::MatrixBase<Derived3> &_y_n)
{
    eigenSparse_data_t data(y_dn, pos2row);
    return run_pmf_stat(data, _log_x, pos2row, options, _r_nk, _y_n);
}

////////////////////////////
// implementation details //
////////////////////////////

template <typename IN1,
          typename IN2,
          typename IN3,
          typename RET1,
          typename RET2>
int
run_pmf_theta(const Eigen::MatrixBase<IN1> &log_beta_dk, // log dictionary
              const Eigen::MatrixBase<IN2> &r_nk,        // correlation
              const Eigen::MatrixBase<IN3> &ysum_n,      // y sum
              Eigen::MatrixBase<RET1> &_theta,           // theta
              Eigen::MatrixBase<RET2> &_log_theta,       // log theta
              const double a0 = 1.0,
              const double b0 = 1.0,
              const std::size_t max_iter = 10,
              const bool do_stdize_col = true,
              const bool verbose = true)
{
    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Mat, RNG>;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    RNG rng;

    exp_op<Mat> exp;
    softmax_op_t<Mat> softmax;

    const Mat beta_dk = log_beta_dk.unaryExpr(exp);
    const std::size_t N = r_nk.rows();
    const std::size_t D = beta_dk.rows();
    const std::size_t K = beta_dk.cols();

    ASSERT_RET(ysum_n.rows() == N,
               "check ysum_n: " << ysum_n.rows() << " x " << ysum_n.cols()
                                << " vs. " << N);

    const ColVec degree_n = ysum_n / static_cast<Scalar>(D);

    gamma_t param_theta_nk(N, K, a0, b0, rng); // n x K
    Mat logRho_nk(N, K), rho_nk(N, K);         // n x K

    RET1 &theta = _theta.derived();
    RET2 &log_theta = _log_theta.derived();

    theta = Mat::Zero(N, K);
    log_theta = Mat::Zero(N, K);

    const Mat x_nk = degree_n * beta_dk.colwise().sum(); // N x K

    const Scalar qq_min = .01, qq_max = .99;

    for (std::size_t t = 0; t < max_iter; ++t) {
        logRho_nk = r_nk + param_theta_nk.log_mean();

        if (do_stdize_col) {
            asap::util::stretch_matrix_columns_inplace(logRho_nk,
                                                       qq_min,
                                                       qq_max,
                                                       verbose);
        }

        for (Index jj = 0; jj < N; ++jj) {
            logRho_nk.row(jj) = softmax.log_row(logRho_nk.row(jj));
        }
        rho_nk = logRho_nk.unaryExpr(exp);

        param_theta_nk.update(ysum_n.asDiagonal() * rho_nk, x_nk);
        param_theta_nk.calibrate();
    }

    theta += param_theta_nk.mean();
    log_theta += param_theta_nk.log_mean();

    return EXIT_SUCCESS;
}

template <typename Data,
          typename Derived1,
          typename Derived2,
          typename Derived3>
int
run_pmf_stat(Data &data,
             const Eigen::MatrixBase<Derived1> &_log_x,
             const std::vector<std::string> &pos2row,
             const stat_options_t &options,
             Eigen::MatrixBase<Derived2> &_r_nk,
             Eigen::MatrixBase<Derived3> &_y_n)
{

    const bool do_log1p = options.do_log1p;
    const bool verbose = options.verbose;
    const std::size_t NUM_THREADS =
        (options.NUM_THREADS > 0 ? options.NUM_THREADS : omp_get_max_threads());
    const std::size_t BLOCK_SIZE = options.BLOCK_SIZE;

    TLOG_(verbose, NUM_THREADS << " threads");

    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    Mat logX_dk = _log_x.derived();

    const Scalar cell_norm = options.CELL_NORM;

    if (options.do_stdize_x) {
        asap::util::stretch_matrix_columns_inplace(logX_dk);
    }

    Mat &Rtot_nk = _r_nk.derived();
    Mat &Ytot_n = _y_n.derived();

    //////////////////////////////////////
    // take care of different row names //
    //////////////////////////////////////

    const Index D = pos2row.size(); // dimensionality

    ASSERT_RET(D > 0, "No features found!");

    ASSERT_RET(logX_dk.rows() == D,
               "#Rows in the logX_dk matrix !=  the size of x_row_names: "
                   << logX_dk.rows() << " != " << D);

    std::unordered_map<std::string, Index> row2pos;
    for (Index r = 0; r < pos2row.size(); ++r) {
        row2pos[pos2row.at(r)] = r;
    }

    ASSERT_RET(row2pos.size() == D, "Redundant row names exist");
    TLOG_(verbose, "Found " << row2pos.size() << " unique row names");

    const Index N = data.max_col();
    const Index K = logX_dk.cols();      // number of topics
    const Index block_size = BLOCK_SIZE; // memory block size

    TLOG_(verbose, "lnX: " << logX_dk.rows() << " x " << logX_dk.cols());
    Rtot_nk.resize(N, K);
    Rtot_nk.setZero();
    Ytot_n.resize(N, 1);
    Ytot_n.setZero();
    Index Nprocessed = 0;

    data.relocate_rows(row2pos);

    if (verbose) {
        Rcpp::Rcerr << "Calibrating " << N << " samples..." << std::endl;
    }

    at_least_one_op<Mat> at_least_one;
    at_least_zero_op<Mat> at_least_zero;
    exp_op<Mat> exp;
    log1p_op<Mat> log1p;
    softmax_op_t<Mat> softmax;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {
        const Index ub = std::min(N, block_size + lb);
        Mat y = data.read_reloc(lb, ub);

        if (cell_norm >= 1.) {
            normalize_columns_inplace(y);
            y *= cell_norm;
        }

        ///////////////////////////////////////
        // do log1p transformation if needed //
        ///////////////////////////////////////

        const Mat y_dn = (do_log1p ? y.unaryExpr(log1p) : y);

        const Index n = y_dn.cols();

        ColVec Y_n = y_dn.colwise().sum().transpose(); // n x 1
        ColVec Y_n1 = Y_n.unaryExpr(at_least_one);     // n x 1

        ///////////////////////////
        // parameter of interest //
        ///////////////////////////

        Mat R_nk =
            (y_dn.transpose() * logX_dk).array().colwise() / Y_n1.array();

#pragma omp critical
        {
            for (Index i = 0; i < (ub - lb); ++i) {
                const Index j = i + lb;
                Rtot_nk.row(j) = R_nk.row(i);
                Ytot_n(j, 0) = Y_n(i);
            }

            Nprocessed += n;
            if (verbose) {
                Rcpp::Rcerr << "\rProcessed: " << Nprocessed << std::flush;
            } else {
                Rcpp::Rcerr << "+ " << std::flush;
                if (Nprocessed % 1000 == 0)
                    Rcpp::Rcerr << "\r" << std::flush;
            }
        } // end of omp critical
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index j = 0; j < N; ++j) {
        Rtot_nk.row(j) = softmax.log_row(Rtot_nk.row(j));
    }

    if (!verbose)
        Rcpp::Rcerr << std::endl;
    TLOG_(verbose, "done -> topic stat");

    return EXIT_SUCCESS;
}

template <typename Data, typename Derived, typename Derived2>
Rcpp::List
run_pmf_regression(Data &data,
                   const Eigen::MatrixBase<Derived> &log_beta,
                   const Eigen::MatrixBase<Derived2> &log_delta,
                   const std::vector<std::string> &row_names,
                   const std::vector<std::string> &col_names,
                   const stat_options_t &options)
{

    const bool do_log1p = options.do_log1p;
    const bool verbose = options.verbose;
    const std::size_t NUM_THREADS =
        (options.NUM_THREADS > 0 ? options.NUM_THREADS : omp_get_max_threads());
    const std::size_t BLOCK_SIZE = options.BLOCK_SIZE;
    const bool do_stdize_col = options.do_stdize_r;
    const Scalar a0 = options.a0, b0 = options.b0;
    const std::size_t max_iter = options.max_iter;

    exp_op<Mat> exp;

    const Index D = log_beta.rows();

    Mat Rtot_nk, Ytot_n, delta_db;

    CHK_RETL_(run_pmf_stat(data, log_beta, row_names, options, Rtot_nk, Ytot_n),
              "failed to compute topic pmf stat");

    const std::size_t N = Rtot_nk.rows(), K = Rtot_nk.cols();
    Mat theta_nk(N, K), log_theta_nk(N, K);
    theta_nk.setZero();
    log_theta_nk.setZero();

    if (log_delta.rows() == D) {

        const Index B = log_delta.cols();

        TLOG_(verbose, "Correlations with" << B << " delta factors");

        Mat R0tot_nb, Y0tot_n;
        CHK_RETL_(run_pmf_stat(data,
                               log_delta,
                               row_names,
                               options,
                               R0tot_nb,
                               Y0tot_n),
                  "failed to compute null stat");

        // 2. Take residuals
        TLOG_(verbose, "Regress out the batch effect correlations");
        residual_columns_inplace(Rtot_nk, R0tot_nb);

        if (do_stdize_col) {
            asap::util::stretch_matrix_columns_inplace(Rtot_nk);
        }

        // 3. Estimate theta based on the new R
        CHK_RETL_(run_pmf_theta(log_beta,
                                Rtot_nk,
                                Ytot_n,
                                theta_nk,
                                log_theta_nk,
                                a0,
                                b0,
                                max_iter,
                                do_stdize_col,
                                verbose),
                  "unable to calibrate theta values");

    } else {

        TLOG_(verbose, "Calibrating the loading coefficients (theta)");

        CHK_RETL_(run_pmf_theta(log_beta,
                                Rtot_nk,
                                Ytot_n,
                                theta_nk,
                                log_theta_nk,
                                a0,
                                b0,
                                max_iter,
                                do_stdize_col,
                                verbose),
                  "unable to calibrate theta values");
    }

    using namespace rcpp::util;
    using namespace Rcpp;

    const std::vector<std::string> &d_ = row_names;
    std::vector<std::string> k_;
    for (std::size_t k = 1; k <= K; ++k) {
        k_.push_back(std::to_string(k));
    }

    return List::create(_["beta"] = named(log_beta.unaryExpr(exp), d_, k_),
                        _["delta"] = named(log_delta.unaryExpr(exp), d_, k_),
                        _["corr"] = named(Rtot_nk, col_names, k_),
                        _["theta"] = named(theta_nk, col_names, k_),
                        _["log.theta"] = named(log_theta_nk, col_names, k_),
                        _["colsum"] = named_rows(Ytot_n, row_names),
                        _["rownames"] = row_names,
                        _["colnames"] = col_names);
}

}} // namespace asap::regression
#endif
