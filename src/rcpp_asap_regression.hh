#ifndef RCPP_ASAP_REGRESSION_HH_
#define RCPP_ASAP_REGRESSION_HH_

#include "rcpp_asap.hh"
#include "rcpp_asap_batch.hh"
#include "rcpp_mtx_data.hh"
#include "rcpp_eigenSparse_data.hh"

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
        , CELL_NORM(1e4)
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
    double CELL_NORM;
};

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

template <typename Data,
          typename Derived0,
          typename Derived1,
          typename Derived2,
          typename Derived3>
int run_pmf_stat_adj(Data &data,
                     const Eigen::MatrixBase<Derived0> &_log_x,
                     const Eigen::MatrixBase<Derived1> &_log_x0,
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
pmf_stat_(const Eigen::SparseMatrix<float> &y_dn,
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

    const Scalar cell_norm = options.CELL_NORM;
    const bool do_log1p = options.do_log1p;
    const bool verbose = options.verbose;
    const std::size_t NUM_THREADS =
        (options.NUM_THREADS > 0 ? options.NUM_THREADS : omp_get_max_threads());
    const std::size_t BLOCK_SIZE = options.BLOCK_SIZE;

    TLOG_(verbose, NUM_THREADS << " threads");

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    Mat logX_dk = _log_x.derived();
    if (options.do_stdize_x) {
        standardize_columns_inplace(logX_dk);
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
        SpMat _y = data.read_reloc(lb, ub);
        normalize_columns_inplace(_y);
        const Mat y = _y * cell_norm;

        ///////////////////////////////////////
        // do log1p transformation if needed //
        ///////////////////////////////////////

        const Mat y_dn = do_log1p ? y.unaryExpr(log1p) : y;

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

template <typename Data,
          typename Derived0,
          typename Derived1,
          typename Derived2,
          typename Derived3>
int
run_pmf_stat_adj(Data &data,
                 const Eigen::MatrixBase<Derived0> &_log_x,
                 const Eigen::MatrixBase<Derived1> &_log_x0,
                 const std::vector<std::string> &pos2row,
                 const stat_options_t &options,
                 Eigen::MatrixBase<Derived2> &_r_nk,
                 Eigen::MatrixBase<Derived3> &_y_n)
{

    const Scalar cell_norm = options.CELL_NORM;
    const bool do_log1p = options.do_log1p;
    const bool verbose = options.verbose;
    const std::size_t NUM_THREADS = options.NUM_THREADS;
    const std::size_t BLOCK_SIZE = options.BLOCK_SIZE;

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    Mat logX_dk = _log_x.derived();
    if (options.do_stdize_x) {
        standardize_columns_inplace(logX_dk);
    }

    Mat logX0_db = _log_x0.derived();
    if (options.do_stdize_x) {
        standardize_columns_inplace(logX0_db);
    }

    Derived2 &Rtot_nk = _r_nk.derived();
    Derived3 &Ytot_n = _y_n.derived();

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
    const Index B = logX0_db.cols();     // number of batches
    const Index block_size = BLOCK_SIZE; // memory block size

    Mat R0tot_nk = Mat::Zero(N, K);

    TLOG_(verbose, "lnX: " << logX_dk.rows() << " x " << logX_dk.cols());
    TLOG_(verbose, "lnX0: " << logX0_db.rows() << " x " << logX0_db.cols());
    Rtot_nk.resize(N, K);
    Ytot_n.resize(N, 1);
    Index Nprocessed = 0;

    if (verbose) {
        Rcpp::Rcerr << "Calibrating " << N << " samples..." << std::endl;
    }

    data.relocate_rows(row2pos);

    at_least_one_op<Mat> at_least_one;
    at_least_zero_op<Mat> at_least_zero;
    exp_op<Mat> exp;
    softmax_op_t<Mat> softmax;
    log1p_op<Mat> log1p;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);
        SpMat _y = data.read_reloc(lb, ub);
        normalize_columns_inplace(_y);
        const Mat y = _y * cell_norm;

        ///////////////////////////////////////
        // do log1p transformation if needed //
        ///////////////////////////////////////

        const Mat y_dn = do_log1p ? y.unaryExpr(log1p) : y;

        const Index n = y_dn.cols();

        ColVec Y_n = y_dn.colwise().sum().transpose(); // n x 1
        ColVec Y_n1 = Y_n.unaryExpr(at_least_one);     // n x 1

        //////////////////////////////
        // calibrate the null model //
        //////////////////////////////

        Mat log_r0_nb =
            (y_dn.transpose() * logX0_db).array().colwise() / Y_n1.array();

        RowVec tempB(B);

        for (Index j = 0; j < n; ++j) {
            tempB = log_r0_nb.row(j);
            log_r0_nb.row(j) = softmax.log_row(tempB);
        }

        const Mat y0_dn =
            logX0_db.unaryExpr(exp) * (log_r0_nb.unaryExpr(exp)).transpose();

        ColVec Y0_n = y0_dn.colwise().sum().transpose(); // n x 1
        ColVec Y0_n1 = Y0_n.unaryExpr(at_least_one);     // n x 1

        ///////////////////////////
        // parameter of interest //
        ///////////////////////////

        Mat R_nk =
            (y_dn.transpose() * logX_dk).array().colwise() / Y_n1.array();

        // Adjust R0_nk
        Mat R0_nk =
            (y0_dn.transpose() * logX_dk).array().colwise() / Y0_n1.array();

#pragma omp critical
        {
            for (Index i = 0; i < (ub - lb); ++i) {
                const Index j = i + lb;
                Rtot_nk.row(j) = R_nk.row(i);
                R0tot_nk.row(j) = R0_nk.row(i);
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

    residual_columns_inplace(Rtot_nk, R0tot_nk);

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

}} // namespace asap::regression
#endif
