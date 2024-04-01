#ifndef RCPP_ASAP_REGRESSION_HH_
#define RCPP_ASAP_REGRESSION_HH_

#include "rcpp_asap.hh"
#include "rcpp_asap_batch.hh"
#include "rcpp_asap_mtx_data.hh"

struct topic_stat_options_t {
    bool do_log1p;
    bool verbose;
    std::size_t NUM_THREADS;
    std::size_t BLOCK_SIZE;
    std::size_t MAX_ROW_WORD;
    char ROW_WORD_SEP;
    std::size_t MAX_COL_WORD;
    char COL_WORD_SEP;
    bool do_stdize_x;
};

template <typename Derived1, typename Derived2, typename Derived3, typename OPT>
int
asap_topic_stat_mtx(const std::string mtx_file,
                    const std::string row_file,
                    const std::string col_file,
                    const std::string idx_file,
                    const Eigen::MatrixBase<Derived1> &_log_x,
                    const std::vector<std::string> &pos2row,
                    const OPT &options,
                    Eigen::MatrixBase<Derived2> &_r_nk,
                    Eigen::MatrixBase<Derived3> &_y_n)
{

    const bool do_log1p = options.do_log1p;
    const bool verbose = options.verbose;
    const std::size_t NUM_THREADS = options.NUM_THREADS;
    const std::size_t BLOCK_SIZE = options.BLOCK_SIZE;
    const std::size_t MAX_ROW_WORD = options.MAX_ROW_WORD;
    const char ROW_WORD_SEP = options.ROW_WORD_SEP;
    const std::size_t MAX_COL_WORD = options.MAX_COL_WORD;
    const char COL_WORD_SEP = options.COL_WORD_SEP;

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    Derived1 logX_dk = _log_x.derived();
    if (options.do_stdize_x) {
        standardize_columns_inplace(logX_dk);
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

    ///////////////////////////////
    // read mtx data information //
    ///////////////////////////////

    CHK_RET_(convert_bgzip(mtx_file),
             "mtx file " << mtx_file << " was not bgzipped.");

    mm_info_reader_t info;
    CHK_RET_(peek_bgzf_header(mtx_file, info),
             "Failed to read the size of this mtx file:" << mtx_file);

    const Index N = info.max_col;        // number of cells
    const Index K = logX_dk.cols();      // number of topics
    const Index block_size = BLOCK_SIZE; // memory block size

    mtx_data_t data(mtx_data_t::MTX { mtx_file },
                    mtx_data_t::ROW { row_file },
                    mtx_data_t::IDX { idx_file },
                    MAX_ROW_WORD,
                    ROW_WORD_SEP);

    TLOG_(verbose, "lnX: " << logX_dk.rows() << " x " << logX_dk.cols());
    Rtot_nk.resize(N, K);
    Ytot_n.resize(N, 1);
    Index Nprocessed = 0;

    if (verbose) {
        Rcpp::Rcerr << "Calibrating " << N << " columns..." << std::endl;
    }

    data.relocate_rows(row2pos);

    at_least_one_op<Mat> at_least_one;
    at_least_zero_op<Mat> at_least_zero;
    exp_op<Mat> exp;
    log1p_op<Mat> log1p;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {
        Index ub = std::min(N, block_size + lb);
        const SpMat y = data.read_reloc(lb, ub);

        ///////////////////////////////////////
        // do log1p transformation if needed //
        ///////////////////////////////////////

        Mat y_dn = do_log1p ? y.unaryExpr(log1p) : y;

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

    if (!verbose)
        Rcpp::Rcerr << std::endl;
    TLOG_(verbose, "done -> topic stat");

    return EXIT_SUCCESS;
}

template <typename Derived1,
          typename Derived0,
          typename Derived2,
          typename Derived3,
          typename OPT>
int
asap_topic_stat_ipw_mtx(const std::string mtx_file,
                        const std::string row_file,
                        const std::string col_file,
                        const std::string idx_file,
                        const Eigen::MatrixBase<Derived1> &_log_x,
                        const Eigen::MatrixBase<Derived0> &_log_w,
                        const std::vector<std::string> &pos2row,
                        const OPT &options,
                        Eigen::MatrixBase<Derived2> &_r_nk,
                        Eigen::MatrixBase<Derived3> &_y_n)
{

    const bool do_log1p = options.do_log1p;
    const bool verbose = options.verbose;
    const std::size_t NUM_THREADS = options.NUM_THREADS;
    const std::size_t BLOCK_SIZE = options.BLOCK_SIZE;
    const std::size_t MAX_ROW_WORD = options.MAX_ROW_WORD;
    const char ROW_WORD_SEP = options.ROW_WORD_SEP;
    const std::size_t MAX_COL_WORD = options.MAX_COL_WORD;
    const char COL_WORD_SEP = options.COL_WORD_SEP;

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    Derived1 logX_dk = _log_x.derived();
    if (options.do_stdize_x) {
        standardize_columns_inplace(logX_dk);
    }

    Derived2 &Rtot_nk = _r_nk.derived();
    Derived3 &Ytot_n = _y_n.derived();

    Derived0 logW_db = _log_w.derived();

    ///////////////////////////////////
    // check inverse weight features //
    ///////////////////////////////////

    ASSERT_RET(logW_db.rows() == logX_dk.rows(),
               "X and W should have the same number of rows");

    const Index B = logW_db.cols(); // batches

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

    ///////////////////////////////
    // read mtx data information //
    ///////////////////////////////

    CHK_RET_(convert_bgzip(mtx_file),
             "mtx file " << mtx_file << " was not bgzipped.");

    mm_info_reader_t info;
    CHK_RET_(peek_bgzf_header(mtx_file, info),
             "Failed to read the size of this mtx file:" << mtx_file);

    const Index N = info.max_col;        // number of cells
    const Index K = logX_dk.cols();      // number of topics
    const Index block_size = BLOCK_SIZE; // memory block size

    mtx_data_t data(mtx_data_t::MTX { mtx_file },
                    mtx_data_t::ROW { row_file },
                    mtx_data_t::IDX { idx_file },
                    MAX_ROW_WORD,
                    ROW_WORD_SEP);

    TLOG_(verbose, "lnX: " << logX_dk.rows() << " x " << logX_dk.cols());
    Rtot_nk.resize(N, K);
    Ytot_n.resize(N, 1);
    Index Nprocessed = 0;

    if (verbose) {
        Rcpp::Rcerr << "Calibrating " << N << " columns..." << std::endl;
    }

    data.relocate_rows(row2pos);

    at_least_one_op<Mat> at_least_one;
    at_least_zero_op<Mat> at_least_zero;
    exp_op<Mat> exp;
    softmax_op_t<Mat> softmax;
    log1p_op<Mat> log1p;

    RowVec tempB(B);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {
        Index ub = std::min(N, block_size + lb);
        const SpMat y = data.read_reloc(lb, ub);

        ///////////////////////////////////////
        // do log1p transformation if needed //
        ///////////////////////////////////////

        Mat y_dn = do_log1p ? y.unaryExpr(log1p) : y;

        const Index n = y_dn.cols();

        ColVec Y_n = y_dn.colwise().sum().transpose(); // n x 1
        ColVec Y_n1 = Y_n.unaryExpr(at_least_one);     // n x 1

        //////////////////////////////
        // calibrate the null model //
        //////////////////////////////

        Mat log_r0_nb =
            (y_dn.transpose() * logW_db).array().colwise() / Y_n1.array();

        for (Index j = 0; j < n; ++j) {
            tempB = log_r0_nb.row(j);
            log_r0_nb.row(j) = softmax.log_row(tempB);
        }

        Mat y0_dn = logW_db * (log_r0_nb.unaryExpr(exp)).transpose();

        // TODO
        // ( sum_g Y(g,j) logX(g, k) / Y0(g,j) ) / ( sum_g Y(g,j) / Y0(g,j) )

        ///////////////////////////
        // parameter of interest //
        ///////////////////////////

        // TODO: stabilize Y/Y0

        // ColVec Y_n = y_dn.colwise().sum().transpose(); // n x 1
        // ColVec Y_n1 = Y_n.unaryExpr(at_least_one);     // n x 1

        Mat R_nk =
            (y_dn.transpose() * logX_dk).array().colwise() / Y_n1.array();

        // TODO

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

    if (!verbose)
        Rcpp::Rcerr << std::endl;
    TLOG_(verbose, "done -> topic stat");

    return EXIT_SUCCESS;
}

#endif
