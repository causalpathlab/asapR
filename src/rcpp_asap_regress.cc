#include "rcpp_asap_regress.hh"

//' Poisson regression to estimate factor loading
//'
//' @param Y D x N data matrix
//' @param log_x D x K log dictionary/design matrix
//' @param r_batch_effect D x B batch effect matrix (default: NULL)
//' @param a0 gamma(a0, b0) (default: 1)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param do_log1p do log(1+y) transformation (default: FALSE)
//' @param verbose verbosity (default: false)
//' @param do_stdize do the standardization of log_x
//'
// [[Rcpp::export]]
Rcpp::List
asap_regression(
    const Eigen::MatrixXf Y_,
    const Eigen::MatrixXf log_x,
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_batch_effect = R_NilValue,
    const double a0 = 1.,
    const double b0 = 1.,
    const std::size_t max_iter = 10,
    const bool do_log1p = false,
    const bool verbose = true,
    const bool do_stdize_x = false)
{

    exp_op<Mat> exp;
    at_least_one_op<Mat> at_least_one;
    softmax_op_t<Mat> softmax;
    log1p_op<Mat> log1p;

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    Mat Y_dn = do_log1p ? Y_.unaryExpr(log1p) : Y_;

    const Index D = Y_dn.rows();  // number of features
    const Index N = Y_dn.cols();  // number of samples
    const Index K = log_x.cols(); // number of topics

    const Mat delta_db = r_batch_effect.isNotNull() ?
        Rcpp::as<Mat>(Rcpp::NumericMatrix(r_batch_effect)) :
        Mat::Ones(D, 1);

    const Index B = delta_db.cols();
    ASSERT_RETL(D == delta_db.rows(), "batch effect should be of D x r");

    Mat logX_dk = log_x;
    stdizer_t<Mat> stdizer_x(logX_dk);
    residual_columns(logX_dk, delta_db);
    if (do_stdize_x)
        stdizer_x.colwise();

    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Mat, RNG>;
    RNG rng;

    Mat logRho_nk(Y_dn.cols(), K), rho_nk(Y_dn.cols(), K);

    gamma_t theta_nk(Y_dn.cols(), K, a0, b0, rng); // N x K

    const ColVec Y_n = Y_dn.colwise().sum().transpose();
    const ColVec Y_n1 = Y_n.unaryExpr(at_least_one);

    // X[j,k] = sum_i X[i,k]
    RowVec Xsum = logX_dk.unaryExpr(exp).colwise().sum();
    Mat x_nk = ColVec::Ones(N) * Xsum; // n x K
    Mat R_nk = (Y_dn.transpose() * logX_dk).array().colwise() / Y_n1.array();

    if (verbose)
        TLOG("Correlation with the topics");

    RowVec tempK(K);

    for (std::size_t t = 0; t < max_iter; ++t) {

        logRho_nk = R_nk + theta_nk.log_mean();

        for (Index jj = 0; jj < Y_dn.cols(); ++jj) {
            tempK = logRho_nk.row(jj);
            logRho_nk.row(jj) = softmax.log_row(tempK);
        }
        rho_nk = logRho_nk.unaryExpr(exp);

        theta_nk.update((rho_nk.array().colwise() * Y_n.array()).matrix(),
                        x_nk);
        theta_nk.calibrate();
    }

    if (verbose)
        TLOG("Calibrated topic portions");

    return Rcpp::List::create(Rcpp::_["beta"] = logX_dk.unaryExpr(exp),
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.theta.sd"] = theta_nk.log_sd(),
                              Rcpp::_["corr"] = R_nk,
                              Rcpp::_["latent"] = rho_nk,
                              Rcpp::_["log.latent"] = logRho_nk);
}

//' Poisson regression to estimate factor loading
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param row_file row names file (D x 1)
//' @param col_file column names file (N x 1)
//' @param mtx_idx_file matrix-market colum index file
//' @param log_x D x K log dictionary/design matrix
//' @param x_row_names row names log_x (D vector)
//' @param _log_batch_effect D x B batch effect matrix (default: NULL)
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param do_log1p do log(1+y) transformation
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_stdize do the standardization of log_x
//'
// [[Rcpp::export]]
Rcpp::List
asap_regression_mtx(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    const std::string mtx_idx_file,
    const Eigen::MatrixXf log_x,
    const Rcpp::StringVector &x_row_names,
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_batch_effect = R_NilValue,
    const double a0 = 1.,
    const double b0 = 1.,
    const std::size_t max_iter = 10,
    const bool do_log1p = false,
    const bool verbose = false,
    const std::size_t NUM_THREADS = 1,
    const std::size_t BLOCK_SIZE = 100,
    const bool do_stdize_x = false)
{

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    std::vector<std::string> pos2row;

    rcpp::util::copy(x_row_names, pos2row);

    //////////////////////////////////////
    // take care of different row names //
    //////////////////////////////////////

    const Index D = pos2row.size(); // dimensionality

    ASSERT_RETL(log_x.rows() == D,
                "#Rows in the log_x matrix !=  the size of x_row_names: "
                    << log_x.rows() << " != " << D);

    std::unordered_map<std::string, Index> row2pos;
    for (Index r = 0; r < pos2row.size(); ++r) {
        row2pos[pos2row.at(r)] = r;
    }

    ASSERT_RETL(row2pos.size() == D, "Redundant row names exist");
    TLOG_(verbose, "Found " << row2pos.size() << " unique row names");

    //////////////
    // functors //
    //////////////

    exp_op<Mat> exp;
    at_least_one_op<Mat> at_least_one;
    softmax_op_t<Mat> softmax;
    log1p_op<Mat> log1p;

    ///////////////////////////////
    // read mtx data information //
    ///////////////////////////////

    CHK_RETL_(convert_bgzip(mtx_file),
              "mtx file " << mtx_file << " was not bgzipped.");

    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    const Index N = info.max_col;        // number of cells
    const Index K = log_x.cols();        // number of topics
    const Index block_size = BLOCK_SIZE; // memory block size

    std::vector<std::string> coln;
    CHK_RETL_(read_vector_file(col_file, coln),
              "Failed to read the column name file: " << col_file);

    ASSERT_RETL(N == coln.size(),
                "Different #columns: " << N << " vs. " << coln.size());

    mtx_data_t data(mtx_data_t::MTX { mtx_file },
                    mtx_data_t::ROW { row_file },
                    mtx_data_t::IDX { mtx_idx_file });

    ///////////////////////////////
    // preprocess logX_dk matrix //
    ///////////////////////////////

    if (verbose) {
        TLOG("Start recalibrating column-wise loading parameters...");
        TLOG("Theta: " << N << " x " << K);
    }

    const Mat delta_db = r_batch_effect.isNotNull() ?
        Rcpp::as<Mat>(Rcpp::NumericMatrix(r_batch_effect)) :
        Mat::Ones(D, 1);

    const Index B = delta_db.cols();
    ASSERT_RETL(D == delta_db.rows(), "batch effect should be of D x r");

    Mat logX_dk = log_x;
    if (r_batch_effect.isNotNull()) {
        residual_columns(logX_dk, delta_db);
        if (verbose)
            TLOG("Removed the batch effects");
    }

    stdizer_t<Mat> stdizer_x(logX_dk);

    if (do_stdize_x)
        stdizer_x.colwise();

    TLOG_(verbose, "lnX: " << logX_dk.rows() << " x " << logX_dk.cols());

    Mat Rtot_nk(N, K);
    Mat Ztot_nk(N, K);
    Mat logZtot_nk(N, K);
    Mat thetaTot_nk(N, K);
    Mat logThetaTot_nk(N, K);

    Index Nprocessed = 0;

    if (verbose) {
        Rcpp::Rcerr << "Calibrating " << N << " columns..." << std::endl;
    }

    data.relocate_rows(row2pos);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {
        Index ub = std::min(N, block_size + lb);
        const SpMat y = data.read_reloc(lb, ub);

        //////////////////////
        // normalize matrix //
        //////////////////////

        Mat Y_dn = do_log1p ? y.unaryExpr(log1p) : y;

        ///////////////////////////////////////
        // do log1p transformation if needed //
        ///////////////////////////////////////
        const Index n = Y_dn.cols();
        using RNG = dqrng::xoshiro256plus;
        using gamma_t = gamma_param_t<Mat, RNG>;
        RNG rng;

        const ColVec Y_n = Y_dn.colwise().sum().transpose(); // n x 1
        const ColVec Y_n1 = Y_n.unaryExpr(at_least_one);     // n x 1
        const ColVec Y_d = Y_dn.rowwise().sum();             // d x 1
        const ColVec Y_d1 = Y_d.unaryExpr(at_least_one);     // d x 1
        gamma_t theta_b(Y_dn.cols(), K, a0, b0, rng);        // n x K

        Mat logRho_nk(Y_dn.cols(), K), rho_nk(Y_dn.cols(), K);

        const Mat R_nk =
            (Y_dn.transpose() * logX_dk).array().colwise() / Y_n1.array();

        // X[j,k] = sum_i X[i,k]
        RowVec Xsum = logX_dk.unaryExpr(exp).colwise().sum();
        Mat x_nk = ColVec::Ones(n) * Xsum; // n x K

        RowVec tempK(K);

        for (std::size_t t = 0; t < max_iter; ++t) {

            logRho_nk = R_nk + theta_b.log_mean();

            for (Index jj = 0; jj < Y_dn.cols(); ++jj) {
                tempK = logRho_nk.row(jj);
                logRho_nk.row(jj) = softmax.log_row(tempK);
            }
            rho_nk = logRho_nk.unaryExpr(exp);

            theta_b.update((rho_nk.array().colwise() * Y_n.array()).matrix(),
                           x_nk);
            theta_b.calibrate();
        }

#pragma omp critical
        {
            for (Index i = 0; i < (ub - lb); ++i) {
                const Index j = i + lb;
                Ztot_nk.row(j) = rho_nk.row(i);
                logZtot_nk.row(j) = logRho_nk.row(i);
                Rtot_nk.row(j) = R_nk.row(i);
                thetaTot_nk.row(j) = theta_b.mean().row(i);
                logThetaTot_nk.row(j) = theta_b.log_mean().row(i);
            }

            Nprocessed += Y_dn.cols();
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
    TLOG("Done");

    using namespace rcpp::util;
    using namespace Rcpp;

    std::vector<std::string> d_ = pos2row;
    std::vector<std::string> k_;
    for (std::size_t k = 1; k <= K; ++k)
        k_.push_back(std::to_string(k));

    return List::create(_["beta"] = named(logX_dk.unaryExpr(exp), d_, k_),
                        _["theta"] = named(thetaTot_nk, coln, k_),
                        _["corr"] = named(Rtot_nk, coln, k_),
                        _["latent"] = named(Ztot_nk, coln, k_),
                        _["log.latent"] = named(logZtot_nk, coln, k_),
                        _["log.theta"] = named(logThetaTot_nk, coln, k_));
}
