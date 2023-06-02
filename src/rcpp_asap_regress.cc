#include "rcpp_asap_regress.hh"

//' Poisson regression to estimate factor loading
//'
//' @param Y D x N data matrix
//' @param log_x D x K log dictionary/design matrix
//' @param r_batch_effect D x B batch effect matrix (default: NULL)
//' @param a0 gamma(a0, b0) (default: 1)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param do_scale scale each column by standard deviation (default: TRUE)
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
    const bool do_scale = false,
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
    if (do_scale) {
        normalize_columns(Y_dn);
        scale_columns(Y_dn);
    }

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
                              Rcpp::_["corr"] = R_nk,
                              Rcpp::_["latent"] = rho_nk,
                              Rcpp::_["log.latent"] = logRho_nk);
}

//' Poisson regression to estimate factor loading
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param mtx_idx_file matrix-market colum index file
//' @param log_x D x K log dictionary/design matrix
//' @param r_batch_effect D x B batch effect matrix (default: NULL)
//' @param r_x_row_names (default: NULL)
//' @param r_mtx_row_names (default: NULL)
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param do_scale scale each column by standard deviation (default: TRUE)
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
    const std::string mtx_idx_file,
    const Eigen::MatrixXf log_x,
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_batch_effect = R_NilValue,
    const Rcpp::Nullable<Rcpp::StringVector> r_x_row_names = R_NilValue,
    const Rcpp::Nullable<Rcpp::StringVector> r_mtx_row_names = R_NilValue,
    const Rcpp::Nullable<Rcpp::StringVector> r_taboo_names = R_NilValue,
    const double a0 = 1.,
    const double b0 = 1.,
    const std::size_t max_iter = 10,
    const bool do_scale = false,
    const bool do_log1p = false,
    const bool verbose = false,
    const std::size_t NUM_THREADS = 1,
    const std::size_t BLOCK_SIZE = 100,
    const bool do_stdize_x = false)
{

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    //////////////////////////////////////
    // take care of different row names //
    //////////////////////////////////////

    std::vector<Index> x_row_idx;
    std::vector<Index> mtx_row_idx;
    bool take_row_subset = false;

    std::unordered_map<Rcpp::String, Index> taboo;
    if (r_taboo_names.isNotNull()) {
        const Rcpp::StringVector taboo_names(r_taboo_names);
        for (auto r : taboo_names) {
            taboo[r] = 1;
        }
    }

    if (r_x_row_names.isNotNull() && r_mtx_row_names.isNotNull()) {

        const Rcpp::StringVector x_row_names(r_x_row_names);
        const Rcpp::StringVector mtx_row_names(r_mtx_row_names);

        std::unordered_map<Rcpp::String, Index> mtx_row_pos;
        {
            Index mi = 0;
            for (auto r : mtx_row_names) {
                mtx_row_pos[r] = mi++;
            }
        }
        Index xi = 0;
        for (auto r : x_row_names) {
            if (mtx_row_pos.count(r) > 0 && taboo.count(r) == 0) {
                Index mi = mtx_row_pos[r];
                mtx_row_idx.emplace_back(mi);
                x_row_idx.emplace_back(xi);
            }
            ++xi;
        }
        if (verbose) {
            TLOG(x_row_idx.size() << " rows matched between X and MTX");
        }
        ASSERT_RETL(x_row_idx.size() > 0,
                    " At least one common name should be present "
                        << " in both x_row_names and mtx_row_names");
        take_row_subset = true;
    }

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

    const Index D = info.max_row;        // dimensionality
    const Index N = info.max_col;        // number of cells
    const Index K = log_x.cols();        // number of topics
    const Index block_size = BLOCK_SIZE; // memory block size

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

    ///////////////////////////////
    // preprocess logX_dk matrix //
    ///////////////////////////////

    eigen_triplet_reader_remapped_rows_cols_t::index_map_t mtx_row_loc;

    if (!take_row_subset) {
        ASSERT_RETL(log_x.rows() == D,
                    "The log-X matrix contains different"
                        << " numbers of rows from the one in " << mtx_file
                        << ": " << log_x.rows() << " vs. " << D);
    } else {
        const Index d = x_row_idx.size();
        Mat temp(d, logX_dk.cols());
        for (Index r = 0; r < d; ++r) {
            temp.row(r) = log_x.row(x_row_idx.at(r));
        }
        logX_dk = temp;

        for (Index r = 0; r < d; ++r) {
            mtx_row_loc[mtx_row_idx.at(r)] = r;
        }
    }

    stdizer_t<Mat> stdizer_x(logX_dk);

    if (do_stdize_x)
        stdizer_x.colwise();

    if (verbose) {
        TLOG("log.X: " << logX_dk.rows() << " x " << logX_dk.cols());
    }

    Mat R_tot(N, K);
    Mat Z_tot(N, K);
    Mat logZ_tot(N, K);
    Mat theta_tot(N, K);
    Mat log_theta_tot(N, K);

    Index Nprocessed = 0;

    if (verbose) {
        Rcpp::Rcerr << "Calibrating total = " << N << " columns" << std::endl;
    }

    std::vector<Index> mtx_idx;

    CHK_RETL_(read_mmutil_index(mtx_idx_file, mtx_idx),
              "Failed to read the index file:" << std::endl
                                               << mtx_idx_file << std::endl
                                               << "Consider rebuilding it."
                                               << std::endl);

    TLOG("Read the mtx index file: " << mtx_idx_file
                                     << " n=" << mtx_idx.size());

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {
        Index ub = std::min(N, block_size + lb);
        Index col_lb_mem = mtx_idx.at(lb);
        Index col_ub_mem = ub < N ? mtx_idx.at(ub) : 0; // 0 = the end

        const SpMat y = take_row_subset ?
            (read_eigen_sparse_subset_row_col(mtx_file,
                                              mtx_row_loc,
                                              lb,
                                              ub,
                                              col_lb_mem,
                                              col_ub_mem)) :
            (read_eigen_sparse_subset_col(mtx_file,
                                          lb,
                                          ub,
                                          col_lb_mem,
                                          col_ub_mem));

        //////////////////////
        // normalize matrix //
        //////////////////////

        Mat Y_dn = do_log1p ? y.unaryExpr(log1p) : y;

        if (do_scale) {
            normalize_columns(Y_dn);
            scale_columns(Y_dn);
        }

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
                Z_tot.row(j) = rho_nk.row(i);
                logZ_tot.row(j) = logRho_nk.row(i);
                R_tot.row(j) = R_nk.row(i);
                theta_tot.row(j) = theta_b.mean().row(i);
                log_theta_tot.row(j) = theta_b.log_mean().row(i);
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

    Rcpp::Rcerr << std::endl;
    TLOG("Done");

    return Rcpp::List::create(Rcpp::_["beta"] = logX_dk.unaryExpr(exp),
                              Rcpp::_["theta"] = theta_tot,
                              Rcpp::_["corr"] = R_tot,
                              Rcpp::_["latent"] = Z_tot,
                              Rcpp::_["log.latent"] = logZ_tot,
                              Rcpp::_["log.theta"] = log_theta_tot);
}
