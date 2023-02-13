#include "rcpp_asap.hh"

//' Poisson regression to estimate factor loading
//'
//' @param mtx_file matrix-market-formatted data file (bgzip)
//' @param memory_location column indexing for the mtx
//' @param log_x D x K log dictionary/design matrix
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//'
// [[Rcpp::export]]
Rcpp::List
asap_regression_mtx(const std::string mtx_file,
                    const Rcpp::NumericVector &memory_location,
                    const Eigen::MatrixXf log_x,
                    const double a0 = 1.,
                    const double b0 = 1.,
                    const std::size_t max_iter = 10,
                    const bool verbose = false,
                    const std::size_t NUM_THREADS = 1,
                    const std::size_t BLOCK_SIZE = 100)
{

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    CHK_RETL(convert_bgzip(mtx_file));
    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    const Index D = info.max_row; // dimensionality
    const Index N = info.max_col; // number of cells
    const Index K = log_x.cols(); // number of topics
    const Index block_size = BLOCK_SIZE;
    const Scalar TOL = 1e-20;

    auto exp_op = [](const Scalar &_x) -> Scalar { return fasterexp(_x); };

    ASSERT_RETL(log_x.rows() == D, "incompatible log X: " << mtx_file);

    if (verbose) {
        TLOG("Start recalibrating column-wise loading parameters...");
        TLOG("Theta: " << N << " x " << K);
        Rcpp::Rcerr << "Calibrating total = " << N << std::flush;
    }

    const Mat log_X = standardize(log_x);
    const RowVec Xsum = log_X.unaryExpr(exp_op).colwise().sum();

    Mat Z_tot(N, K);
    Mat theta_tot(N, K);
    Mat log_theta_tot(N, K);

    Index Nprocessed = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);

        const Index l = memory_location[lb];
        const Index u = ub < N ? memory_location[ub] : 0; // 0 = the end
        const SpMat y = read_eigen_sparse_subset_col(mtx_file, lb, ub, l, u);

        Mat Y = Mat(y);

        using RNG = dqrng::xoshiro256plus;
        using gamma_t = gamma_param_t<Mat, RNG>;
        RNG rng;
        softmax_op_t<Mat> softmax;

        ColVec Ysum = Y.colwise().sum().transpose(); // N x 1
        gamma_t theta_b(Y.cols(), K, a0, b0, rng);   // N x K
        Mat log_Z(Y.cols(), K), Z(Y.cols(), K);      // N x K
        Mat R = (Y.transpose() * log_X).array().colwise() / Ysum.array();
        //          N x D        D x K                      N x 1

        ColVec onesN(N); // N x 1
        onesN.setOnes(); //

        for (std::size_t t = 0; t < max_iter; ++t) {

            log_Z = theta_b.log_mean() + R;
            for (Index i = 0; i < Y.cols(); ++i) {
                Z.row(i) = softmax(log_Z.row(i));
            }

            for (Index k = 0; k < K; ++k) {
                const Scalar xk = Xsum(k);
                theta_b.update_col(Z.col(k).cwiseProduct(Ysum), onesN * xk, k);
            }
            theta_b.calibrate();
        }

        for (Index i = 0; i < (ub - lb); ++i) {
            const Index j = i + lb;
            Z_tot.row(j) = Z.row(i);
            theta_tot.row(j) = theta_b.mean().row(i);
            log_theta_tot.row(j) = theta_b.log_mean().row(i);
        }

        Nprocessed += Y.cols();
        if (verbose) {
            Rcpp::Rcerr << "\rprocessed: " << Nprocessed << std::flush;
        } else {
            Rcpp::Rcerr << "+ " << std::flush;
        }
    }

    Rcpp::Rcerr << std::endl;
    TLOG("Done");

    return Rcpp::List::create(Rcpp::_["beta"] = log_X.unaryExpr(exp_op),
			      Rcpp::_["theta"] = theta_tot,
                              Rcpp::_["latent"] = Z_tot,
                              Rcpp::_["log.theta"] = log_theta_tot);
}
