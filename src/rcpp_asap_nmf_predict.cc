#include "rcpp_asap.hh"

//' Predict NMF loading -- this may be slow for high-dim data
//'
//' @param mtx_file matrix-market-formatted data file (bgzip)
//' @param memory_location column indexing for the mtx
//' @param beta_dict row x factor dictionary (beta) matrix
//' @param do_beta_rescale rescale the columns of the beta matrix
//' @param collapsing r x row collapsing matrix (r < row)
//' @param mcem number of Monte Carlo Expectation Maximization
//' @param burnin burn-in period
//' @param latent_iter latent sampling steps
//' @param thining thining interval in record keeping
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//'
// [[Rcpp::export]]
Rcpp::List
asap_predict_mtx(const std::string mtx_file,
                 const Rcpp::NumericVector &memory_location,
                 const Eigen::MatrixXf beta_dict,
                 const bool do_beta_rescale = true,
                 Rcpp::Nullable<Rcpp::NumericMatrix> collapsing = R_NilValue,
                 const std::size_t mcem = 100,
                 const std::size_t burnin = 10,
                 const std::size_t latent_iter = 10,
                 const std::size_t thining = 3,
                 const double a0 = 1.,
                 const double b0 = 1.,
                 const std::size_t rseed = 42,
                 const bool verbose = false,
                 const std::size_t NUM_THREADS = 1,
                 const std::size_t BLOCK_SIZE = 100,
                 const bool gibbs_sampling = false)
{
    CHK_RETL(convert_bgzip(mtx_file));
    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    const Index D = info.max_row;     // dimensionality
    const Index N = info.max_col;     // number of cells
    const Index K = beta_dict.cols(); // number of topics
    const Index block_size = BLOCK_SIZE;
    const Scalar TOL = 1e-20;

    ASSERT_RETL(beta_dict.rows() == D,
                "incompatible Beta with the file: " << mtx_file);

    if (verbose) {
        TLOG("Dictionary Beta: " << D << " x " << K << " ["
                                 << beta_dict.minCoeff() << " .. "
                                 << beta_dict.maxCoeff() << "]");
    }

    const bool do_collapse = collapsing.isNotNull() ? true : false;

    Mat C;

    if (do_collapse) {
        C = Rcpp::as<Mat>(Rcpp::NumericMatrix(collapsing));
        ASSERT_RETL(C.cols() == D, "incompatible collapsing matrix");
    }

    Mat onesD = Mat::Ones(D, 1);

    Mat theta(N, K);
    Mat theta_sd(N, K);
    Mat log_theta(N, K);
    Mat log_theta_sd(N, K);

    if (verbose) {
        TLOG("Start recalibrating column-wise loading parameters...");
        TLOG("Theta: " << N << " x " << K);
        Rcpp::Rcerr << "Calibrating total = " << N << std::flush;
    }

    Mat llik_mat(static_cast<Index>(std::ceil(N / block_size)),
                 static_cast<Index>(mcem + burnin));

    Index Nprocessed = 0;

    Mat B = do_collapse ? (C * beta_dict) : beta_dict;
    if (do_beta_rescale) {
        normalize_columns(B);
    }

    auto log_op = [&TOL](const Scalar &x) -> Scalar {
        if (x < TOL)
            return fasterlog(TOL);
        return fasterlog(x);
    };

    Mat log_B = B.unaryExpr(log_op);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);
        const Index b = lb / block_size;

        ///////////////////////////////////////
        // memory location = 0 means the end //
        ///////////////////////////////////////

        const Index lb_mem = memory_location[lb];
        const Index ub_mem = ub < N ? memory_location[ub] : 0;

        const SpMat xx =
            read_eigen_sparse_subset_col(mtx_file, lb, ub, lb_mem, ub_mem);

        const Mat Y = do_collapse ? (C * xx).eval() : xx;

        running_stat_t<Mat> stat(Y.cols(), K);
        running_stat_t<Mat> log_stat(Y.cols(), K);

        using RNG = dqrng::xoshiro256plus;
        using gamma_t = gamma_param_t<Mat, RNG>;
        RNG rng(rseed + lb);

        using latent_t = latent_matrix_t<RNG>;
        latent_t aux(Y.rows(), Y.cols(), K, rng);
        gamma_t theta_b(Y.cols(), K, a0, b0, rng);
        gamma_t row_degree(Y.rows(), 1, a0, b0, rng);
        gamma_t column_degree(Y.cols(), 1, a0, b0, rng);

        const Mat onesN = Mat::Ones(Y.cols(), 1);

        matrix_sampler_t<Mat, RNG> row_proposal(rng, K);

        for (std::size_t t = 0; t < (mcem + burnin); ++t) {

            /////////////////////////////
            // E-step: latent sampling //
            /////////////////////////////

            for (std::size_t s = 0; s < latent_iter; ++s) {
                if (gibbs_sampling) {
                    aux.gibbs_sample_row_col(log_B, theta_b.log_mean());
                } else {
                    aux.mh_sample_row_col(row_proposal.sample_logit(log_B),
                                          theta_b.log_mean());
                }
            }

            ///////////////////////////////
            // M-step: update parameters //
            ///////////////////////////////

            Vec S = B.transpose() * row_degree.mean();

            for (Index k = 0; k < K; ++k) {
                theta_b.update_col(aux.slice_k(Y, k).transpose() * onesD,
                                   column_degree.mean() * S(k),
                                   k);
            }

            theta_b.calibrate();

            ///////////////////////
            // Calibrate degrees //
            ///////////////////////

            column_degree.update(Y.transpose() * onesD,
                                 theta_b.mean() *
                                     (B.transpose() * row_degree.mean()));
            column_degree.calibrate();

            row_degree.update(Y * onesN,
                              B *
                                  (theta_b.mean().transpose() *
                                   column_degree.mean()));
            row_degree.calibrate();

            if (t >= burnin && t % thining == 0) {
                stat(theta_b.mean());
                log_stat(theta_b.log_mean());
            }

            Scalar llik = 0;
            for (Index k = 0; k < K; ++k) {
                llik += (aux.slice_k(Y, k).array().rowwise() *
                         theta_b.log_mean().col(k).transpose().array())
                            .sum();
            }
            llik -= (B * theta_b.mean().transpose()).sum();
            llik_mat(b, t) = llik;
        }

        Nprocessed += Y.cols();
        if (verbose) {
            Rcpp::Rcerr << "\rprocessed: " << Nprocessed << std::flush;
        } else {
            Rcpp::Rcerr << "+ " << std::flush;
        }

        const Mat _mean = stat.mean(), _sd = stat.sd();
        const Mat _log_mean = log_stat.mean(), _log_sd = log_stat.sd();

        for (Index i = 0; i < (ub - lb); ++i) {
            const Index j = i + lb;
            theta.row(j) = _mean.row(i);
            theta_sd.row(j) = _sd.row(i);
            log_theta.row(j) = _log_mean.row(i);
            log_theta_sd.row(j) = _log_sd.row(i);
        }
    }

    Rcpp::Rcerr << std::endl;
    TLOG("Done");

    return Rcpp::List::create(Rcpp::_["theta"] = theta,
                              Rcpp::_["theta.sd"] = theta_sd,
                              Rcpp::_["log.theta"] = log_theta,
                              Rcpp::_["log.theta.sd"] = log_theta_sd,
                              Rcpp::_["log.likelihood"] = llik_mat,
                              Rcpp::_["collapsing.factor"] = C,
                              Rcpp::_["beta.collapsed"] = B);
}
