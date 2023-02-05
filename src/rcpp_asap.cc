#include "rcpp_asap.hh"

//' Predict NMF loading -- this may be slow for high-dim data
//'
//' @param mtx_file matrix-market-formatted data file (bgzip)
//' @param memory_location column indexing for the mtx
//' @param beta_dict row x factor dictionary
//' @param mcem number of Monte Carlo Expectation Maximization
//' @param burnin burn-in period
//' @param thining thining interval in record keeping
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param rseed random seed
//' @param verbose verbosity
//' @param do_collapse collapse the dictionary matrix by clustering
//' @param discrete_collapse do the row collapsing after discretization
//' @param collapsing_level # clusters while collapsing
//' @param collapsing_dpm_alpha collapsing cluster ~ DPM(alpha)
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//'
// [[Rcpp::export]]
Rcpp::List
asap_predict_mtx(const std::string mtx_file,
                 const Rcpp::NumericVector &memory_location,
                 const Eigen::MatrixXf &beta_dict,
                 const std::size_t mcem = 100,
                 const std::size_t burnin = 10,
                 const std::size_t thining = 3,
                 const double a0 = 1.,
                 const double b0 = 1.,
                 const std::size_t rseed = 42,
                 const bool verbose = false,
                 const bool do_collapse = true,
                 const bool do_beta_rescale = false,
                 const bool discrete_collapse = true,
                 const std::size_t collapsing_level = 100,
                 const double collapsing_dpm_alpha = 1.,
                 const std::size_t collapsing_mcmc = 200,
                 const std::size_t NUM_THREADS = 1,
                 const std::size_t BLOCK_SIZE = 100)
{
    CHK_RETL(convert_bgzip(mtx_file));
    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    const Index D = info.max_row;     // dimensionality
    const Index N = info.max_col;     // number of cells
    const Index K = beta_dict.cols(); // number of topics
    const Index block_size = BLOCK_SIZE;

    ASSERT_RETL(beta_dict.rows() == D,
                "incompatible Beta with the file: " << mtx_file);

    if (verbose) {
        TLOG("Dictionary Beta: " << D << " x " << K);
    }

    Mat C;
    std::vector<Scalar> collapsing_elbo;

    if (do_collapse) {

        using F = poisson_component_t<Mat>;
        using F0 = trunc_dpm_t<Mat>;
        auto L = std::min(collapsing_level, static_cast<std::size_t>(D));

        if (verbose) {
            TLOG("Collapsing Beta matrix... " << D << " -> " << L);
        }

        Mat beta_scaled = beta_dict;
        if (do_beta_rescale) {
            normalize_columns(beta_scaled);
            beta_scaled *= static_cast<Scalar>(beta_dict.rows());
        }

        clustering_status_t<Mat> status(beta_dict.rows(), beta_dict.cols(), L);
        clustering_by_lcvi<F, F0>(status,
                                  beta_scaled,
                                  L,
                                  collapsing_dpm_alpha,
                                  a0,
                                  b0,
                                  rseed,
                                  collapsing_mcmc,
                                  burnin,
                                  verbose);

        C.resize(L, beta_scaled.rows());

        if (discrete_collapse) {
            C.setZero();
            const Mat Z = status.latent.mean();
            for (Index r = 0; r < Z.rows(); ++r) {
                Index argmax;
                Z.row(r).maxCoeff(&argmax);
                C(argmax, r) = 1.;
            }
        } else {
            C = status.latent.mean().transpose();
        }

        collapsing_elbo.resize(status.elbo.size());
        std::copy(std::begin(status.elbo),
                  std::end(status.elbo),
                  std::begin(collapsing_elbo));
    }

    Mat onesD = Mat::Ones(D, 1);

    Mat theta(N, K);
    Mat theta_sd(N, K);
    Mat log_theta(N, K);
    Mat log_theta_sd(N, K);
    Vec Degree(N);

    if (verbose) {
        TLOG("Start recalibrating column-wise loading parameters...");
        TLOG("Theta: " << N << " x " << K);
        Rcpp::Rcerr << "Total " << N << std::flush;
    }

    Mat llik_mat(static_cast<Index>(std::ceil(N / block_size)),
                 static_cast<Index>(mcem + burnin));

    Index Nprocessed = 0;

    const Scalar TOL = 1e-8;

    // auto log_op = [&TOL](const Scalar &x) -> Scalar {
    //     if (x < TOL)
    //         return fasterlog(TOL);
    //     return fasterlog(x);
    // };

    Mat beta_scaled = beta_dict;
    if (do_beta_rescale) {
        normalize_columns(beta_scaled);
    }

    const Mat B = do_collapse ? (C * beta_scaled) : beta_scaled;
    const Vec S = B.transpose().colwise().sum();

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

        const Mat degree = Y.transpose().colwise().sum();

        using latent_t = latent_matrix_t<RNG>;
        latent_t aux(Y.rows(), Y.cols(), K, rng);
        discrete_sampler_t<Mat, RNG> proposal_by_row(rng, K);
        discrete_sampler_t<Mat, RNG> proposal_by_col(rng, K);
        gamma_t theta_b(Y.cols(), K, a0, b0, rng);

        for (std::size_t t = 0; t < (mcem + burnin); ++t) {

            /////////////////////////////
            // E-step: latent sampling //
            /////////////////////////////

            aux.sample_row_col(proposal_by_row.sample(B), theta_b.log_mean());

            ///////////////////////////////
            // M-step: update parameters //
            ///////////////////////////////

            for (Index k = 0; k < K; ++k) {
                theta_b.update_col(aux.slice_k(Y, k).transpose() * onesD,
                                   degree * S(k),
                                   k);
            }

            theta_b.calibrate();

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
            Degree(j) = degree(i);
        }
    }

    if (verbose) {
        Rcpp::Rcerr << std::endl;
    } else {
        Rcpp::Rcerr << std::endl;
    }

    TLOG("Done");

    return Rcpp::List::create(Rcpp::_["beta"] = beta_scaled,
                              Rcpp::_["theta"] = theta,
                              Rcpp::_["theta.sd"] = theta_sd,
                              Rcpp::_["log.theta"] = log_theta,
                              Rcpp::_["log.theta.sd"] = log_theta_sd,
                              Rcpp::_["log.likelihood"] = llik_mat,
                              Rcpp::_["sample.degree"] = Degree,
                              Rcpp::_["collapsing.factor"] = C,
			      Rcpp::_["beta.collapsed"] = B,
                              Rcpp::_["collapsing.elbo"] = collapsing_elbo);
}

//' Non-negative matrix factorization
//'
//' @param Y data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param mcem number of Monte Carl Expectation Maximization
//' @param burnin burn-in period
//' @param latent_iter latent sampling steps
//' @param degree_iter row and column degree optimization steps
//' @param thining thining interval in record keeping
//' @param verbose verbosity
//' @param eval_llik evaluate log-likelihood
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param rseed random seed
//' @param NUM_THREADS number of parallel jobs
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_nmf(const Eigen::MatrixXf &Y,
             const std::size_t maxK,
             const std::size_t mcem = 100,
             const std::size_t burnin = 10,
             const std::size_t latent_iter = 1,
             const std::size_t degree_iter = 1,
             const std::size_t thining = 3,
             const bool verbose = true,
             const bool eval_llik = true,
             const double a0 = 1.,
             const double b0 = 1.,
             const std::size_t rseed = 42,
             const std::size_t NUM_THREADS = 1)
{

    const Index D = Y.rows();
    const Index N = Y.cols();
    const Index K = maxK;

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    using gamma_t = gamma_param_t<Mat, RNG>;

    poisson_nmf_t<Mat, RNG, gamma_t> model(D, N, K, a0, b0, rseed);
    using latent_t = latent_matrix_t<RNG>;
    latent_t aux(D, N, K, rng);

    running_stat_t<Mat> row_stat(D, K);
    running_stat_t<Mat> row_log_stat(D, K);
    running_stat_t<Mat> column_stat(N, K);
    running_stat_t<Mat> column_log_stat(N, K);

    ////////////////////////////
    // Step 3. Monte Carlo EM //
    ////////////////////////////

    discrete_sampler_t<Mat, RNG> proposal_by_row(rng, K);
    discrete_sampler_t<Mat, RNG> proposal_by_col(rng, K);

    model.initialize_degree(Y);
    for (std::size_t s = 0; s < degree_iter; ++s) {
        model.update_degree(Y);
    }

    if (verbose)
        TLOG("Randomizing auxiliary/latent matrix");

    aux.randomize();

    model.update_column_topic(Y, aux);
    model.update_row_topic(Y, aux);

    if (verbose)
        TLOG("Initialized model parameters");

    Scalar llik;
    std::vector<Scalar> llik_trace;

    if (eval_llik) {
        llik = model.log_likelihood(Y, aux);
        llik_trace.emplace_back(llik);
        if (verbose)
            TLOG("Initial log-likelihood: " << llik);
    }

    for (std::size_t t = 0; t < (mcem + burnin); ++t) {
        for (std::size_t s = 0; s < latent_iter; ++s) {

            aux.sample_row_col(proposal_by_row.sample_logit(
                                   model.row_topic.log_mean()),
                               model.column_topic.log_mean(),
                               NUM_THREADS);

            aux.sample_col_row(proposal_by_col.sample_logit(
                                   model.column_topic.log_mean()),
                               model.row_topic.log_mean(),
                               NUM_THREADS);
        }

        model.update_column_topic(Y, aux);
        model.update_row_topic(Y, aux);

        if (eval_llik && t % thining == 0) {
            llik = model.log_likelihood(Y, aux);
            llik_trace.emplace_back(llik);
        }

        if (verbose && eval_llik) {
            TLOG("MCEM: " << t << " " << llik);
        } else {
            if (t % 10 == 0)
                Rcpp::Rcerr << ". " << std::flush;
            if (t > 0 && t % 100 == 0)
                Rcpp::Rcerr << std::endl;
        }

        if (t >= burnin && t % thining == 0) {
            row_stat(model.row_topic.mean());
            column_stat(model.column_topic.mean());
            row_log_stat(model.row_topic.log_mean());
            column_log_stat(model.column_topic.log_mean());
        }

        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            WLOG("Interruption by a user at t=" << t);
            break;
        }
    }

    Rcpp::Rcerr << std::endl;

    auto _summary = [](running_stat_t<Mat> &stat) {
        return Rcpp::List::create(Rcpp::_["mean"] = stat.mean(),
                                  Rcpp::_["sd"] = stat.sd());
    };

    Rcpp::List deg_out =
        Rcpp::List::create(Rcpp::_["row"] = model.row_degree.mean(),
                           Rcpp::_["column"] = model.column_degree.mean());

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["degree"] = deg_out,
                              Rcpp::_["row"] = _summary(row_stat),
                              Rcpp::_["column"] = _summary(column_stat),
                              Rcpp::_["log.row"] = _summary(row_log_stat),
                              Rcpp::_["log.column"] =
                                  _summary(column_log_stat));

    return Rcpp::List::create();
}

//' Generate approximate pseudo-bulk data by random projections
//'
//' @param mtx_file matrix-market-formatted data file (bgzip)
//' @param memory_location column indexing for the mtx
//' @param num_factors a desired number of random factors
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//'
// [[Rcpp::export]]
Rcpp::List
asap_random_bulk_data(const std::string mtx_file,
                      const Rcpp::NumericVector &memory_location,
                      const std::size_t num_factors,
                      const std::size_t rseed = 42,
                      const bool verbose = false,
                      const std::size_t NUM_THREADS = 1,
                      const std::size_t BLOCK_SIZE = 100)
{
    CHK_RETL(convert_bgzip(mtx_file));
    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    const Index D = info.max_row; // dimensionality
    const Index N = info.max_col; // number of cells
    const Index K = num_factors;  // tree depths in implicit bisection
    const Index block_size = BLOCK_SIZE;

    if (verbose) {
        TLOG(D << " x " << N << " single cell matrix");
    }

    /////////////////////////////////////////////
    // Step 1. sample random projection matrix //
    /////////////////////////////////////////////

    using norm_dist_t = boost::random::normal_distribution<Scalar>;
    dqrng::xoshiro256plus rng(rseed);
    norm_dist_t norm_dist(0., 1.);

    auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };
    Mat R = Mat::NullaryExpr(K, D, rnorm);

    if (verbose) {
        TLOG("Random projection: " << R.rows() << " x " << R.cols());
    }

    Mat Q(K, N);
    Q.setZero();

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);

        ///////////////////////////////////////
        // memory location = 0 means the end //
        ///////////////////////////////////////

        const Index lb_mem = memory_location[lb];
        const Index ub_mem = ub < N ? memory_location[ub] : 0;

        const SpMat xx =
            read_eigen_sparse_subset_col(mtx_file, lb, ub, lb_mem, ub_mem);

        const Mat temp = R * xx;
        for (Index i = 0; i < temp.cols(); ++i) {
            const Index j = i + lb;
            Q.col(j) += temp.col(i);
        }
    }

    if (verbose)
        TLOG("Finished random matrix projection");

    /////////////////////////////////////////////////
    // Step 2. Orthogonalize the projection matrix //
    /////////////////////////////////////////////////

    const std::size_t lu_iter = 5;      // this should be good
    RandomizedSVD<Mat> svd(K, lu_iter); //
    if (verbose)
        svd.set_verbose();

    normalize_columns(Q);
    svd.compute(Q);
    Mat random_dict = standardize(svd.matrixV()); // N x K

    if (verbose)
        TLOG("Finished SVD on the projected data");

    ////////////////////////////////////////////////
    // Step 3. sorting in an implicit binary tree //
    ////////////////////////////////////////////////

    IntVec bb(N);
    bb.setZero();

    is_positive_op<Vec> is_positive; // check x > 0

    for (Index k = 0; k < K; ++k) {
        auto binary_shift = [&k](const Scalar &x) -> Index {
            return x > 0. ? (1 << k) : 0;
        };
        bb += random_dict.col(k).unaryExpr(binary_shift);
    }

    const std::vector<Index> membership = std_vector(bb);

    std::unordered_map<Index, Index> pb_position;
    {
        Index pos = 0;
        for (Index k : membership) {
            if (pb_position.count(k) == 0)
                pb_position[k] = pos++;
        }
    }

    const Index S = pb_position.size();

    if (verbose) {
        TLOG("Identified " << S << " pseudo-bulk samples");
    }

    ///////////////////////////////////////
    // Step 4. create pseudoubulk matrix //
    ///////////////////////////////////////

    std::vector<Index> positions(membership.size());

    auto _pos_op = [&pb_position](const std::size_t x) {
        return pb_position.at(x);
    };
    std::transform(std::begin(membership),
                   std::end(membership),
                   std::begin(positions),
                   _pos_op);

    Mat PB(D, S);
    PB.setZero();

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);

        ///////////////////////////////////////
        // memory location = 0 means the end //
        ///////////////////////////////////////

        const Index lb_mem = memory_location[lb];
        const Index ub_mem = ub < N ? memory_location[ub] : 0;

        const SpMat xx =
            read_eigen_sparse_subset_col(mtx_file, lb, ub, lb_mem, ub_mem);

        const Mat x = xx;
        for (Index i = 0; i < (ub - lb); ++i) {
            const Index j = i + lb;
            const Index k = positions.at(j);
            PB.col(k) += x.col(i);
        }
    }

    if (verbose)
        TLOG("Finished populating the PB matrix: " << PB.rows() << " x "
                                                   << PB.cols());

    return Rcpp::List::create(Rcpp::_["PB"] = PB,
                              Rcpp::_["rand.dict"] = random_dict,
                              Rcpp::_["positions"] = positions,
                              Rcpp::_["rand.proj"] = R);
}
