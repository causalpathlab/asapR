#include "rcpp_asap.hh"

//' Non-negative matrix factorization with the row modules
//'
//' @param Y data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param maxL maximum number of row modules
//' @param collapsing L x row collapsing matrix (L < gene)
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
asap_fit_modular_nmf(
    const Eigen::MatrixXf Y,
    const std::size_t maxK,
    const std::size_t maxL,
    Rcpp::Nullable<Rcpp::NumericMatrix> collapsing = R_NilValue,
    const std::size_t mcem = 100,
    const std::size_t burnin = 10,
    const std::size_t latent_iter = 10,
    const std::size_t degree_iter = 1,
    const std::size_t thining = 3,
    const bool verbose = true,
    const bool eval_llik = true,
    const double a0 = 1.,
    const double b0 = 1.,
    const std::size_t rseed = 42,
    const std::size_t NUM_THREADS = 1,
    const bool update_loading = true,
    const bool gibbs_sampling = false)
{

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    using gamma_t = gamma_param_t<Mat, RNG>;

    const Index D = Y.rows();
    const Index N = Y.cols();
    const Index K = std::min(static_cast<Index>(maxK), N);
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;
    const ColVec y_sum = Y.rowwise().sum(); // D x 1

    Mat C_DL;
    if (collapsing.isNotNull()) {
        C_DL = Rcpp::as<Mat>(Rcpp::NumericMatrix(collapsing));
        ASSERT_RETL(C_DL.rows() == D, "incompatible collapsing matrix");
    }

    const Index L = collapsing.isNotNull() ? C_DL.cols() : maxL;

    Mat lnG_NL(N, L), tempNL(N, L), logP_DL(D, L);
    logP_DL.setZero();

    matrix_sampler_t<Mat, RNG> row_clust_sampler(rng, L);
    using Idx = matrix_sampler_t<Mat, RNG>::IndexVec;
    if (collapsing.isNull()) {
        C_DL.resize(D, L);
        C_DL.setZero();
        const Idx &sampled = row_clust_sampler.sample_logit(logP_DL);
        for (Index i = 0; i < D; ++i) {
            const Index k = sampled.at(i);
            C_DL(i, k) = 1.;
        }
    }

    poisson_modular_nmf_t<Mat, RNG, gamma_t> model(D, N, K, L, a0, b0, rseed);
    using latent_t = latent_matrix_t<RNG>;
    latent_t Z_NL(N, L, K, rng);

    //////////////////////////
    // degree distributions //
    //////////////////////////

    model.initialize_degree(Y);

    for (std::size_t s = 0; s < degree_iter; ++s) {
        model.update_degree(Y);
    }

    if (verbose)
        TLOG("Randomizing auxiliary/latent matrix");

    Z_NL.randomize();
    model.initialize_by_svd(Y);

    if (verbose)
        TLOG("Initialized model parameters");

    Scalar llik;
    std::vector<Scalar> llik_trace;

    if (eval_llik) {
        llik = model.log_likelihood(Y, C_DL, Z_NL);
        llik_trace.emplace_back(llik);
        if (verbose)
            TLOG("Initial log-likelihood: " << llik);
    }

    running_stat_t<Mat> dict_stat(D, K);
    running_stat_t<Mat> row_stat(D, L);
    running_stat_t<Mat> C_stat(D, L);
    running_stat_t<Mat> middle_stat(L, K);
    running_stat_t<Mat> column_stat(N, K);

    Mat logP_LK(L, K);

    matrix_sampler_t<Mat, RNG> col_proposal(rng, K);

    for (std::size_t t = 0; t < (mcem + burnin); ++t) {

        //////////////////////////////
        // Update the middle factor //
        //////////////////////////////

        logP_LK = model.middle_LK.log_mean().array().rowwise() +
            model.get_loading_logK().transpose().array();

        for (Index s = 0; s < latent_iter; ++s) {

            if (gibbs_sampling) {
                Z_NL.gibbs_sample_row_col(model.column_NK.log_mean(),
                                          logP_LK,
                                          NUM_THREADS);
            } else {

                Z_NL.mh_sample_col_row(col_proposal.sample_logit(logP_LK),
                                       model.column_NK.log_mean(),
                                       NUM_THREADS);
            }
        }

        if (update_loading) {
            model.update_loading_K(Y, C_DL, Z_NL);
        }
        model.update_middle_topic(Y, C_DL, Z_NL);

        ///////////////////////////////
        // update the column factors //
        ///////////////////////////////

        logP_LK = model.middle_LK.log_mean().array().rowwise() +
            model.get_loading_logK().transpose().array();

        for (Index s = 0; s < latent_iter; ++s) {

            if (gibbs_sampling) {
                Z_NL.gibbs_sample_row_col(model.column_NK.log_mean(),
                                          logP_LK,
                                          NUM_THREADS);
            } else {

                Z_NL.mh_sample_col_row(col_proposal.sample_logit(logP_LK),
                                       model.column_NK.log_mean(),
                                       NUM_THREADS);
            }
        }

        if (update_loading) {
            model.update_loading_K(Y, C_DL, Z_NL);
        }

        model.update_column_topic(Y, C_DL, Z_NL);

        /////////////////////////////////////
        // latent variables for clustering //
        /////////////////////////////////////

        lnG_NL.setZero();

        for (Index k = 0; k < K; ++k) {
            tempNL.setZero();
            tempNL.colwise() += model.column_NK.log_mean().col(k);
            tempNL.rowwise() += model.middle_LK.log_mean().col(k).transpose();
            lnG_NL += Z_NL.slice_k(tempNL, k);
        }

        logP_DL = (Y * lnG_NL).array().colwise() / y_sum.array();

        {
            C_DL.setZero();
            const Idx &sampled = row_clust_sampler.sample_logit(logP_DL);
            for (Index i = 0; i < D; ++i) {
                C_DL(i, sampled.at(i)) = 1.;
            }
        }

        model.update_row_topic(Y, C_DL, Z_NL);

        if (eval_llik) {
            llik = model.log_likelihood(Y, C_DL, Z_NL);
            llik_trace.emplace_back(llik);
        }

        if (verbose && eval_llik) {
            TLOG("modNMF MCEM: " << t << " " << llik);
        } else {
            if (t > 0 && t % 10 == 0)
                Rcpp::Rcerr << ". " << std::flush;
            if (t > 0 && t % 100 == 0)
                Rcpp::Rcerr << std::endl;
        }

        if (t >= burnin && t % thining == 0) {
            C_stat(C_DL);
            dict_stat(model.row_DL.mean() * model.middle_LK.mean());
            row_stat(model.row_DL.mean());
            middle_stat(model.middle_LK.mean());
            column_stat(model.column_NK.mean());
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
        Rcpp::List::create(Rcpp::_["row"] = model.row_D.mean(),
                           Rcpp::_["column"] = model.column_N.mean());

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["degree"] = deg_out,
                              Rcpp::_["row.clust"] = _summary(C_stat),
                              Rcpp::_["dict"] = _summary(dict_stat),
                              Rcpp::_["row"] = _summary(row_stat),
                              Rcpp::_["middle"] = _summary(middle_stat),
                              Rcpp::_["column"] = _summary(column_stat),
                              Rcpp::_["C"] = C_DL);
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
asap_fit_nmf(const Eigen::MatrixXf Y,
             const std::size_t maxK,
             const std::size_t mcem = 100,
             const std::size_t burnin = 10,
             const std::size_t latent_iter = 10,
             const std::size_t degree_iter = 1,
             const std::size_t thining = 3,
             const bool verbose = true,
             const bool eval_llik = true,
             const double a0 = 1.,
             const double b0 = 1.,
             const std::size_t rseed = 42,
             const std::size_t NUM_THREADS = 1,
             const bool update_loading = true,
             const bool gibbs_sampling = false)
{

    const Index D = Y.rows();
    const Index N = Y.cols();
    const Index K = std::min(static_cast<Index>(maxK), N);

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    using gamma_t = gamma_param_t<Mat, RNG>;

    poisson_nmf_t<Mat, RNG, gamma_t> model(D, N, K, a0, b0, rseed);
    using latent_t = latent_matrix_t<RNG>;
    latent_t aux(D, N, K, rng);

    running_stat_t<Mat> dict_stat(D, K);
    running_stat_t<Mat> loading_stat(K, 1);
    running_stat_t<Mat> column_stat(N, K);

    model.initialize_degree(Y);
    for (std::size_t s = 0; s < degree_iter; ++s) {
        model.update_degree(Y);
    }

    if (verbose)
        TLOG("Randomizing auxiliary/latent matrix");

    aux.randomize();
    model.initialize_by_svd(Y);

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

    matrix_sampler_t<Mat, RNG> row_proposal(rng, K);

    Mat logP_DK(D, K);

    for (std::size_t t = 0; t < (mcem + burnin); ++t) {

        /////////////////////////////////
        // sampling to update the rows //
        /////////////////////////////////

        logP_DK = model.row_topic.log_mean().array().rowwise() +
            model.take_topic_log_loading().transpose().array();

        for (std::size_t s = 0; s < latent_iter; ++s) {

            if (gibbs_sampling) {
                aux.gibbs_sample_row_col(logP_DK,
                                         model.column_topic.log_mean(),
                                         NUM_THREADS);
            } else {
                aux.mh_sample_row_col(row_proposal.sample_logit(logP_DK),
                                      model.column_topic.log_mean(),
                                      NUM_THREADS);
            }
        }

        if (update_loading) {
            model.update_topic_loading(Y, aux);
        }
        model.update_row_topic(Y, aux);

        ////////////////////////////////
        // sampling to update columns //
        ////////////////////////////////

        logP_DK = model.row_topic.log_mean().array().rowwise() +
            model.take_topic_log_loading().transpose().array();

        for (std::size_t s = 0; s < latent_iter; ++s) {

            if (gibbs_sampling) {
                aux.gibbs_sample_row_col(logP_DK,
                                         model.column_topic.log_mean(),
                                         NUM_THREADS);
            } else {
                aux.mh_sample_row_col(row_proposal.sample_logit(logP_DK),
                                      model.column_topic.log_mean(),
                                      NUM_THREADS);
            }
        }

        if (update_loading) {
            model.update_topic_loading(Y, aux);
        }

        model.update_column_topic(Y, aux);

        if (eval_llik && t % thining == 0) {
            llik = model.log_likelihood(Y, aux);
            llik_trace.emplace_back(llik);
        }

        if (verbose && eval_llik) {
            TLOG("NMF MCEM: " << t << " " << llik);
        } else {
            if (t > 0 && t % 10 == 0)
                Rcpp::Rcerr << ". " << std::flush;
            if (t > 0 && t % 100 == 0)
                Rcpp::Rcerr << std::endl;
        }

        if (t >= burnin && t % thining == 0) {
            loading_stat(model.topic_loading.mean());
            dict_stat(model.row_topic.mean());
            column_stat(model.column_topic.mean());
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
                              Rcpp::_["dict"] = _summary(dict_stat),
                              Rcpp::_["loading"] = _summary(loading_stat),
                              Rcpp::_["column"] = _summary(column_stat));

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
                 const bool do_beta_rescale = false,
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
    const Scalar TOL = 1e-8;

    ASSERT_RETL(beta_dict.rows() == D,
                "incompatible Beta with the file: " << mtx_file);

    if (verbose) {
        TLOG("Dictionary Beta: " << D << " x " << K);
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
    Vec Degree(N);

    if (verbose) {
        TLOG("Start recalibrating column-wise loading parameters...");
        TLOG("Theta: " << N << " x " << K);
        Rcpp::Rcerr << "Total " << N << std::flush;
    }

    Mat llik_mat(static_cast<Index>(std::ceil(N / block_size)),
                 static_cast<Index>(mcem + burnin));

    Index Nprocessed = 0;

    Mat beta_scaled = beta_dict;
    if (do_beta_rescale) {
        normalize_columns(beta_scaled);
    }

    const Mat B = do_collapse ? (C * beta_scaled) : beta_scaled;
    const Vec S = B.transpose().colwise().sum();

    auto log_op = [&TOL](const Scalar &x) -> Scalar {
        if (x < TOL)
            return fasterlog(TOL);
        return fasterlog(x);
    };

    const Mat log_B = B.unaryExpr(log_op);

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
        gamma_t theta_b(Y.cols(), K, a0, b0, rng);

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
                              Rcpp::_["beta.collapsed"] = B);
}
