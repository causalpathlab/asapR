#include "rcpp_asap.hh"

//' Non-negative matrix factorization
//'
//' @param Y data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param mcmc number of MCMC steps
//' @param verbose verbosity
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param rseed random seed
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_nmf(const Eigen::MatrixXf Y,
             const std::size_t maxK,
             const std::size_t mcem = 100,
             const std::size_t burnin = 10,
             const std::size_t latent_iter = 10,
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

    ///////////////////////////////
    // Step 1. Degree correction //
    ///////////////////////////////

    if (verbose)
        TLOG("Degree distributions...");

    for (Index t = 0; t < 5; ++t) {
        model.update_degree(Y);
    }

    ////////////////////////////
    // Step 2. Initialization //
    ////////////////////////////

    if (verbose)
        TLOG("Randomized auxiliary/latent matrix");

    aux.randomize();
    // std::cout << aux.Z << std::endl;
    model.update_column_topic(Y, aux);
    model.update_row_topic(Y, aux);

    running_stat_t<Mat> row_stat(D, K);
    running_stat_t<Mat> column_stat(N, K);

    ////////////////////////////
    // Step 3. Monte Carlo EM //
    ////////////////////////////

    discrete_sampler_t<Mat, RNG> proposal(rng, K);

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
            aux.sample_mh(proposal,
                          model.row_topic.log_mean(),
                          model.column_topic.log_mean(),
                          NUM_THREADS);
        }

        model.update_column_topic(Y, aux);
        model.update_row_topic(Y, aux);

        if (eval_llik && t % thining == 0) {
            llik = model.log_likelihood(Y, aux);
            llik_trace.emplace_back(llik);
        }

        if (verbose) {
            if (eval_llik) {
                TLOG("MCEM: " << t << " " << llik);
            } else {
                TLOG("MCEM: " << t);
            }
        } else {
            // TODO
        }

        if (t >= burnin && t % thining == 0) {

            if (verbose) {
                TLOG("Record keeping ...");
            }

            row_stat(model.row_topic.mean());
            column_stat(model.column_topic.mean());
        }

        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            TLOG("User abort at " << t);
            break;
        }
    }

    auto _summary = [](running_stat_t<Mat> &stat) {
        return Rcpp::List::create(Rcpp::_["mean"] = stat.mean(),
                                  Rcpp::_["sd"] = stat.sd());
    };

    Rcpp::List null_out =
        Rcpp::List::create(Rcpp::_["row"] = model.row_degree.mean(),
                           Rcpp::_["column"] = model.column_degree.mean());

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["null"] = null_out,
                              Rcpp::_["row"] = _summary(row_stat),
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
    Mat Y = standardize(svd.matrixV()); // N x K

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
        bb += Y.col(k).unaryExpr(binary_shift);
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
                              Rcpp::_["Y"] = Y,
                              Rcpp::_["positions"] = positions,
                              Rcpp::_["rand.proj"] = R);
}
