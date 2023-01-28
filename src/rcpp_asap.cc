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
asap_fit_nmf(const Eigen::MatrixXf &Y,
             const std::size_t maxK,
             const std::size_t mcmc = 100,
             const std::size_t burnin = 10,
             const bool verbose = true,
             const double a0 = 1.,
             const double b0 = 1.,
             const std::size_t rseed = 42)
{

    const Index D = Y.rows();
    const Index N = Y.cols();
    const Index K = maxK;

    Mat onesD(D, 1);
    onesD.setOnes();
    Mat onesN(N, 1);
    onesN.setOnes();

    using idx_vec = std::vector<Index>;
    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);

    ///////////////////////////////
    // Step 1. Degree correction //
    ///////////////////////////////
    using gamma_t = gamma_param_t<RNG>;
    gamma_t row_degree(D, 1, a0, b0, rng);
    gamma_t column_degree(N, 1, a0, b0, rng);

    auto update_degree = [&]() {
        column_degree.update(Y.transpose() * onesD,            //
                             onesN * row_degree.mean().sum()); //
        column_degree.calibrate();

        row_degree.update(Y * onesN,                           //
                          onesD * column_degree.mean().sum()); //
        row_degree.calibrate();
    };

    update_degree();

    ////////////////////////////
    // Step 2. Initialization //
    ////////////////////////////

    gamma_t row_param(D, K, a0, b0, rng);
    gamma_t column_param(N, K, a0, b0, rng);

    ////////////////////////////
    // Step 3. MC-EM or VB-EM //
    ////////////////////////////

    using latent_t = latent_matrix_t<RNG>;
    latent_t latent(D, N, rng, K);
    latent.randomize();

    // Update topic-specific sample patterns
    auto update_column_param = [&]() {
        for (Index k = 0; k < K; ++k) {
            const Scalar row_sum = row_param.mean().col(k).sum();
            column_param.update_col(latent.slice_k(Y, k).transpose() * onesD,
                                    column_degree.mean() * row_sum,
                                    k);
        }
        column_param.calibrate();
    };

    // Update topic-specific gene patterns
    auto update_row_param = [&]() {
        for (Index k = 0; k < K; ++k) {
            const Scalar column_sum = column_param.mean().col(k).sum();
            row_param.update_col(latent.slice_k(Y, k) * onesN,
                                 row_degree.mean() * column_sum,
                                 k);
        }
        row_param.calibrate();
    };

    update_column_param();
    update_row_param();

    auto average_log_likelihood = [&]() {
        constexpr Scalar tol = 1e-8;
        return (Y.cwiseProduct(
                    ((row_param.mean() * column_param.mean().transpose())
                         .array() +
                     tol)
                        .log()
                        .matrix()) -
                row_param.mean() * column_param.mean().transpose())
            .colwise()
            .sum()
            .mean();
    };

    Scalar llik = average_log_likelihood();
    std::vector<Scalar> llik_trace;
    llik_trace.emplace_back(llik);

    if (verbose)
        TLOG("Initial log-likelihood: " << llik);

    boost::random::uniform_01<Scalar> runif;
    discrete_sampler_t<RNG> proposal_sampler(rng, K);

    auto update_latent_mh = [&]() {
        //////////////////////////////
        // Make gene-based proposal //
        //////////////////////////////

        const idx_vec &prop = proposal_sampler(row_param.mean());

        ////////////////////////////////////////////////////
        // Take a Metropolis-Hastings step for each (i,j) //
        ////////////////////////////////////////////////////

        const Mat &log_beta = column_param.log_mean();
        constexpr Scalar zero = 0;

        for (Index j = 0; j < N; ++j) {
            for (Index i = 0; i < D; ++i) {
                const Index k_old = latent.coeff(i, j), k_new = prop.at(i);
                if (k_old != k_new) {
                    const Scalar l_new = log_beta.coeff(j, k_new);
                    const Scalar l_old = log_beta.coeff(j, k_old);
                    const Scalar log_mh_ratio = std::min(zero, l_new - l_old);
                    const Scalar u = runif(rng);
                    if (u <= 0 || fasterlog(u) < log_mh_ratio) {
                        latent.set(i, j, k_new);
                    }
                }
            }
        }
    };

    running_stat_t<Mat> row_stat(D, K);
    running_stat_t<Mat> column_stat(N, K);

    for (std::size_t t = 0; t < (mcmc + burnin); ++t) {
        update_latent_mh();
        update_column_param();
        update_row_param();
        llik = average_log_likelihood();
        llik_trace.emplace_back(llik);
        if (verbose) {
            TLOG("MCMC: " << t << " " << llik);
        } else {
            // TODO
        }
        if (t >= burnin) {
            row_stat(row_param.mean());
            column_stat(column_param.mean());
        }
    }

    // const Mat dd = column_degree.mean();
    // const Mat ff = row_degree.mean();

    auto _summary = [](running_stat_t<Mat> &stat) {
        return Rcpp::List::create(Rcpp::_["mean"] = stat.mean(),
                                  Rcpp::_["sd"] = stat.sd());
    };

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["row"] = _summary(row_stat),
                              Rcpp::_["column"] = _summary(column_stat));
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
