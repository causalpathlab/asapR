#include "rcpp_asap.hh"

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
