#include "rcpp_asap.hh"

//' Generate approximate pseudo-bulk data by random projections
//'
//' @param mtx_file matrix-market-formatted data file (bgzip)
//' @param memory_location column indexing for the mtx
//' @param num_factors a desired number of random factors
//' @param r_covar covariates (default: NULL)
//' @param r_batch batch information (default: NULL)
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_normalize normalize each column after random projection
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_row_std rowwise standardization (default: FALSE)
//' @param KNN_CELL k-NN matching between cells (default: 10)
//' @param KNN_BILINK num. of bidirectional links (default: 10)
//' @param KNN_NNLIST num. of nearest neighbor lists (default: 10)
//'
// [[Rcpp::export]]
Rcpp::List
asap_random_bulk_data(
    const std::string mtx_file,
    const Rcpp::NumericVector &memory_location,
    const std::size_t num_factors,
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_covar = R_NilValue,
    const Rcpp::Nullable<Rcpp::StringVector> r_batch = R_NilValue,
    const std::size_t rseed = 42,
    const bool verbose = false,
    const std::size_t NUM_THREADS = 1,
    const std::size_t BLOCK_SIZE = 100,
    const bool do_normalize = false,
    const bool do_log1p = false,
    const bool do_row_std = false,
    const std::size_t KNN_CELL = 10,
    const std::size_t KNN_BILINK = 10,
    const std::size_t KNN_NNLIST = 10)
{

    log1p_op<Mat> log1p;
    at_least_one_op<Mat> at_least_one;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    CHK_RETL(convert_bgzip(mtx_file));
    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    const Index D = info.max_row; // dimensionality
    const Index N = info.max_col; // number of cells
    const Index K = num_factors;  // tree depths in implicit bisection
    const Index block_size = BLOCK_SIZE;
    const Mat X = r_covar.isNotNull() ?
        Rcpp::as<Mat>(Rcpp::NumericMatrix(r_covar)) :
        Mat::Ones(N, 1);

    ASSERT_RETL(X.rows() == N, "incompatible covariate matrix");

    if (verbose) {
        TLOG(D << " x " << N << " single cell matrix");
    }

    std::vector<Index> mtx_idx;
    std::copy(std::begin(memory_location),
              std::end(memory_location),
              std::back_inserter(mtx_idx));

    using data_t = mmutil::match::data_loader_t;
    data_t matched_data(mtx_file,
                        mtx_idx,
                        mmutil::match::KNN(KNN_CELL),
                        mmutil::match::BILINK(KNN_BILINK),
                        mmutil::match::NNLIST(KNN_NNLIST));

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

    mmutil::stat::row_collector_t collector(do_log1p);

    ColVec mu(D), sig(D);

    mu.setZero();
    sig.setOnes();

    if (do_row_std) {

        ColVec s1(D), s2(D);
        s1.setZero();
        s2.setZero();

        if (verbose)
            TLOG("Collecting row-wise statistics...");

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index lb = 0; lb < N; lb += block_size) {

            const Index ub = std::min(N, block_size + lb);

            ///////////////////////////////////////
            // memory location = 0 means the end //
            ///////////////////////////////////////

            const Index lb_mem = lb < N ? mtx_idx[lb] : 0;
            const Index ub_mem = ub < N ? mtx_idx[ub] : 0;

            mmutil::stat::row_collector_t collector(do_log1p);
            collector.set_size(info.max_row, info.max_col, info.max_elem);

            CHECK(visit_bgzf_block(mtx_file, lb_mem, ub_mem, collector));

            s1 += collector.Row_S1;
            s2 += collector.Row_S2;
        }

        const Scalar eps = 1e-8;
        safe_sqrt_op<Mat> safe_sqrt;
        const Scalar nn = static_cast<Scalar>(N);

        mu = s1 / nn;
        sig = (s2 / nn - mu.cwiseProduct(mu)).unaryExpr(safe_sqrt);
        sig.array() += eps;
    }

    if (verbose)
        TLOG("Collecting random projection data");
#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);

        Mat yy = do_log1p ? matched_data.read(lb, ub).unaryExpr(log1p) :
                            matched_data.read(lb, ub);

        Mat temp;
        if (do_row_std) {
            temp = R *
                ((yy.array().colwise() - mu.array()) / sig.array()).matrix();
        } else {
            temp = R * yy;
        }

        for (Index i = 0; i < temp.cols(); ++i) {
            const Index j = i + lb;
            Q.col(j) += temp.col(i);
        }
    }

    if (verbose)
        TLOG("Finished random matrix projection");

    // Regress out
    if (X.cols() < 2) {
        Mat Qt = Q.transpose();
        ColVec denom = (X.cwiseProduct(X))
                           .colwise()
                           .sum()
                           .unaryExpr(at_least_one)
                           .transpose();
        Mat B = (X.transpose() * Qt).array().colwise() / denom.array();
        Q = (Qt - X * B).transpose();
    } else {
        Mat Qt = Q.transpose();
        const std::size_t r = std::min(X.cols(), Qt.cols());
        RandomizedSVD<Mat> svd_x(r, 5);
        svd_x.compute(X);
        const ColVec d = svd_x.singularValues();
        Mat u = svd_x.matrixU();
        const Scalar eps = 1e-8;
        for (Index k = 0; k < r; ++k) {
            if (d(k) < eps)
                u.col(k).setZero();
        }
        // X theta = X inv(X'X) X' Y
        //         = U D V' V inv(D^2) V' (U D V')' Y
        //         = U inv(D) V' V D U' Y
        //         = U U' Y
        Q = (Qt - u * u.transpose() * Qt).transpose();
    }
    if (verbose)
        TLOG("Regress out known covariates");

    /////////////////////////////////////////////////
    // Step 2. Orthogonalize the projection matrix //
    /////////////////////////////////////////////////

    const std::size_t lu_iter = 5;      // this should be good
    RandomizedSVD<Mat> svd(K, lu_iter); //
    // if (verbose) svd.set_verbose();

    if (do_normalize)
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

    // Pseudobulk samples to cells
    auto pb_cells = make_index_vec_vec(positions);

    Mat mu_ds = Mat::Zero(D, S);
    Mat Qpb = Mat::Zero(K, S);

    ////////////////////////////
    // Read batch information //
    ////////////////////////////

    if (r_batch.isNotNull()) {
        std::vector<std::string> batch =
            rcpp::util::copy(Rcpp::StringVector(r_batch));
        matched_data.set_exposure_info(batch);
    }

    const Index B = matched_data.num_exposure();

    Mat delta_db;

    if (B > 1) {

        delta_db.resize(D, B); // gene x batch
        delta_db.setOnes();

        if (verbose) {
            TLOG("Pseudobulk estimation while "
                 << "accounting for batch effects");
        }

        Mat delta_num_db = Mat::Zero(D, B);   // gene x batch numerator
        Mat delta_denom_db = Mat::Zero(D, B); // gene x batch denominator
        Mat ybar_ds = Mat::Zero(D, S);        // gene x PB mean
        Mat zbar_ds = Mat::Zero(D, S);        // gene x PB counterfactual
        Mat prob_bs = Mat::Zero(B, S);        // batch x PB prob
        Mat n_bs = Mat::Zero(B, S);           // batch x PB freq

        matched_data.build_dictionary(Q, NUM_THREADS);

        ////////////////////////////
        // Step a. precalculation //
        ////////////////////////////

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index s = 0; s < S; ++s) {

            if (pb_cells.at(s).size() < 1)
                continue;

            const Index ns = pb_cells.at(s).size();
            const Scalar ns_ = static_cast<Scalar>(ns);

            const Mat yy = matched_data.read(pb_cells.at(s));
            const Mat zz = matched_data.read_counterfactual(pb_cells.at(s));

            ybar_ds.col(s) = yy.rowwise().mean();
            zbar_ds.col(s) = zz.rowwise().mean();

            prob_bs.col(s).setZero();
            for (Index j = 0; j < pb_cells.at(s).size(); ++j) {
                const Index b = matched_data.exposure_group(j);
                prob_bs(b, s) = prob_bs(b, s) + 1. / ns_;
                n_bs(b, s) = n_bs(b, s) + 1.;
                delta_num_db.col(b) += yy.col(j);
            }
        }

        ///////////////////////////////
        // Step b. Iterative updates //
        ///////////////////////////////

        const Scalar tol = 1e-8;
        const Index max_iter = 100;

        for (Index t = 0; t < max_iter; ++t) {
            mu_ds = (ybar_ds + zbar_ds).array() /
                ((delta_db * prob_bs).array() + 1.);

            delta_denom_db = mu_ds * n_bs.transpose();
            delta_db = delta_num_db.array() / (delta_denom_db.array() + tol);
        }
    } else {

        //////////////////////////////////////////////////
        // Pseudobulk without considering batch effects //
        //////////////////////////////////////////////////

        if (verbose)
            TLOG("Pseudobulk estimation in a vanilla mode");

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index lb = 0; lb < N; lb += block_size) {
            const Index ub = std::min(N, block_size + lb);
            Mat y = matched_data.read(lb, ub);
            for (Index i = 0; i < (ub - lb); ++i) {
                const Index j = i + lb;
                const Index s = positions.at(j);
                mu_ds.col(s) += y.col(i);
                Qpb.col(s) += Q.col(j);
            }
        }
    }

    // convert zero-based to 1-based
    std::transform(std::begin(positions),
                   std::end(positions),
                   std::begin(positions),
                   [](Index x) -> Index { return (x + 1); });

    if (do_normalize)
        normalize_columns(Qpb);

    if (verbose)
        TLOG("Finished populating the PB matrix: " << mu_ds.rows() << " x "
                                                   << mu_ds.cols());

    return Rcpp::List::create(Rcpp::_["PB"] = mu_ds,
                              Rcpp::_["Q"] = Q,
                              Rcpp::_["Q.pb"] = Qpb,
                              Rcpp::_[""] = delta_db,
                              Rcpp::_["rand.dict"] = random_dict,
                              Rcpp::_["svd.u"] = svd.matrixU(),
                              Rcpp::_["svd.d"] = svd.singularValues(),
                              Rcpp::_["svd.v"] = svd.matrixV(),
                              Rcpp::_["positions"] = positions,
                              Rcpp::_["rand.proj"] = R);
}
