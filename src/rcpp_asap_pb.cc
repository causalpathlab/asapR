#include "rcpp_asap.hh"
#include "rcpp_asap_stat.hh"

//' Generate approximate pseudo-bulk data by random projections
//'
//' @param mtx_file matrix-market-formatted data file (bgzip)
//' @param mtx_idx_file matrix-market colum index file
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
//' @param BATCH_ADJ_ITER batch Adjustment steps (default: 100)
//' @param a0 gamma(a0, b0) (default: 1)
//' @param b0 gamma(a0, b0) (default: 1)
//'
// [[Rcpp::export]]
Rcpp::List
asap_random_bulk_data(
    const std::string mtx_file,
    const std::string mtx_idx_file,
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
    const std::size_t BATCH_ADJ_ITER = 100,
    const double a0 = 1,
    const double b0 = 1)
{

    log1p_op<Mat> log1p;
    at_least_one_op<Mat> at_least_one;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    CHK_RETL(convert_bgzip(mtx_file));
    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    std::vector<Index> mtx_idx;
    CHK_RETL_(read_mmutil_index(mtx_idx_file, mtx_idx),
              "Failed to read the index file:" << std::endl
                                               << mtx_idx_file << std::endl
                                               << "Consider rebuilding it."
                                               << std::endl);

    const Index D = info.max_row; // dimensionality
    const Index N = info.max_col; // number of cells
    const Index K = num_factors;  // tree depths in implicit bisection
    const Index block_size = BLOCK_SIZE;
    const Mat X = r_covar.isNotNull() ?
        Rcpp::as<Mat>(Rcpp::NumericMatrix(r_covar)) :
        Mat::Ones(N, 1);

    const Scalar eps = 1e-8;
    ASSERT_RETL(X.rows() == N, "incompatible covariate matrix");

    if (verbose) {
        TLOG(D << " x " << N << " single cell matrix");
    }

    using data_t = mmutil::match::data_loader_t;
    data_t matched_data(mtx_file, mtx_idx, mmutil::match::KNN(KNN_CELL));

    /////////////////////////////////////////////
    // Step 1. sample random projection matrix //
    /////////////////////////////////////////////

    using norm_dist_t = boost::random::normal_distribution<Scalar>;
    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    norm_dist_t norm_dist(0., 1.);

    auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };
    Mat R = Mat::NullaryExpr(K, D, rnorm);

    if (verbose) {
        TLOG("Random projection: " << R.rows() << " x " << R.cols());
    }

    Mat Q = Mat::Zero(K, N);
    ColVec mu = ColVec::Zero(D), sig = ColVec::Ones(D);

    if (do_row_std) {

        std::tie(mu, sig) = compute_row_stat(mtx_file,
                                             mtx_idx,
                                             block_size,
                                             do_log1p,
                                             NUM_THREADS,
                                             verbose);
    }

    if (verbose) {
        TLOG("Collecting random projection data");
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);

        Mat yy = do_log1p ? matched_data.read(lb, ub).unaryExpr(log1p) :
                            matched_data.read(lb, ub);

        Mat temp(K, yy.cols());

        if (do_row_std) {
            temp = R *
                ((yy.array().colwise() - mu.array()) / sig.array()).matrix();
        } else {
            temp = R * yy;
        }

        for (Index i = 0; i < temp.cols(); ++i) {
            const Index j = i + lb;
            Q.col(j) = temp.col(i);
        }
    }

    if (verbose) {
        TLOG("Finished random matrix projection");
    }

    // Regress out
    if (X.cols() > 0) {
        Mat Qt = Q.transpose(); // N x K
        residual_columns(Qt, X);
        Q = Qt.transpose();
    }

    if (verbose) {
        TLOG("Regressed out Q: " << Q.rows() << " x " << Q.cols());
    }

    /////////////////////////////////////////////////
    // Step 2. Orthogonalize the projection matrix //
    /////////////////////////////////////////////////

    if (do_normalize) {
        normalize_columns(Q);
    }

    Eigen::BDCSVD<Mat> svd;
    svd.compute(Q, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Mat vv = svd.matrixV();
    ASSERT_RETL(vv.rows() == N, " failed SVD for Q");

    Mat random_dict = standardize(vv); // N x K
    TLOG(random_dict.rows() << " x " << random_dict.cols());

    if (verbose) {
        TLOG("SVD on the projected: " << random_dict.rows() << " x "
                                      << random_dict.cols());
    }

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

    TLOG("Assigned random membership: [0, " << bb.maxCoeff() << ")");

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

    // convert zero-based to 1-based for R
    std::vector<Index> r_positions(positions.size());
    std::transform(std::begin(positions),
                   std::end(positions),
                   std::begin(r_positions),
                   [](Index x) -> Index { return (x + 1); });

    // Pseudobulk samples to cells
    const std::vector<std::vector<Index>> pb_cells =
        make_index_vec_vec(positions);

    if (verbose) {
        TLOG("Start collecting statistics... "
             << " for " << pb_cells.size() << " samples");
    }

    Mat mu_ds = Mat::Ones(D, S);
    Mat log_mu_ds = Mat::Ones(D, S);
    Mat ysum_ds = Mat::Zero(D, S);
    RowVec size_s = RowVec::Zero(S);

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
            ysum_ds.col(s) += y.col(i);
            size_s(s) += 1.;
        }
    }

    ////////////////////////////
    // Read batch information //
    ////////////////////////////

    if (r_batch.isNotNull()) {
        std::vector<std::string> batch =
            rcpp::util::copy(Rcpp::StringVector(r_batch));
        matched_data.set_exposure_info(batch);
    }

    const Index B = matched_data.num_exposure();

    Mat delta_db, delta_sd_db, log_delta_db, log_delta_sd_db, delta_ds;
    Mat prob_bs, n_bs;

    if (B > 1) {

        delta_db.resize(D, B); // gene x batch
        delta_db.setOnes();

        if (verbose) {
            TLOG("Random pseudo-bulk estimation while "
                 << "accounting for " << B << " batch effects");
        }

        Mat delta_num_db = Mat::Zero(D, B);   // gene x batch numerator
        Mat delta_denom_db = Mat::Zero(D, B); // gene x batch denominator

        prob_bs = Mat::Zero(B, S); // batch x PB prob
        n_bs = Mat::Zero(B, S);    // batch x PB freq

        CHK_RETL_(matched_data.build_annoy_index(Q),
                  "Failed to build Annoy Indexes: " << Q.rows() << " x "
                                                    << Q.cols());

        ////////////////////////////
        // Step a. precalculation //
        ////////////////////////////

        if (verbose) {
            TLOG("Start collecting sufficient statistics");
        }

        Mat zsum_ds = Mat::Zero(D, S); // gene x PB mean

        Index Nprocessed = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index s = 0; s < pb_cells.size(); ++s) {

            const std::vector<Index> &_cells_s = pb_cells.at(s);

            const Mat yy = matched_data.read(_cells_s);
            const Mat zz = matched_data.read_counterfactual(_cells_s);

            prob_bs.col(s).setZero();
            n_bs.col(s).setZero();

            zsum_ds.col(s) += zz.rowwise().sum();
            for (Index j = 0; j < _cells_s.size(); ++j) {
                const Index k = _cells_s.at(j);
                const Index b = matched_data.exposure_group(k);
                n_bs(b, s) = n_bs(b, s) + 1.;
                delta_num_db.col(b) += yy.col(j);
            }
            prob_bs.col(s) = n_bs.col(s) / n_bs.col(s).sum();

#pragma omp critical
            {
                Nprocessed += 1;
                if (verbose) {
                    Rcpp::Rcerr << "\rProcessed: " << Nprocessed << std::flush;
                } else {
                    Rcpp::Rcerr << "+ " << std::flush;
                    if (Nprocessed % 100 == 0)
                        Rcpp::Rcerr << "\r" << std::flush;
                }
            }
        }

        Rcpp::Rcerr << std::endl;

        if (verbose) {
            TLOG("Collected sufficient statistics");
        }

        gamma_param_t<Mat, RNG> delta_param(D, B, a0, b0, rng);
        gamma_param_t<Mat, RNG> mu_param(D, S, a0, b0, rng);
        gamma_param_t<Mat, RNG> gamma_param(D, S, a0, b0, rng);

        ///////////////////////////////
        // Step b. Iterative updates //
        ///////////////////////////////

        Mat gamma_ds = Mat::Ones(D, S); // bias on the side of CF

        for (std::size_t t = 0; t < BATCH_ADJ_ITER; ++t) {
            ////////////////////////
            // shared components  //
            ////////////////////////
            mu_param.update(ysum_ds + zsum_ds,
                            delta_db * n_bs +
                                ((gamma_ds.array().rowwise() * size_s.array()))
                                    .matrix());
            mu_param.calibrate();
            mu_ds = mu_param.mean();

            // mu_ds = (ybar_ds + zbar_ds).array() /
            //     ((delta_db * prob_bs).array() + gamma_ds.array() + eps);

            ////////////////////
            // residual for z //
            ////////////////////

            gamma_param
                .update(zsum_ds,
                        (mu_ds.array().rowwise() * size_s.array()).matrix());
            gamma_param.calibrate();
            gamma_ds = gamma_param.mean();

            // gamma_ds = zbar_ds.array() / (mu_ds.array() + eps);

            ///////////////////////////////
            // batch-specific components //
            ///////////////////////////////
            delta_denom_db = mu_ds * n_bs.transpose();
            delta_param.update(delta_num_db, delta_denom_db);
            delta_param.calibrate();
            delta_db = delta_param.mean();
            delta_sd_db = delta_param.sd();
            log_delta_db = delta_param.log_mean();
            log_delta_sd_db = delta_param.log_sd();
        }

        delta_ds = delta_db * prob_bs;

        mu_param.update(ysum_ds, delta_db * n_bs);
        mu_param.calibrate();
        mu_ds = mu_param.mean();
        log_mu_ds = mu_param.log_mean();

    } else {

        //////////////////////////////////////////////////
        // Pseudobulk without considering batch effects //
        //////////////////////////////////////////////////

        if (verbose)
            TLOG("Pseudobulk estimation in a vanilla mode");

        mu_ds = ysum_ds.array().rowwise() / size_s.array();
    }

    if (verbose)
        TLOG("Finished populating the PB matrix: " << mu_ds.rows() << " x "
                                                   << mu_ds.cols());

    return Rcpp::List::create(Rcpp::_["PB"] = mu_ds,
                              Rcpp::_["PB.batch"] = delta_ds,
                              Rcpp::_["log.PB"] = log_mu_ds,
                              Rcpp::_["sum"] = ysum_ds,
                              Rcpp::_["size"] = size_s,
                              Rcpp::_["prob.batch.sample"] = prob_bs,
                              Rcpp::_["size.batch.sample"] = n_bs,
                              Rcpp::_["batch.effect"] = delta_db,
                              Rcpp::_["batch.sd"] = delta_sd_db,
                              Rcpp::_["log.batch.effect"] = log_delta_db,
                              Rcpp::_["log.batch.sd"] = log_delta_sd_db,
                              Rcpp::_["batch.membership"] =
                                  matched_data.get_exposure_mapping(),
                              Rcpp::_["batch.names"] =
                                  matched_data.get_exposure_names(),
                              Rcpp::_["positions"] = r_positions,
                              Rcpp::_["rand.proj"] = R,
                              Rcpp::_["Q"] = Q,
                              Rcpp::_["rand.dict"] = random_dict,
                              Rcpp::_["svd.u"] = svd.matrixU(),
                              Rcpp::_["svd.d"] = svd.singularValues(),
                              Rcpp::_["svd.v"] = svd.matrixV());
}
