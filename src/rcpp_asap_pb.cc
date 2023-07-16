#include "rcpp_asap.hh"
#include "rcpp_asap_stat.hh"

//' Generate approximate pseudo-bulk data by random projections
//'
//' @param mtx_file matrix-market-formatted data file (bgzip)
//' @param row_file row names (gene/feature names)
//' @param col_file column names (cell/column names)
//' @param idx_file matrix-market colum index file
//' @param num_factors a desired number of random factors
//' @param r_covar_n N x r covariates (default: NULL)
//' @param r_covar_d D x r covariates (default: NULL)
//' @param r_batch batch information (default: NULL)
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_down_sample down-sampling (default: FALSE)
//' @param save_rand_proj save random projection (default: FALSE)
//' @param KNN_CELL k-NN matching between cells (default: 10)
//' @param CELL_PER_SAMPLE down-sampling cell per sample (default: 100)
//' @param BATCH_ADJ_ITER batch Adjustment steps (default: 100)
//' @param a0 gamma(a0, b0) (default: 1e-8)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param MAX_ROW_WORD maximum words per line in `row_file`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @param MAX_COL_WORD maximum words per line in `col_file`
//' @param COL_WORD_SEP word separation character to replace white space
//'
//' @return a list
//' \itemize{
//' \item `PB` pseudobulk (average) data (feature x sample)
//' \item `sum` pseudobulk (sum) data (feature x sample)
//' \item `matched.sum` kNN-matched pseudobulk data (feature x sample)
//' \item `sum_db` batch-specific sum (feature x batch)
//' \item `size` size per sample (sample x 1)
//' \item `prob_bs` batch-specific frequency (batch x sample)
//' \item `size_bs` batch-specific size (batch x sample)
//' \item `batch.effect` batch effect (feature x batch)
//' \item `log.batch.effect` log batch effect (feature x batch)
//' \item `batch.names` batch names (batch x 1)
//' \item `positions` pseudobulk sample positions (cell x 1)
//' \item `rand.dict` random dictionary (proj factor x feature)
//' \item `rand.proj` random projection results (sample x proj factor)
//' \item `colnames` column (cell) names
//' \item `rownames` feature (gene) names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_random_bulk_data(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    const std::string idx_file,
    const std::size_t num_factors,
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_covar_n = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_covar_d = R_NilValue,
    const Rcpp::Nullable<Rcpp::StringVector> r_batch = R_NilValue,
    const std::size_t rseed = 42,
    const bool verbose = false,
    const std::size_t NUM_THREADS = 1,
    const std::size_t BLOCK_SIZE = 100,
    const bool do_log1p = false,
    const bool do_down_sample = false,
    const bool save_rand_proj = false,
    const std::size_t KNN_CELL = 10,
    const std::size_t CELL_PER_SAMPLE = 100,
    const std::size_t BATCH_ADJ_ITER = 100,
    const double a0 = 1e-8,
    const double b0 = 1,
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@')
{

    log1p_op<Mat> log1p;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    CHK_RETL(convert_bgzip(mtx_file));
    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    std::vector<std::string> row_names, col_names;
    CHK_RETL_(read_line_file(row_file, row_names, MAX_ROW_WORD, ROW_WORD_SEP),
              "Failed to read the row names");
    CHK_RETL_(read_line_file(col_file, col_names, MAX_COL_WORD, COL_WORD_SEP),
              "Failed to read the col names");

    std::vector<Index> mtx_idx;
    CHK_RETL_(read_mmutil_index(idx_file, mtx_idx),
              "Failed to read the index file:" << std::endl
                                               << idx_file << std::endl
                                               << "Consider rebuilding it."
                                               << std::endl);

    const Index D = info.max_row; // dimensionality
    const Index N = info.max_col; // number of cells
    const Index K = num_factors;  // tree depths in implicit bisection
    const Index block_size = BLOCK_SIZE;

    ASSERT_RETL(D == row_names.size(),
                "|rows| " << row_names.size() << " != " << D);
    ASSERT_RETL(N == col_names.size(),
                "|cols| " << col_names.size() << " != " << N);

    Mat X_nr;
    if (r_covar_n.isNotNull()) {
        X_nr = Rcpp::as<Mat>(Rcpp::NumericMatrix(r_covar_n));
        TLOG_(verbose,
              "Read some covariates "
                  << " X_nr " << X_nr.rows() << " x " << X_nr.cols() << ",");
        TLOG_(verbose, "of which effects will be removed ");
        TLOG_(verbose, "from random projection data.");
        ASSERT_RETL(X_nr.rows() == N, "incompatible covariate matrix");
    }

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
    Mat R_kd = Mat::NullaryExpr(K, D, rnorm);

    if (verbose) {
        TLOG("Random aggregator matrix: " << R_kd.rows() << " x "
                                          << R_kd.cols());
    }

    Mat X_dr;
    if (r_covar_d.isNotNull()) {
        X_dr = Rcpp::as<Mat>(Rcpp::NumericMatrix(r_covar_d));
        TLOG_(verbose,
              "Read some covariates "
                  << " X_dr " << X_dr.rows() << " x " << X_dr.cols() << ",");
        TLOG_(verbose, "of which effects will be removed ");
        TLOG_(verbose, "from the random aggregator matrix.");
        ASSERT_RETL(X_dr.rows() == D, "incompatible covariate matrix");
    }

    Mat Q_kn = Mat::Zero(K, N);

    TLOG_(verbose, "Collecting random projection data");

    Mat YtX_nr;

    if (X_dr.rows() == D && X_dr.cols() > 0) {
        YtX_nr.resize(N, X_dr.cols());
        YtX_nr.setZero();
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);

        //////////////////////
        // random aggregate //
        //////////////////////

        Mat _y_dn = do_log1p ? matched_data.read(lb, ub).unaryExpr(log1p) :
                               matched_data.read(lb, ub);

#pragma omp critical
        {
            Mat temp_kn = R_kd * _y_dn;
            for (Index i = 0; i < temp_kn.cols(); ++i) {
                const Index j = i + lb;
                Q_kn.col(j) = temp_kn.col(i);
            }
        }

        ///////////////////////////////
        // correlation with the X_dr //
        ///////////////////////////////

        if (X_dr.rows() == D && X_dr.cols() > 0) {
            Mat temp_rn = X_dr.transpose() * _y_dn;
            for (Index i = 0; i < temp_rn.cols(); ++i) {
                const Index j = i + lb;
                YtX_nr.row(j) = temp_rn.col(i).transpose();
            }
        }
    }

    TLOG_(verbose, "Finished random matrix projection");

    // Regress out X_nr
    if (X_nr.cols() > 0 && X_nr.rows() == N) {
        Mat Qt = Q_kn.transpose(); // N x K
        residual_columns(Qt, X_nr);
        standardize_columns_inplace(Qt);
        Q_kn = Qt.transpose();

        TLOG_(verbose,
              "Regressed out X_nr from Q: " << Q_kn.rows() << " x "
                                            << Q_kn.cols());
    }

    // Regress out YtX_nr
    if (YtX_nr.cols() > 0 && YtX_nr.rows() == N) {
        Mat Qt = Q_kn.transpose(); // N x K
        residual_columns(Qt, YtX_nr);
        standardize_columns_inplace(Qt);
        Q_kn = Qt.transpose();

        TLOG_(verbose,
              "Regressed out YtX_nr from Q: " << Q_kn.rows() << " x "
                                              << Q_kn.cols());
    }

    /////////////////////////////////////////////////
    // Step 2. Orthogonalize the projection matrix //
    /////////////////////////////////////////////////

    Mat vv;

    if (Q_kn.cols() < 1000) {
        Eigen::BDCSVD<Mat> svd;
        svd.compute(Q_kn, Eigen::ComputeThinU | Eigen::ComputeThinV);
        vv = svd.matrixV();
    } else {
        const std::size_t lu_iter = 5;
        RandomizedSVD<Mat> svd(Q_kn.rows(), lu_iter);
        svd.compute(Q_kn);
        vv = svd.matrixV();
    }

    ASSERT_RETL(vv.rows() == N, " failed SVD for Q");

    Mat RD = standardize_columns(vv); // N x K
    TLOG(RD.rows() << " x " << RD.cols());

    TLOG_(verbose, "SVD on the projected: " << RD.rows() << " x " << RD.cols());

    ////////////////////////////////////////////////
    // Step 3. sorting in an implicit binary tree //
    ////////////////////////////////////////////////

    IntVec bb(N);
    bb.setZero();

    for (Index k = 0; k < K; ++k) {
        auto binary_shift = [&k](const Scalar &x) -> Index {
            return x > 0. ? (1 << k) : 0;
        };
        bb += RD.col(k).unaryExpr(binary_shift);
    }

    TLOG_(verbose, "Assigned random membership: [0, " << bb.maxCoeff() << ")");

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
    TLOG_(verbose, "Identified " << S << " pseudo-bulk samples");

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
    std::vector<std::vector<Index>> pb_cells = make_index_vec_vec(positions);

    const Index NA_POS = S;
    if (do_down_sample) {
        TLOG_(verbose, "down-sampling to " << CELL_PER_SAMPLE << " per sample");
        down_sample_vec_vec(pb_cells, CELL_PER_SAMPLE, rng);
        std::fill(positions.begin(), positions.end(), NA_POS);
        for (std::size_t s = 0; s < pb_cells.size(); ++s) {
            for (auto x : pb_cells.at(s))
                positions[x] = s;
        }
    }

    TLOG_(verbose,
          "Start collecting statistics... "
              << " for " << pb_cells.size() << " samples");

    Mat mu_ds = Mat::Ones(D, S);
    Mat ysum_ds = Mat::Zero(D, S);
    RowVec size_s = RowVec::Zero(S);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {
        const Index ub = std::min(N, block_size + lb);
        Mat y = matched_data.read(lb, ub);
#pragma omp critical
        {
            for (Index i = 0; i < (ub - lb); ++i) {
                const Index j = i + lb;
                const Index s = positions.at(j);
                ysum_ds.col(s) += y.col(i);
                size_s(s) += 1.;
            }
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

    Mat delta_db, log_delta_db;
    Mat prob_bs, n_bs, gamma_ds, zsum_ds, delta_num_db, delta_denom_db;

    if (B > 1) {

        delta_db.resize(D, B); // gene x batch
        delta_db.setOnes();

        TLOG_(verbose,
              "Random pseudo-bulk estimation while "
                  << "accounting for " << B << " batch effects");

        delta_num_db = Mat::Zero(D, B);   // gene x batch numerator
        delta_denom_db = Mat::Zero(D, B); // gene x batch denominator

        prob_bs = Mat::Zero(B, S); // batch x PB prob
        n_bs = Mat::Zero(B, S);    // batch x PB freq

        CHK_RETL_(matched_data.build_annoy_index(Q_kn),
                  "Failed to build Annoy Indexes: " << Q_kn.rows() << " x "
                                                    << Q_kn.cols());

        ////////////////////////////
        // Step a. precalculation //
        ////////////////////////////

        TLOG_(verbose, "Start collecting sufficient statistics");

        zsum_ds = Mat::Zero(D, S); // gene x PB mean

        Index Nprocessed = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index s = 0; s < pb_cells.size(); ++s) {

            const std::vector<Index> &_cells_s = pb_cells.at(s);

            Mat yy = do_log1p ? matched_data.read(_cells_s).unaryExpr(log1p) :
                                matched_data.read(_cells_s);

            Mat zz = do_log1p ?
                matched_data.read_counterfactual(_cells_s).unaryExpr(log1p) :
                matched_data.read_counterfactual(_cells_s);

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
        TLOG_(verbose, "Collected sufficient statistics");

        gamma_param_t<Mat, RNG> delta_param(D, B, a0, b0, rng);
        gamma_param_t<Mat, RNG> mu_param(D, S, a0, b0, rng);
        gamma_param_t<Mat, RNG> gamma_param(D, S, a0, b0, rng);

        ///////////////////////////////
        // Step b. Iterative updates //
        ///////////////////////////////

        Eigen::setNbThreads(NUM_THREADS);
        TLOG_(verbose,
              "Iterative optimization "
                  << " with " << Eigen::nbThreads()
                  << " Eigen library threads");

        gamma_ds = Mat::Ones(D, S); // bias on the side of CF

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

            ////////////////////
            // residual for z //
            ////////////////////

            gamma_param
                .update(zsum_ds,
                        (mu_ds.array().rowwise() * size_s.array()).matrix());
            gamma_param.calibrate();
            gamma_ds = gamma_param.mean();

            ///////////////////////////////
            // batch-specific components //
            ///////////////////////////////
            delta_denom_db = mu_ds * n_bs.transpose();
            delta_param.update(delta_num_db, delta_denom_db);
            delta_param.calibrate();
            delta_db = delta_param.mean();

            TLOG_(verbose,
                  "Batch optimization [ " << (t + 1) << " / "
                                          << (BATCH_ADJ_ITER) << " ]");

            if (!verbose) {
                Rcpp::Rcerr << "+ " << std::flush;
                if (t > 0 && t % 10 == 0) {
                    Rcpp::Rcerr << "\r" << std::flush;
                }
            }
        }
        Rcpp::Rcerr << "\r" << std::flush;

        delta_db = delta_param.mean();
        log_delta_db = delta_param.log_mean();

        mu_ds = mu_param.mean();

    } else {

        //////////////////////////////////////////////////
        // Pseudobulk without considering batch effects //
        //////////////////////////////////////////////////

        TLOG_(verbose, "Pseudobulk estimation in a vanilla mode");

        gamma_param_t<Mat, RNG> mu_param(D, S, a0, b0, rng);
        Mat temp_ds = Mat::Ones(D, S).array().rowwise() * size_s.array();
        mu_param.update(ysum_ds, temp_ds);
        mu_param.calibrate();
        mu_ds = mu_param.mean();
    }

    TLOG_(verbose, "Final RPB: " << mu_ds.rows() << " x " << mu_ds.cols());

    using namespace rcpp::util;
    using namespace Rcpp;

    // convert zero-based to 1-based for R
    std::vector<Index> r_positions(positions.size());
    convert_r_index(positions, r_positions);

    std::vector<std::string> s_;
    for (std::size_t s = 1; s <= S; ++s)
        s_.push_back(std::to_string(s));
    std::vector<std::string> b_;
    for (std::size_t b = 1; b <= B; ++b)
        b_.push_back(std::to_string(b));

    std::vector<std::string> d_ = row_names;

    if (!save_rand_proj) {
        Q_kn.resize(0, 0);
    }

    TLOG_(verbose, "Done");

    return List::create(_["PB"] = named(mu_ds, d_, s_),
                        _["sum"] = named(ysum_ds, d_, s_),
                        _["matched.sum"] = named(zsum_ds, d_, s_),
                        _["sum_db"] = named(delta_num_db, d_, b_),
                        _["size"] = size_s,
                        _["prob_bs"] = named(prob_bs, b_, s_),
                        _["size_bs"] = named(n_bs, b_, s_),
                        _["batch.effect"] = named(delta_db, d_, b_),
                        _["log.batch.effect"] = named(log_delta_db, d_, b_),
                        _["batch.membership"] =
                            matched_data.get_exposure_mapping(),
                        _["batch.names"] = matched_data.get_exposure_names(),
                        _["positions"] = r_positions,
                        _["rand.dict"] = R_kd,
                        _["rand.proj"] = Q_kn.transpose(),
                        _["colnames"] = col_names,
                        _["rownames"] = row_names);
}
