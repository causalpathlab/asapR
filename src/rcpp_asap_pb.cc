#include "rcpp_asap_pb.hh"

//' Generate approximate pseudo-bulk data by random projections
//'
//' @param mtx_file matrix-market-formatted data file (bgzip)
//' @param row_file row names (gene/feature names)
//' @param col_file column names (cell/column names)
//' @param idx_file matrix-market colum index file
//' @param num_factors a desired number of random factors
//' @param r_covar_n N x r covariates (default: NULL)
//' @param r_covar_d D x r covariates (default: NULL)
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param CELL_NORM sample normalization constant (default: 1e4)
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_down_sample down-sampling (default: FALSE)
//' @param save_aux_data save auxiliary data (default: FALSE)
//' @param weighted_rand_proj save random projection (default: FALSE)
//' @param CELL_PER_SAMPLE down-sampling cell per sample (default: 100)
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
//' \item `size` size per sample (sample x 1)
//' \item `positions` pseudobulk sample positions (cell x 1)
//' \item `rand.dict` random dictionary (proj factor x feature)
//' \item `rand.proj` random projection results (sample x proj factor)
//' \item `colnames` column (cell) names
//' \item `rownames` feature (gene) names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_random_bulk_mtx(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    const std::string idx_file,
    const std::size_t num_factors,
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_covar_n = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_covar_d = R_NilValue,
    const Rcpp::Nullable<Rcpp::StringVector> rows_restrict = R_NilValue,
    const std::size_t rseed = 42,
    const bool verbose = false,
    const std::size_t NUM_THREADS = 0,
    const double CELL_NORM = 1e4,
    const std::size_t BLOCK_SIZE = 1000,
    const bool do_log1p = false,
    const bool do_down_sample = false,
    const bool save_aux_data = false,
    const bool weighted_rand_proj = false,
    const std::size_t CELL_PER_SAMPLE = 100,
    const double a0 = 1e-8,
    const double b0 = 1,
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@')
{

    const std::size_t nthreads =
        (NUM_THREADS > 0 ? NUM_THREADS : omp_get_max_threads());

    using namespace asap::pb;

    log1p_op<Mat> log1p;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;

    CHK_RETL(convert_bgzip(mtx_file));
    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    std::vector<std::string> row_names, col_names;
    CHK_RETL_(read_line_file(row_file, row_names, MAX_ROW_WORD, ROW_WORD_SEP),
              "Failed to read the row names");
    CHK_RETL_(read_line_file(col_file, col_names, MAX_COL_WORD, COL_WORD_SEP),
              "Failed to read the col names");

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

    mtx_data_t mtx_data(mtx_tuple_t(mtx_tuple_t::MTX(mtx_file),
                                    mtx_tuple_t::ROW(row_file),
                                    mtx_tuple_t::COL(col_file),
                                    mtx_tuple_t::IDX(idx_file)),
                        MAX_ROW_WORD,
                        ROW_WORD_SEP);

    std::vector<std::string> pos2row;
    std::unordered_map<std::string, Index> row2pos;

    if (rows_restrict.isNotNull()) {

        rcpp::util::copy(Rcpp::StringVector(rows_restrict), pos2row);
        make_position_dict(pos2row, row2pos);

        ASSERT_RETL(pos2row.size() == D, "check the rows_restrict");

        mtx_data.relocate_rows(row2pos);
    }

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);

    /////////////////////////////////////////////
    // Step 1. sample random projection matrix //
    /////////////////////////////////////////////

    Mat R_kd;
    sample_random_projection(D, K, rseed, R_kd);

    if (weighted_rand_proj) {
        apply_mtx_row_sd(mtx_file,
                         idx_file,
                         R_kd,
                         verbose,
                         nthreads,
                         BLOCK_SIZE,
                         do_log1p);
        if (verbose) {
            TLOG("Weighted random projection matrix");
        }
    }

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

    const Scalar cell_norm = CELL_NORM;

#if defined(_OPENMP)
#pragma omp parallel num_threads(nthreads)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);

        //////////////////////
        // random aggregate //
        //////////////////////

        SpMat _y_dn = do_log1p ? mtx_data.read(lb, ub).unaryExpr(log1p) :
                                 mtx_data.read(lb, ub);
        normalize_columns_inplace(_y_dn);
        const Mat y_dn = _y_dn * cell_norm;

        Mat temp_kn = R_kd * y_dn;

#pragma omp critical
        {
            for (Index i = 0; i < temp_kn.cols(); ++i) {
                const Index j = i + lb;
                Q_kn.col(j) = temp_kn.col(i);
            }
        }

        ///////////////////////////////
        // correlation with the X_dr //
        ///////////////////////////////

        if (X_dr.rows() == D && X_dr.cols() > 0) {
            Mat temp_rn = X_dr.transpose() * y_dn;
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
        standardize_columns_inplace(Qt);
        residual_columns_inplace(Qt, X_nr);
        standardize_columns_inplace(Qt);
        Q_kn = Qt.transpose();

        TLOG_(verbose,
              "Regressed out X_nr from Q: " << Q_kn.rows() << " x "
                                            << Q_kn.cols());
    }

    // Regress out YtX_nr
    if (YtX_nr.cols() > 0 && YtX_nr.rows() == N) {
        Mat Qt = Q_kn.transpose(); // N x K
        standardize_columns_inplace(Qt);
        residual_columns_inplace(Qt, YtX_nr);
        standardize_columns_inplace(Qt);
        Q_kn = Qt.transpose();

        TLOG_(verbose,
              "Regressed out YtX_nr from Q: " << Q_kn.rows() << " x "
                                              << Q_kn.cols());
    }

    ////////////////////////////////////////////////
    // Step 2. Orthogonalize the projected matrix //
    ////////////////////////////////////////////////

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

    std::vector<Index> positions;
    randomly_assign_rows_to_samples(RD,
                                    positions,
                                    rng,
                                    verbose,
                                    do_down_sample,
                                    CELL_PER_SAMPLE);

    const Index S = *std::max_element(positions.begin(), positions.end());
    const Index NA_POS = S;

    ////////////////////////////////
    // Take sufficient statistics //
    ////////////////////////////////

    TLOG_(verbose,
          "Start collecting statistics... "
              << " for " << S << " samples");

    Mat mu_ds = Mat::Ones(D, S);
    Mat ysum_ds = Mat::Zero(D, S);
    RowVec size_s = RowVec::Zero(S);

#if defined(_OPENMP)
#pragma omp parallel num_threads(nthreads)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {
        const Index ub = std::min(N, block_size + lb);
        Mat y = mtx_data.read(lb, ub);
#pragma omp critical
        {
            for (Index i = 0; i < (ub - lb); ++i) {
                const Index j = i + lb;
                const Index s = positions.at(j);
                if (s < NA_POS) {
                    ysum_ds.col(s) += y.col(i);
                    size_s(s) += 1.;
                }
            }
        }
    }

    //////////////////////////////////////////////////
    // Pseudobulk without considering batch effects //
    //////////////////////////////////////////////////

    TLOG_(verbose, "Pseudobulk estimation");

    gamma_param_t<Mat, RNG> mu_param(D, S, a0, b0, rng);
    Mat temp_ds = Mat::Ones(D, S).array().rowwise() * size_s.array();
    mu_param.update(ysum_ds, temp_ds);
    mu_param.calibrate();
    mu_ds = mu_param.mean();

    TLOG_(verbose, "Final RPB: " << mu_ds.rows() << " x " << mu_ds.cols());

    using namespace rcpp::util;
    using namespace Rcpp;

    // convert zero-based to 1-based for R
    std::vector<Index> r_positions(positions.size());
    convert_r_index(positions, r_positions);

    std::vector<std::string> s_;
    for (std::size_t s = 1; s <= S; ++s)
        s_.push_back(std::to_string(s));

    std::vector<std::string> d_ = row_names;

    if (!save_aux_data) {
        Q_kn.resize(0, 0);
        ysum_ds.resize(0, 0);
    }

    TLOG_(verbose, "Done");

    return List::create(_["PB"] = named(mu_ds, d_, s_),
                        _["sum"] = named(ysum_ds, d_, s_),
                        _["size"] = size_s,
                        _["positions"] = r_positions,
                        _["rand.dict"] = R_kd,
                        _["rand.proj"] = Q_kn.transpose(),
                        _["colnames"] = col_names,
                        _["rownames"] = row_names);
}
