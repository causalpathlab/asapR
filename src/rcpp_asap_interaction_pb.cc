#include "rcpp_asap_interaction_pb.hh"

//' Generate approximate pseudo-bulk interaction data by random projections
//'
//' @param mtx_file matrix-market-formatted data file (bgzip)
//' @param row_file row names (gene/feature names)
//' @param col_file column names (cell/column names)
//' @param idx_file matrix-market colum index file
//' @param num_factors a desired number of random factors
//'
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_down_sample down-sampling (default: FALSE)
//' @param save_rand_proj save random projection (default: FALSE)
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
//' \item `positions` pseudobulk sample positions (cell pair x 1)
//' \item `rand.dict` random dictionary (proj factor x feature)
//' \item `rand.proj` random projection results (sample x proj factor)
//' \item `colnames` column (cell) names
//' \item `rownames` feature (gene) names
//' }
//'
//'
// [[Rcpp::export]]
Rcpp::List
asap_interaction_random_bulk(const std::string mtx_file,
                             const std::string row_file,
                             const std::string col_file,
                             const std::string idx_file,
                             const std::size_t num_factors,
                             const std::vector<std::size_t> knn_src,
                             const std::vector<std::size_t> knn_tgt,
                             const std::vector<float> knn_weight,
                             const std::size_t rseed = 42,
                             const bool verbose = false,
                             const std::size_t NUM_THREADS = 1,
                             const std::size_t BLOCK_SIZE = 100,
                             const bool do_log1p = false,
                             const bool do_down_sample = false,
                             const bool save_rand_proj = false,
                             const bool weighted_rand_proj = false,
                             const std::size_t CELL_PER_SAMPLE = 100,
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

    const Index D = info.max_row;        // dimensionality
    const Index Ncell = info.max_col;    // number of cells
    const Index K = num_factors;         // tree depths in implicit bisection
    const Index block_size = BLOCK_SIZE; //

    ASSERT_RETL(D == row_names.size(),
                "|rows| " << row_names.size() << " != " << D);
    ASSERT_RETL(Ncell == col_names.size(),
                "|cols| " << col_names.size() << " != " << Ncell);

    mtx_data_t mtx_data(mtx_data_t::MTX(mtx_file),
                        mtx_data_t::ROW(row_file),
                        mtx_data_t::IDX(idx_file),
                        MAX_ROW_WORD,
                        ROW_WORD_SEP);

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);

    /////////////////////////////////////////////////////
    // Step 1. Build weighted kNN graph from the input //
    /////////////////////////////////////////////////////

    const Index N = knn_src.size(); // number of pairs

    ASSERT_RETL(
        N == knn_tgt.size(),
        "source and target vectors should have the same number of elements");

    ASSERT_RETL(
        N == knn_weight.size(),
        "source and weight vectors should have the same number of elements");

    SpMat W;
    {
        std::vector<std::tuple<Index, Index, Scalar>> knn_index;
        knn_index.reserve(N);
        for (Index j = 0; j < N; ++j) {
            // convert 1-based to 0-based
            const Index s = knn_src[j] - 1, t = knn_tgt[j] - 1;
            if (s < Ncell && t < Ncell && s >= 0 && t >= 0 && s != t) {
                knn_index.emplace_back(std::make_tuple(s, t, knn_weight[j]));
            }
        }

        W = build_eigen_sparse(knn_index, Ncell, Ncell);
    }

    ///////////////////////////////////////////////
    // Step 2. Sample a random projection matrix //
    ///////////////////////////////////////////////

    Mat R_kd;
    sample_random_projection(D, K, rseed, R_kd);

    if (weighted_rand_proj) {
        apply_mtx_row_sd(mtx_file,
                         idx_file,
                         R_kd,
                         verbose,
                         NUM_THREADS,
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

    /////////////////////////////////////////////////////////
    // Step 3. Randomly project feature incidence patterns //
    /////////////////////////////////////////////////////////

    const Index M = W.nonZeros();
    Mat Q_km = Mat::Zero(K, M);
    TLOG_(verbose, "Collecting random projection data");

    {
        Index m = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index i = 0; i < W.outerSize(); ++i) {
            SpMat y_i = mtx_data.read(i, i + 1);
            for (SpMat::InnerIterator jt(W, i); jt; ++jt) {
                const Index j = jt.col();
                const Scalar w_ij = jt.value();
                SpMat y_j = mtx_data.read(j, j + 1);

                // yij <- (yi + yj) * wij (D x 1)
                // Q_km <- R_kd * yij     (K x 1)
#pragma omp critical
                {
                    Q_km.col(m) = (R_kd * y_i + R_kd * y_j) * w_ij;
                    ++m;
                }
            }
        }
    }

    TLOG_(verbose, "Finished random matrix projection");

    ////////////////////////////////////////////////
    // Step 4. Orthogonalize the projected matrix //
    ////////////////////////////////////////////////

    Mat vv;
    {
        const std::size_t lu_iter = 5;
        RandomizedSVD<Mat> svd(Q_km.rows(), lu_iter);
        svd.compute(Q_km);
        vv = svd.matrixV();
    }

    ASSERT_RETL(vv.rows() == M, " failed SVD for Q");

    Mat RD = standardize_columns(vv); // M x K
    TLOG(RD.rows() << " x " << RD.cols());
    TLOG_(verbose, "SVD on the projected: " << RD.rows() << " x " << RD.cols());

    ///////////////////////////////////////////
    // Step 5. Distribute interaction pairs //
    //////////////////////////////////////////

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

    {
        Index m = 0;
#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index i = 0; i < W.outerSize(); ++i) {
            SpMat y_i = mtx_data.read(i, i + 1);
            for (SpMat::InnerIterator jt(W, i); jt; ++jt) {
                const Index j = jt.col();
                const Scalar w_ij = jt.value();
                SpMat y_j = mtx_data.read(j, j + 1);

                // yij <- (yi + yj) * wij (D x 1)
                // Q_km <- R_kd * yij     (K x 1)
#pragma omp critical
                {
                    const Index s = positions.at(m);
                    if (s < NA_POS) {
                        ysum_ds.col(s) += (y_i + y_j) * w_ij;
                        size_s(s) += 1.;
                    }
                    ++m;
                }
            }
        }
    }

    TLOG_(verbose, "Pseudobulk estimation: " << D << " x " << S);

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

    // std::vector<std::string> s_;
    // for (std::size_t s = 1; s <= S; ++s) {
    //     s_.push_back(std::to_string(s));
    // }

    // std::vector<std::string> d_ = row_names;

    if (!save_rand_proj) {
        Q_km.resize(0, 0);
    }

    TLOG_(verbose, "Done");

    return List::create(_["PB"] = mu_ds,    // d_, s_),
                        _["sum"] = ysum_ds, //d_, s_),
                        _["size"] = size_s,
                        _["positions"] = r_positions,
                        _["rand.dict"] = R_kd,
                        _["rand.proj"] = Q_km.transpose(),
                        _["colnames"] = col_names,
                        _["rownames"] = row_names);
}
