#include "rcpp_asap_interaction_pb.hh"

//' Generate approximate pseudo-bulk interaction data by random projections
//'
//' @param mtx_file matrix-market-formatted data file (bgzip)
//' @param row_file row names (gene/feature names)
//' @param col_file column names (cell/column names)
//' @param idx_file matrix-market colum index file
//' @param num_factors a desired number of random factors
//' @param W_nn_list list(src.index, tgt.index, [weights]) for columns
//'
//' @param A_dd_list list(src.index, tgt.index, [weights]) for features
//'
//' @param rseed random seed
//' @param do_product yi * yj for interaction (default: FALSE)
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_down_sample down-sampling (default: FALSE)
//' @param save_rand_proj save random projection (default: FALSE)
//' @param weighted_rand_proj save random projection (default: FALSE)
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param EDGE_PER_SAMPLE down-sampling cell per sample (default: 100)
//' @param a0 gamma(a0, b0) (default: 1e-8)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param MAX_ROW_WORD maximum words per line in `row_file`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @param MAX_COL_WORD maximum words per line in `col_file`
//' @param COL_WORD_SEP word separation character to replace white space
//' @param verbose verbosity
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
asap_interaction_random_bulk(
    const std::string mtx_file,
    const std::string row_file,
    const std::string col_file,
    const std::string idx_file,
    const std::size_t num_factors,
    const Rcpp::List W_nn_list,
    const Rcpp::Nullable<Rcpp::List> A_dd_list = R_NilValue,
    const std::size_t rseed = 42,
    const bool do_product = false,
    const bool do_log1p = false,
    const bool do_down_sample = false,
    const bool save_rand_proj = false,
    const bool weighted_rand_proj = false,
    const std::size_t NUM_THREADS = 1,
    const std::size_t BLOCK_SIZE = 100,
    const std::size_t EDGE_PER_SAMPLE = 100,
    const double a0 = 1e-8,
    const double b0 = 1,
    const std::size_t MAX_ROW_WORD = 2,
    const char ROW_WORD_SEP = '_',
    const std::size_t MAX_COL_WORD = 100,
    const char COL_WORD_SEP = '@',
    const bool verbose = false)
{

    using namespace asap::pb;

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

    mtx_data_t mtx_data(mtx_tuple_t(mtx_tuple_t::MTX(mtx_file),
                                    mtx_tuple_t::ROW(row_file),
                                    mtx_tuple_t::COL(col_file),
                                    mtx_tuple_t::IDX(idx_file)),
                        MAX_ROW_WORD,
                        ROW_WORD_SEP);

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);

    /////////////////////////////////////////////////////
    // Step 1. Build weighted kNN graph from the input //
    /////////////////////////////////////////////////////

    SpMat W_nn;
    rcpp::util::build_sparse_mat(W_nn_list, Ncell, Ncell, W_nn);
    TLOG_(verbose, "kNN graph W: " << W_nn.rows() << " x " << W_nn.cols());

    SpMat A_dd;
    if (A_dd_list.isNotNull()) {
        rcpp::util::build_sparse_mat(Rcpp::List(A_dd_list), D, D, A_dd);
        TLOG_(verbose, "Row Network: " << A_dd.rows() << " x " << A_dd.cols());
    } else {
        build_diagonal(D, 1., A_dd);
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

    const Index M = W_nn.nonZeros();
    Mat Q_km = Mat::Zero(K, M);
    TLOG_(verbose, "Collecting random projection data");

    {
        Index m = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index i = 0; i < W_nn.outerSize(); ++i) {
            SpMat y_di = mtx_data.read(i, i + 1);
            for (SpMat::InnerIterator jt(W_nn, i); jt; ++jt) {
                const Index j = jt.index();
                SpMat y_dj = mtx_data.read(j, j + 1);

                const Scalar w_ij = jt.value() * product_similarity(y_di, y_dj);

                // yij <- (yi + yj) * wij (D x 1)
                // Q_km <- R_kd * yij     (K x 1)
#pragma omp critical
                {
                    if (do_product) {
                        Q_km.col(m) = R_kd * (y_di.cwiseProduct(y_dj)) * w_ij;
                    } else {
                        Q_km.col(m) = w_ij * 0.5 *
                            ((R_kd * A_dd * y_di + R_kd * y_dj) +
                             (R_kd * y_di + R_kd * A_dd * y_dj));
                    }
                    ++m;

                    if (verbose) {
                        Rcpp::Rcerr << "\rProcessed: " << m << std::flush;
                    } else {
                        Rcpp::Rcerr << "+ " << std::flush;
                        if (m % 1000 == 0)
                            Rcpp::Rcerr << "\r" << std::flush;
                    }
                }
            }
        }
    }

    if (!verbose) {
        Rcpp::Rcerr << std::endl;
    } else {
        TLOG("Done");
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
                                    EDGE_PER_SAMPLE);

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
        for (Index i = 0; i < W_nn.outerSize(); ++i) {
            SpMat y_di = mtx_data.read(i, i + 1);
            for (SpMat::InnerIterator jt(W_nn, i); jt; ++jt) {
                const Index j = jt.index();
                SpMat y_dj = mtx_data.read(j, j + 1);

                const Scalar w_ij = jt.value() * product_similarity(y_di, y_dj);

                // yij <- (yi + yj) * wij (D x 1)
                // Q_km <- R_kd * yij     (K x 1)
#pragma omp critical
                {
                    const Index s = positions.at(m);
                    if (s < NA_POS) {
                        if (do_product) {
                            ysum_ds.col(s) += y_di.cwiseProduct(y_dj) * w_ij;
                        } else {
                            ysum_ds.col(s) += w_ij * 0.5 *
                                (A_dd * y_di + y_dj + y_di + A_dd * y_dj);
                        }
                        size_s(s) += 1.;
                    }
                    ++m;
                    if (verbose) {
                        Rcpp::Rcerr << "\rProcessed: " << m << std::flush;
                    } else {
                        Rcpp::Rcerr << "+ " << std::flush;
                        if (m % 1000 == 0)
                            Rcpp::Rcerr << "\r" << std::flush;
                    }
                }
            }
        }
    }

    if (!verbose) {
        Rcpp::Rcerr << std::endl;
    } else {
        TLOG("Done");
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
