#include "rcpp_asap_pb_rbind.hh"

//' Generate approximate pseudo-bulk data by random projections
//' while sharing columns/cells across multiple data sets.
//' Vertical concatenation.
//'
//' @param mtx_files matrix-market-formatted data files (bgzip)
//' @param row_files row names (gene/feature names)
//' @param col_files column names (cell/column names)
//' @param idx_files matrix-market colum index files
//' @param num_factors a desired number of random factors per data set
//'
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param CELL_NORM normalization constant per each data point
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_down_sample down-sampling (default: FALSE)
//' @param save_rand_proj save random projection (default: FALSE)
//' @param weighted_rand_proj save random projection (default: FALSE)
//' @param CELL_PER_SAMPLE down-sampling cell per sample (default: 100)
//' @param a0 gamma(a0, b0) (default: 1e-8)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param MAX_ROW_WORD maximum words per line in `row_files[i]`
//' @param ROW_WORD_SEP word separation character to replace white space
//' @param MAX_COL_WORD maximum words per line in `col_files[i]`
//' @param COL_WORD_SEP word separation character to replace white space
//'
//' @return a list
//' \itemize{
//' \item `PB.list` pseudobulk (average) data (feature x sample) for each type
//' \item `sum.list` pseudobulk (sum) data (feature x sample) for each type
//' \item `size.list` size per sample (sample x 1) for each type
//' \item `rownames.list` feature (gene) names for each type
//' \item `colnames` column (cell) names across data types
//' \item `positions` pseudobulk sample positions (cell x 1)
//' \item `rand.proj` random projection results (sample x proj factor)
//' \item `colnames` column (cell) names
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_random_bulk_rbind_mtx(const std::vector<std::string> mtx_files,
                           const std::vector<std::string> row_files,
                           const std::vector<std::string> col_files,
                           const std::vector<std::string> idx_files,
                           const std::size_t num_factors,
                           const std::size_t rseed = 42,
                           const bool verbose = true,
                           const std::size_t NUM_THREADS = 1,
                           const double CELL_NORM = 1e4,
                           const std::size_t BLOCK_SIZE = 1000,
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

    const std::size_t nthreads =
        (NUM_THREADS > 0 ? NUM_THREADS : omp_get_max_threads());

    const Scalar cell_norm = CELL_NORM;

    using namespace asap::pb;

    log1p_op<Mat> log1p;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    const Index B = mtx_files.size();

    ASSERT_RETL(B > 0, "Empty mtx file names");
    ASSERT_RETL(row_files.size() == B, "Need a row file for each data type");
    ASSERT_RETL(col_files.size() == B, "Need a col file for each data type");

    ASSERT_RETL(all_files_exist(mtx_files, verbose),
                "missing in the mtx files");
    ASSERT_RETL(all_files_exist(row_files, verbose),
                "missing in the row files");
    ASSERT_RETL(all_files_exist(col_files, verbose),
                "missing in the col files");

    for (Index b = 0; b < B; ++b) {
        CHK_RETL(convert_bgzip(mtx_files.at(b)));
    }

    ASSERT_RETL(idx_files.size() == B, "Need an index file for each data type");

    for (Index b = 0; b < B; ++b) {
        if (!file_exists(idx_files.at(b))) {
            CHK_RETL(build_mmutil_index(mtx_files.at(b), idx_files.at(b)));
            TLOG_(verbose, "built the missing index: " << idx_files.at(b));
        }
    }

    TLOG_(verbose, "Checked the files");

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);

    //////////////////////////////////////////////////////
    // First figure out the intersection of the columns //
    //////////////////////////////////////////////////////

    std::vector<std::string> columns;
    std::unordered_map<std::string, Index> col2pos;

    rcpp::util::take_common_names(col_files,
                                  columns,
                                  col2pos,
                                  false,
                                  MAX_COL_WORD,
                                  COL_WORD_SEP);

    TLOG_(verbose, "Found " << col2pos.size() << " column names");

    ASSERT_RETL(columns.size() > 0, "Empty column names!");

    const Index block_size = BLOCK_SIZE; //
    const Index Ntot = columns.size();   //
    const Index Ktot = num_factors;      // tree depths in implicit bisection

    const Index K = std::max(1., std::ceil(num_factors / B));

    //////////////////////////////////////////////////
    // Step 1. random projection for each data type //
    //////////////////////////////////////////////////

    Mat Q_kn = Mat::Zero(K * B, Ntot);

    for (Index b = 0; b < B; ++b) {

        mtx_tuple_t tup(mtx_tuple_t::MTX(mtx_files.at(b)),
                        mtx_tuple_t::ROW(row_files.at(b)),
                        mtx_tuple_t::COL(col_files.at(b)),
                        mtx_tuple_t::IDX(idx_files.at(b)));

        mtx_data_t data(tup, MAX_ROW_WORD, ROW_WORD_SEP);

        std::vector<std::string> cols_b;
        CHK_RETL_(read_line_file(col_files.at(b),
                                 cols_b,
                                 MAX_COL_WORD,
                                 COL_WORD_SEP),
                  "Unable to read the column file: " << col_files.at(b));

        const Index D = data.max_row();
        const Index Nb = data.max_col();
        ASSERT_RETL(Nb == cols_b.size(),
                    "Found: " << Nb << " != " << cols_b.size());

        Mat Rb_kd;
        sample_random_projection(D, K, rseed + b, Rb_kd);

        if (weighted_rand_proj) {
            apply_mtx_row_sd(mtx_files.at(b),
                             idx_files.at(b),
                             Rb_kd,
                             verbose,
                             nthreads,
                             BLOCK_SIZE,
                             do_log1p);
            if (verbose) {
                TLOG("Weighted random projection matrix");
            }
        }

        const Index kk = K * b; // a starting point of the factors

        if (verbose) {
            Rcpp::Rcerr << "Calibrating " << Ntot << " columns..." << std::endl;
        }

        Index Nprocessed = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(nthreads)
#pragma omp for
#endif
        for (Index lb = 0; lb < Nb; lb += block_size) {

            const Index ub = std::min(Nb, block_size + lb);

            SpMat _y_dn = do_log1p ? data.read(lb, ub).unaryExpr(log1p) :
                                     data.read(lb, ub);
            normalize_columns_inplace(_y_dn);
            const Mat y_dn = _y_dn * cell_norm;

            Mat temp_kn = Rb_kd * y_dn;

#pragma omp critical
            {
                for (Index loc = 0; loc < temp_kn.cols(); ++loc) {
                    const Index j = loc + lb;
                    if (col2pos.count(cols_b[j]) > 0) {
                        const Index glob = col2pos[cols_b[j]];
                        Q_kn.block(kk, glob, K, 1) = temp_kn.col(loc);
                    }
                }
                Nprocessed += temp_kn.cols();
            }

            if (verbose) {
                Rcpp::Rcerr << "\rProcessed: " << Nprocessed << std::flush;
            } else {
                Rcpp::Rcerr << "+ " << std::flush;
                if (Nprocessed % 1000 == 0)
                    Rcpp::Rcerr << "\r" << std::flush;
            }
        } // end of batch

        Rcpp::Rcerr << std::endl;

        TLOG_(verbose,
              "processed file set #" << (b + 1) << " for random projection");
    }
    TLOG_(verbose, "Finished random matrix projection");

    ////////////////////////////////////////////////
    // Step 2. Orthogonalize the projected matrix //
    ////////////////////////////////////////////////

    {
        Q_kn.transposeInPlace();
        standardize_columns_inplace(Q_kn);
        Q_kn.transposeInPlace();
    }

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

    ASSERT_RETL(vv.rows() == Ntot, " failed SVD for Q");

    Mat RD_nk = standardize_columns(vv); // N x K

    TLOG_(verbose,
          "SVD on the projected: " << RD_nk.rows() << " x " << RD_nk.cols());

    ////////////////////////////////////////////////
    // Step 3. sorting in an implicit binary tree //
    ////////////////////////////////////////////////

    std::vector<Index> positions;
    randomly_assign_rows_to_samples(RD_nk,
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

    std::list<Rcpp::NumericMatrix> mu_list, sum_list, size_list;
    std::list<Rcpp::StringVector> row_list;

    for (Index b = 0; b < B; ++b) {

        mtx_tuple_t tup(mtx_tuple_t::MTX(mtx_files.at(b)),
                        mtx_tuple_t::ROW(row_files.at(b)),
                        mtx_tuple_t::COL(col_files.at(b)),
                        mtx_tuple_t::IDX(idx_files.at(b)));

        mtx_data_t data(tup, MAX_ROW_WORD, ROW_WORD_SEP);

        std::vector<std::string> row_names, col_names;
        CHK_RETL_(read_line_file(row_files.at(b),
                                 row_names,
                                 MAX_ROW_WORD,
                                 ROW_WORD_SEP),
                  "Failed to read the row names: " << row_files.at(b));
        CHK_RETL_(read_line_file(col_files.at(b),
                                 col_names,
                                 MAX_COL_WORD,
                                 COL_WORD_SEP),
                  "Failed to read the col names: " << col_files.at(b));

        const Index D = data.max_row();
        const Index Nb = data.max_col();

        Mat mu_ds = Mat::Zero(D, S);
        Mat ysum_ds = Mat::Zero(D, S);
        RowVec size_s = RowVec::Zero(S);

        if (verbose) {
            Rcpp::Rcerr << "Calibrating " << Ntot << " columns..." << std::endl;
        }

        Index Nprocessed = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(nthreads)
#pragma omp for
#endif
        for (Index lb = 0; lb < Nb; lb += block_size) {

            const Index ub = std::min(Nb, block_size + lb);
            Mat y = data.read(lb, ub);

#pragma omp critical
            {
                for (Index i = 0; i < (ub - lb); ++i) {
                    const Index j = i + lb;
                    if (col2pos.count(col_names[j]) > 0) {
                        const Index glob = col2pos[col_names[j]];
                        const Index s = positions.at(glob);
                        if (s < NA_POS) {
                            ysum_ds.col(s) += y.col(i);
                            size_s(s) += 1.;
                        }
                    }
                }
                Nprocessed += y.cols();
            }

            if (verbose) {
                Rcpp::Rcerr << "\rProcessed: " << Nprocessed << std::flush;
            } else {
                Rcpp::Rcerr << "+ " << std::flush;
                if (Nprocessed % 1000 == 0)
                    Rcpp::Rcerr << "\r" << std::flush;
            }
        }

        Rcpp::Rcerr << std::endl;
        TLOG_(verbose, "Done");

        //////////////////////////////////////////////////
        // Pseudobulk without considering batch effects //
        //////////////////////////////////////////////////

        TLOG_(verbose, "Pseudobulk estimation");

        gamma_param_t<Mat, RNG> mu_param(D, S, a0, b0, rng);
        Mat temp_ds = Mat::Ones(D, S).array().rowwise() * size_s.array();
        mu_param.update(ysum_ds, temp_ds);
        mu_param.calibrate();
        mu_ds = mu_param.mean();

        TLOG_(verbose,
              "RPB [" << (b + 1) << "]: " << mu_ds.rows() << " x "
                      << mu_ds.cols());

        mu_list.push_back(rcpp::util::named_rows(mu_ds, row_names));
        sum_list.push_back(rcpp::util::named_rows(ysum_ds, row_names));
        size_list.push_back(Rcpp::wrap(size_s));
        row_list.push_back(Rcpp::wrap(row_names));
    }

    if (!save_rand_proj) {
        Q_kn.resize(0, 0);
    }

    TLOG_(verbose, "Done");

    // convert zero-based to 1-based for R
    std::vector<Index> r_positions(positions.size());
    rcpp::util::convert_r_index(positions, r_positions);

    return Rcpp::List::create(Rcpp::_["PB.list"] = mu_list,
                              Rcpp::_["sum.list"] = sum_list,
                              Rcpp::_["size.list"] = size_list,
                              Rcpp::_["rownames.list"] = row_list,
                              Rcpp::_["colnames"] = columns,
                              Rcpp::_["positions"] = r_positions,
                              Rcpp::_["rand.proj"] =
                                  rcpp::util::named_rows(Q_kn.transpose(),
                                                         columns));
}
