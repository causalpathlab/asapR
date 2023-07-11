#include "rcpp_asap_pb_multi.hh"

//' Generate approximate pseudo-bulk data by random projections
//'
//' @param mtx_files matrix-market-formatted data files (bgzip)
//' @param row_files row names (gene/feature names)
//' @param col_files column names (cell/column names)
//' @param idx_files matrix-market colum index files
//' @param num_factors a desired number of random factors
//' @param take_union_rows take union of rows (default: FALSE)
//' @param rseed random seed
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//' @param do_batch_adj (default: TRUE)
//' @param do_log1p log(x + 1) transformation (default: FALSE)
//' @param do_down_sample down-sampling (default: FALSE)
//' @param KNN_CELL k-NN matching between cells (default: 10)
//' @param CELL_PER_SAMPLE down-sampling cell per sample (default: 100)
//' @param BATCH_ADJ_ITER batch Adjustment steps (default: 100)
//' @param a0 gamma(a0, b0) (default: 1e-8)
//' @param b0 gamma(a0, b0) (default: 1)
//'
// [[Rcpp::export]]
Rcpp::List
asap_random_bulk_data_multi(const std::vector<std::string> mtx_files,
                            const std::vector<std::string> row_files,
                            const std::vector<std::string> col_files,
                            const std::vector<std::string> idx_files,
                            const std::size_t num_factors,
                            const bool take_union_rows = false,
                            const std::size_t rseed = 42,
                            const bool verbose = false,
                            const std::size_t NUM_THREADS = 1,
                            const std::size_t BLOCK_SIZE = 100,
                            const bool do_batch_adj = true,
                            const bool do_log1p = false,
                            const bool do_down_sample = false,
                            const std::size_t KNN_CELL = 10,
                            const std::size_t CELL_PER_SAMPLE = 100,
                            const std::size_t BATCH_ADJ_ITER = 100,
                            const double a0 = 1e-8,
                            const double b0 = 1)
{

    log1p_op<Mat> log1p;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    const Index B = mtx_files.size();

    ASSERT_RETL(B > 0, "Empty mtx file names");
    ASSERT_RETL(row_files.size() == B, "Need a row file for each batch");
    ASSERT_RETL(col_files.size() == B, "Need a col file for each batch");
    ASSERT_RETL(idx_files.size() == B, "Need an index file for each batch");

    ERR_RET(!all_files_exist(mtx_files), "missing in the mtx files");
    ERR_RET(!all_files_exist(row_files), "missing in the row files");
    ERR_RET(!all_files_exist(col_files), "missing in the col files");
    ERR_RET(!all_files_exist(idx_files), "missing in the idx files");

    ////////////////////////////
    // first read global rows //
    ////////////////////////////

    TLOG_(verbose, "Checked the files");

    std::vector<std::string> pos2row;
    std::unordered_map<std::string, Index> row2pos;

    take_row_names(row_files, pos2row, row2pos, take_union_rows);
    TLOG_(verbose, "Found " << row2pos.size() << " row names");

    ASSERT_RETL(pos2row.size() > 0, "Empty row names!");

    const Index D = pos2row.size(); // dimensionality
    const Index K = num_factors;    // tree depths in implicit bisection
    auto count_add_cols = [](Index a, std::string mtx) -> Index {
        mm_info_reader_t info;
        CHECK(peek_bgzf_header(mtx, info));
        return a + info.max_col;
    };
    const Index Ntot =
        std::accumulate(mtx_files.begin(), mtx_files.end(), 0, count_add_cols);
    const Index block_size = BLOCK_SIZE;

    TLOG_(verbose, D << " x " << Ntot);

    for (Index b = 0; b < B; ++b) {
        CHK_RETL(convert_bgzip(mtx_files.at(b)));
    }

    std::vector<std::string> columns;
    columns.reserve(Ntot);

    for (Index b = 0; b < B; ++b) {
        std::vector<std::string> col_b;
        CHECK(read_vector_file(col_files.at(b), col_b))
        std::copy(col_b.begin(), col_b.end(), std::back_inserter(columns));
    }

    /////////////////////////////////////////////
    // Step 1. sample random projection matrix //
    /////////////////////////////////////////////

    using norm_dist_t = boost::random::normal_distribution<Scalar>;
    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    norm_dist_t norm_dist(0., 1.);

    auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };
    Mat R_kd = Mat::NullaryExpr(K, D, rnorm);

    Mat Q_kn = Mat::Zero(K, Ntot);

    std::vector<std::vector<Index>> batch_glob_map;
    std::vector<Index> batch_membership(Ntot);

    TLOG_(verbose, "Collecting random projection data");

    Index offset = 0;

    for (Index b = 0; b < B; ++b) {

        mtx_data_t data(mtx_data_t::MTX(mtx_files.at(b)),
                        mtx_data_t::ROW(row_files.at(b)),
                        mtx_data_t::IDX(idx_files.at(b)));

        // Find features in the global mapping
        Mat R_matched_kd(K, data.info.max_row);
        R_matched_kd.setZero();

        std::vector<std::string> &sub_rows = data.sub_rows;

        for (Index ri = 0; ri < sub_rows.size(); ++ri) {
            const std::string &r = sub_rows.at(ri);
            if (row2pos.count(r) > 0) {
                const Index d = row2pos.at(r);
                R_matched_kd.col(ri) = R_kd.col(d);
            }
        }

        const Index Nb = data.info.max_col;
        batch_glob_map.emplace_back(std::vector<Index> {});

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index lb = 0; lb < Nb; lb += block_size) {

            const Index ub = std::min(Nb, block_size + lb);
            Mat y_dn = do_log1p ? data.read(lb, ub).unaryExpr(log1p) :
                                  data.read(lb, ub);
            Mat temp_kn = R_matched_kd * y_dn;

#pragma omp critical
            {
                for (Index loc = 0; loc < temp_kn.cols(); ++loc) {
                    const Index glob = loc + lb + offset;
                    batch_glob_map[b].emplace_back(glob);
                    batch_membership[glob] = b;
                    Q_kn.col(glob) = temp_kn.col(loc);
                }
            }
        }

        offset += Nb;

        TLOG_(verbose,
              "processed file set #" << (b + 1) << " for random projection of "
                                     << Nb << " / " << offset << " cells");
    }

    if (B > 1 && do_batch_adj) {

        at_least_one_op<Mat> at_least_one;
        at_least_zero_op<Mat> at_least_zero;
        const Scalar tol = 1e-4;

        Mat s1_kb = Mat::Zero(K, B);
        Mat s2_kb = Mat::Zero(K, B);
        Mat n_kb = Mat::Zero(K, B);
        for (Index j = 0; j < batch_membership.size(); ++j) {
            const Index b = batch_membership.at(j);
            s1_kb.col(b) += Q_kn.col(j);
            s2_kb.col(b) += Q_kn.col(j).cwiseProduct(Q_kn.col(j));
            n_kb.col(b).array() += 1.;
        }

        Mat mu_kb = s1_kb.cwiseQuotient(n_kb.unaryExpr(at_least_one));
        Mat sig_kb = (s2_kb.cwiseQuotient(n_kb.unaryExpr(at_least_one)) -
                      mu_kb.cwiseProduct(mu_kb))
                         .unaryExpr(at_least_zero)
                         .cwiseSqrt();

        for (Index j = 0; j < batch_membership.size(); ++j) {
            const Index b = batch_membership.at(j);
            Q_kn.col(j) -= mu_kb.col(b);
            Q_kn.col(j).array() /= (sig_kb.array().col(b) + tol);
        }

        TLOG_(verbose,
              "Regressed out "
                  << "batch info from Q: " << Q_kn.rows() << " x "
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

    ASSERT_RETL(vv.rows() == Ntot, " failed SVD for Q");

    Mat Q_nk = Q_kn.transpose();      // N x K
    Mat RD = standardize_columns(vv); // N x K
    TLOG(RD.rows() << " x " << RD.cols());

    TLOG_(verbose, "SVD on the projected: " << RD.rows() << " x " << RD.cols());

    ////////////////////////////////////////////////
    // Step 3. sorting in an implicit binary tree //
    ////////////////////////////////////////////////

    IntVec bb(Ntot);
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
    // Step 4. Create pseudoubulk matrix //
    ///////////////////////////////////////

    std::vector<Index> positions(membership.size());

    auto _pos_op = [&pb_position](const Index x) -> Index {
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

    Mat delta_num_db = Mat::Zero(D, B);   // gene x batch numerator
    Mat delta_denom_db = Mat::Zero(D, B); // gene x batch denominator
    Mat n_bs = Mat::Zero(B, S);           // batch x PB freq
    Mat prob_bs = Mat::Zero(B, S);        // batch x PB prob

    Mat delta_db, log_delta_db;

    {
        offset = 0;
        delta_num_db.setZero();

        for (Index b = 0; b < B; ++b) { // each batch

            mtx_data_t data(mtx_data_t::MTX(mtx_files.at(b)),
                            mtx_data_t::ROW(row_files.at(b)),
                            mtx_data_t::IDX(idx_files.at(b)));

            const Index Nb = data.info.max_col;

            TLOG_(verbose, Nb << " samples");
            data.relocate_rows(row2pos);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
            for (Index lb = 0; lb < Nb; lb += block_size) {

                const Index ub = std::min(Nb, block_size + lb);
                const Mat y = data.read_reloc(lb, ub);

                for (Index loc = 0; loc < (ub - lb); ++loc) {
                    const Index glob = loc + lb + offset;
                    const Index s = positions.at(glob);

                    if (s < S) {
                        size_s(s) += 1.;
                        ysum_ds.col(s) += y.col(loc);
                        n_bs(b, s) = n_bs(b, s) + 1.;
                    }
                }
                delta_num_db.col(b) += y.rowwise().sum();
            }

            offset += Nb;

            TLOG_(verbose,
                  "processed file set [" << (b + 1) << "] for pseudobulk for "
                                         << Nb << " / " << offset << " cells");
        } // for each batch
    }

    for (Index s = 0; s < S; ++s) {
        prob_bs.col(s) = n_bs.col(s) / n_bs.col(s).sum();
    }

    //////////////////////////////
    // Step 5. Batch adjustment //
    //////////////////////////////

    Mat zsum_ds; // data x PB

    if (B > 1 && do_batch_adj) {

        std::vector<std::shared_ptr<mtx_data_t>> mtx_ptr;
        for (Index b = 0; b < B; ++b) {
            mtx_ptr.emplace_back(
                std::make_shared<mtx_data_t>(mtx_data_t::MTX(mtx_files.at(b)),
                                             mtx_data_t::ROW(row_files.at(b)),
                                             mtx_data_t::IDX(idx_files.at(b))));
            mtx_ptr[b]->relocate_rows(row2pos);
        }

        TLOG_(verbose,
              "Building annoy index using random proj. results "
                  << RD.rows() << " x " << RD.cols());

        const Index rank = RD.cols();

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index b = 0; b < B; ++b) {

            const std::vector<Index> &batch_cells = batch_glob_map.at(b);
            const Index Nb = batch_cells.size();

            Mat Q = Mat::Zero(rank, Nb);

            for (Index loc = 0; loc < Nb; ++loc) {
                const Index glob = batch_cells.at(loc);
                Q.col(loc) = RD.row(glob).transpose();
            }

            CHECK(mtx_ptr[b]->build_index(Q, verbose));
            TLOG_(verbose, "Built annoy index [" << (b + 1) << "]");
        }

        ////////////////////////////
        // Step a. precalculation //
        ////////////////////////////

        TLOG_(verbose, "Collecting sufficient statistics...");

        zsum_ds = Mat::Zero(D, S); // gene x PB counterfactual

        Index Nprocessed = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index s = 0; s < pb_cells.size(); ++s) {

            const std::vector<Index> &_cells_s = pb_cells.at(s);
            std::vector<Scalar> query(rank);

            const std::size_t nneigh_max = (B - 1) * KNN_CELL;
            Mat z_per_cell = Mat::Zero(D, nneigh_max);
            ColVec w_per_cell = ColVec::Zero(nneigh_max);

            std::size_t nneigh = 0;
            std::vector<Scalar> cum_dist(nneigh_max, 0);
            std::vector<Scalar> weights(nneigh_max, 0);

            zsum_ds.col(s).setZero();

            for (Index glob : _cells_s) {

                const Index a = batch_membership.at(glob);

                // For each cell:
                // 1. retrieve matched cells
                // 2. take weighted average of them

                z_per_cell.setZero();
                w_per_cell.setZero();

                nneigh = 0;

                Eigen::Map<Mat>(query.data(), 1, rank) = RD.row(glob);

                for (Index b = 0; b < B; ++b) {
                    if (a != b) {

                        mtx_data_t &mtx = *mtx_ptr[b].get();

                        std::vector<Index> neigh_index;
                        std::vector<Scalar> neigh_dist;

                        const Mat z = mtx.read_matched_reloc(query,
                                                             KNN_CELL,
                                                             neigh_index,
                                                             neigh_dist);

                        for (Index k = 0; k < z.cols(); ++k) {
                            z_per_cell.col(nneigh) = z.col(k);
                            cum_dist[nneigh] = neigh_dist.at(k);
                            ++nneigh;
                        }
                    }
                }

                mmutil::match::normalize_weights(nneigh, cum_dist, weights);

                for (Index k = 0; k < nneigh; ++k) {
                    w_per_cell(k) = weights.at(k);
                }

                zsum_ds.col(s) += z_per_cell * w_per_cell / w_per_cell.sum();
            } // for each glob index

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

        } // for each PB sample s

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
              "Iterative optimization"
                  << " using " << Eigen::nbThreads()
                  << " Eigen library threads");

        Mat gamma_ds = Mat::Ones(D, S); // bias on the side of CF

        delta_db.resize(D, B); // gene x batch
        delta_db.setOnes();

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

    std::vector<Index> r_batch(batch_membership.size());
    convert_r_index(batch_membership, r_batch);

    TLOG_(verbose, "position and batch names");

    std::vector<std::string> d_ = pos2row;
    std::vector<std::string> s_;
    for (std::size_t s = 1; s <= S; ++s)
        s_.push_back(std::to_string(s));
    std::vector<std::string> b_ = mtx_files;

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
                        _["mtx.files"] = mtx_files,
                        _["batch.membership"] = r_batch,
                        _["positions"] = r_positions,
                        _["rand.proj"] = R_kd,
                        _["colnames"] = columns,
                        _["rownames"] = pos2row);
}
