#ifndef RCPP_ASAP_PB_CBIND_HH_
#define RCPP_ASAP_PB_CBIND_HH_

#include "rcpp_asap.hh"
#include "rcpp_asap_stat.hh"
#include "rcpp_asap_util.hh"
#include "rcpp_mtx_data.hh"
#include "rcpp_eigenSparse_data.hh"
#include "rcpp_asap_pb.hh"

template <typename T>
Rcpp::List
run_asap_pb_cbind(std::vector<T> &data_loaders,
                  const std::vector<std::string> &row_names,
                  const std::vector<std::string> &column_names,
                  const std::vector<std::string> &batch_names,
                  const std::vector<std::string> &control_rows,
                  const asap::pb::options_t &options)
{

    using namespace asap::pb;

    const Index Ndata = data_loaders.size();
    const Index D = row_names.size();       // dimensionality
    const Index Ntot = column_names.size(); // samples

    ASSERT_RETL(Ndata == batch_names.size(),
                "Should have matching batch names");

    const Index K = options.K;

    const Scalar cell_norm = options.CELL_NORM;
    const bool do_batch_adj = options.do_batch_adj;
    const bool do_log1p = options.do_log1p;
    const bool do_down_sample = options.do_down_sample;
    const bool save_aux_data = options.save_aux_data;
    const std::size_t KNN_CELL = options.KNN_CELL;
    const std::size_t CELL_PER_SAMPLE = options.CELL_PER_SAMPLE;
    const std::size_t BATCH_ADJ_ITER = options.BATCH_ADJ_ITER;
    const Scalar a0 = options.a0, b0 = options.b0;
    const std::size_t rseed = options.rseed;
    const bool verbose = options.verbose;
    const std::size_t NUM_THREADS =
        (options.NUM_THREADS > 0 ? options.NUM_THREADS : omp_get_max_threads());

    const std::size_t min_control_features = options.MIN_CONTROL_FEATURES;

    const bool do_outlier_qc = options.do_outlier_qc;
    const Scalar q_min = options.qc_q_min, q_max = options.qc_q_max;

    const Index block_size = options.BLOCK_SIZE;

    log1p_op<Mat> log1p;
    at_least_one_op<Mat> at_least_one;

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    TLOG_(verbose, D << " x " << Ntot);

    ASSERT_RETL(Ntot > 0, "Ntot must be positive");

    ASSERT_RETL(Ntot > K, "Ntot should be more than K factors");

    {
        Index ntot = 0;
        for (Index b = 0; b < Ndata; ++b) { // each batch
            ntot += data_loaders.at(b).max_col();
        }
        ASSERT_RETL(Ntot == ntot, "|column names| != columns(data)");
    }

    ColVec neg_features(D), pos_features(D);
    if (control_rows.size() > 0) {
        std::unordered_map<std::string, Index> controls;
        make_position_dict(control_rows, controls);
        neg_features = ColVec::Zero(D);
        pos_features = ColVec::Ones(D);
        for (std::size_t d = 0; d < D; ++d) {
            if (controls.count(row_names.at(d)) > 0) {
                neg_features(d) = 1.;
                pos_features(d) = 0.;
            }
        }
        if (neg_features.sum() < min_control_features) {
            WLOG("Found too little control features overlap with the rows");
            WLOG(D << " features will be considered control features.");
            neg_features.setOnes();
        }
    } else {
        // we will ignore
        neg_features.setOnes();
        pos_features.setOnes();
    }

    /////////////////////////////////////////////
    // Step 1. sample random projection matrix //
    /////////////////////////////////////////////

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);

    Mat R_kd;
    sample_random_projection(D, K, rseed, R_kd);

    Mat Q_kn = Mat::Zero(K, Ntot);
    Mat X_nk; // negative control

    if (control_rows.size() > 0 && neg_features.sum() > 0) {
        X_nk.resize(K, Ntot);
        X_nk.setZero();
    }

    std::vector<std::vector<Index>> batch_glob_map;
    std::vector<Index> batch_membership(Ntot);

    TLOG_(verbose, "Collecting random projection data");

    Index offset = 0;

    for (Index b = 0; b < Ndata; ++b) {

        const Index Nb = data_loaders.at(b).max_col();
        batch_glob_map.emplace_back(std::vector<Index> {});

        Mat Q_kn_b = Mat::Zero(K, Nb);
        Mat Q0_kn_b = Mat::Zero(K, Nb);

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index lb = 0; lb < Nb; lb += block_size) {

            const Index ub = std::min(Nb, block_size + lb);

            Mat y_dn = do_log1p ?
                data_loaders.at(b).read_reloc(lb, ub).unaryExpr(log1p) :
                data_loaders.at(b).read_reloc(lb, ub);

            if (do_outlier_qc) {
                auto [q1, q2] = quantile(y_dn, q_min, q_max);
                clamp_op<Mat> clamp(q1, q2);
                y_dn.noalias() = y_dn.unaryExpr(clamp);
            }

            const ColVec denom = y_dn.transpose()
                                     .rowwise()
                                     .sum()
                                     .unaryExpr(at_least_one)
                                     .cwiseInverse();

#pragma omp critical
            {
                Q_kn_b.middleCols(lb, y_dn.cols()) =
                    (R_kd * pos_features.asDiagonal() * y_dn *
                     denom.asDiagonal());

                Q0_kn_b.middleCols(lb, y_dn.cols()) =
                    (R_kd * neg_features.asDiagonal() * y_dn *
                     denom.asDiagonal());

                for (Index loc = 0; loc < y_dn.cols(); ++loc) {
                    const Index glob = loc + lb + offset;
                    batch_glob_map[b].emplace_back(glob);
                    batch_membership[glob] = b;
                }
            }
        }

        if (control_rows.size() > 0 && neg_features.sum() > 0) {
            X_nk.middleRows(offset, Nb) = Q0_kn_b.transpose();
        }

        Q_kn.middleCols(offset, Nb) = Q_kn_b;

        offset += Nb;

        TLOG_(verbose,
              "Random proj file set [" << (b + 1) << "] of " << Nb << " / "
                                       << offset << " cells");
    } // batch

    if (control_rows.size() > 0 && neg_features.sum() > 0) {
        Mat Y_nk = Q_kn.transpose();
        residual_columns_inplace(Y_nk, X_nk);
        Q_kn = Y_nk.transpose();
        TLOG_(verbose, "removed control feature effects");
    }

    /////////////////////////////////////////////////
    // Step 2. Orthogonalize the projection matrix //
    /////////////////////////////////////////////////

    Mat Qstd_nk;
    {
        const std::size_t lu_iter = 5;
        RandomizedSVD<Mat> svd(Q_kn.rows(), lu_iter);
        svd.compute(Q_kn);
        Qstd_nk = svd.matrixV();
    }

    ASSERT_RETL(Qstd_nk.rows() == Ntot, " failed SVD for Q");
    standardize_columns_inplace(Qstd_nk);

    TLOG_(verbose,
          "SVD on the projected: " << Qstd_nk.rows() << " x "
                                   << Qstd_nk.cols());

    /////////////////////////////////////////////////////
    // Step 3. sorting through an implicit binary tree //
    /////////////////////////////////////////////////////

    IntVec bb(Ntot);
    bb.setZero();

    for (Index k = 0; k < K; ++k) {
        auto binary_shift = [&k](const Scalar &x) -> Index {
            return x > 0. ? (1 << k) : 0;
        };
        bb += Qstd_nk.col(k).unaryExpr(binary_shift);
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
    Mat mu_cf_ds;
    Mat ysum_ds = Mat::Zero(D, S);
    RowVec size_s = RowVec::Zero(S);

    Mat delta_num_db = Mat::Zero(D, Ndata);   // gene x batch numerator
    Mat delta_denom_db = Mat::Zero(D, Ndata); // gene x batch denominator
    Mat n_bs = Mat::Zero(Ndata, S);           // batch x PB freq
    Mat prob_bs = Mat::Zero(Ndata, S);        // batch x PB prob

    Mat delta_db, log_delta_db;

    {
        offset = 0;
        delta_num_db.setZero();

        for (Index b = 0; b < Ndata; ++b) { // each batch

            const Index Nb = data_loaders.at(b).max_col();

            TLOG_(verbose, Nb << " samples");

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
            for (Index lb = 0; lb < Nb; lb += block_size) {

                const Index ub = std::min(Nb, block_size + lb);

                Mat y = data_loaders.at(b).read_reloc(lb, ub);
                normalize_columns_inplace(y);
                y *= cell_norm;

                Mat z = neg_features.asDiagonal() * y;

#pragma omp critical
                {
                    for (Index loc = 0; loc < (ub - lb); ++loc) {
                        const Index glob = loc + lb + offset;
                        const Index s = positions.at(glob);

                        if (s < NA_POS) {
                            size_s(s) += 1.;
                            ysum_ds.col(s) += y.col(loc);
                            n_bs(b, s) = n_bs(b, s) + 1.;
                        }
                    }

                    delta_num_db.col(b) += z.rowwise().sum();
                }
            }

            offset += Nb;

            TLOG_(verbose,
                  "Obs data [" << (b + 1) << "] -> PB " << Nb << " / " << offset
                               << " cells");
        } // for each batch
    }

    for (Index s = 0; s < S; ++s) {
        prob_bs.col(s) = n_bs.col(s) / n_bs.col(s).sum();
    }

    //////////////////////////////
    // Step 5. Batch adjustment //
    //////////////////////////////

    Mat zsum_ds; // data x PB

    if (Ndata > 1 && do_batch_adj) {

        TLOG_(verbose,
              "Building annoy index using random proj. results "
                  << Qstd_nk.rows() << " x " << Qstd_nk.cols());

        const Index rank = Qstd_nk.cols();

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index b = 0; b < Ndata; ++b) {

            const std::vector<Index> &batch_cells = batch_glob_map.at(b);
            const Index Nb = batch_cells.size();

            Mat Q = Mat::Zero(rank, Nb);

            for (Index loc = 0; loc < Nb; ++loc) {
                const Index glob = batch_cells.at(loc);
                Q.col(loc) = Qstd_nk.row(glob).transpose();
            }

            CHECK(data_loaders.at(b).build_index(Q, verbose));
            TLOG_(verbose, "Built annoy index [" << (b + 1) << "]");
        }

        ////////////////////////////
        // Step a. precalculation //
        ////////////////////////////
        Index Ntot = 0;
        for (Index s = 0; s < pb_cells.size(); ++s) {
            const std::vector<Index> &_cells_s = pb_cells.at(s);
            Ntot += _cells_s.size();
        }

        TLOG_(verbose,
              "Matching " << Ntot << " cells across " << S << " samples...");

        zsum_ds = Mat::Zero(D, S); // gene x PB counterfactual

        Index Nprocessed = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index s = 0; s < pb_cells.size(); ++s) {

            const std::vector<Index> &_cells_s = pb_cells.at(s);
            std::vector<Scalar> query(rank);

            const std::size_t nneigh_max =
                std::max(static_cast<std::size_t>(1),
                         static_cast<std::size_t>((Ndata - 1) * KNN_CELL));
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

                Eigen::Map<Mat>(query.data(), 1, rank) = Qstd_nk.row(glob);

                for (Index b = 0; b < Ndata; ++b) {
                    if (a != b) {

                        std::vector<Index> neigh_index;
                        std::vector<Scalar> neigh_dist;

                        const Mat z =
                            (neg_features.asDiagonal() *
                             data_loaders.at(b).read_matched_reloc(query,
                                                                   KNN_CELL,
                                                                   neigh_index,
                                                                   neigh_dist));

                        for (Index k = 0; k < z.cols(); ++k) {
                            z_per_cell.col(nneigh) = z.col(k);
                            cum_dist[nneigh] = neigh_dist.at(k);
                            ++nneigh;
                        }
                    }
                } // each batch

                if (nneigh > 1) {
                    mmutil::match::normalize_weights(nneigh, cum_dist, weights);

                    for (Index k = 0; k < nneigh; ++k) {
                        w_per_cell(k) = weights.at(k);
                    }
#pragma omp critical
                    {
                        zsum_ds.col(s) +=
                            z_per_cell * w_per_cell / w_per_cell.sum();
                    }
                } else {
#pragma omp critical
                    {
                        zsum_ds.col(s) += z_per_cell.col(0);
                    }
                }
            } // for each glob index
#pragma omp critical
            {
                Nprocessed += 1;
                if (verbose) {
                    Rcpp::Rcerr << "\rProcessed: " << Nprocessed
                                << " PB sample(s)" << std::flush;
                } else {
                    Rcpp::Rcerr << "+ " << std::flush;
                    if (Nprocessed % 100 == 0)
                        Rcpp::Rcerr << "\r" << std::flush;
                }
            }

        } // for each PB sample s

        Rcpp::Rcerr << std::endl;
        TLOG_(verbose, "Collected sufficient statistics");

        gamma_param_t<Mat, RNG> delta_param(D, Ndata, a0, b0, rng);
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

        mu_cf_ds.resize(D, S); // bias on the side of CF
        mu_cf_ds.setOnes();

        delta_db.resize(D, Ndata); // gene x batch
        delta_db.setOnes();

        for (std::size_t t = 0; t < BATCH_ADJ_ITER; ++t) {
            ////////////////////////
            // shared components  //
            ////////////////////////
            mu_param.update(ysum_ds + zsum_ds,
                            delta_db * n_bs + mu_cf_ds * size_s.asDiagonal());
            mu_param.calibrate();
            mu_ds = mu_param.mean();

            ////////////////////
            // residual for z //
            ////////////////////

            gamma_param.update(zsum_ds, mu_ds * size_s.asDiagonal());
            gamma_param.calibrate();
            mu_cf_ds = gamma_param.mean();

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

    std::vector<std::string> d_ = row_names;
    std::vector<std::string> s_;
    for (std::size_t s = 1; s <= S; ++s)
        s_.push_back(std::to_string(s));

    Mat mean_ds;

    if (!save_aux_data) {
        Qstd_nk.resize(0, 0);
        ysum_ds.resize(0, 0);
        zsum_ds.resize(0, 0);
        delta_num_db.resize(0, 0);
    } else if (do_batch_adj) {
        mean_ds.resize(D, S);
        gamma_param_t<Mat, RNG> mean_param(D, S, a0, b0, rng);
        Mat temp_ds = Mat::Ones(D, S).array().rowwise() * size_s.array();
        mean_param.update(ysum_ds, temp_ds);
        mean_param.calibrate();
        mean_ds = mean_param.mean();
    }

    TLOG_(verbose, "Done");

    return List::create(_["PB"] = named(mu_ds, d_, s_),
                        _["CF"] = named(mu_cf_ds, d_, s_),
                        _["mean"] = named(mean_ds, d_, s_),
                        _["sum"] = named(ysum_ds, d_, s_),
                        _["matched.sum"] = named(zsum_ds, d_, s_),
                        _["sum_db"] = named_rows(delta_num_db, d_),
                        _["size"] = size_s,
                        _["prob_sb"] = named_rows(prob_bs.transpose(), s_),
                        _["size_sb"] = named_rows(n_bs.transpose(), s_),
                        _["batch.effect"] = named(delta_db, d_, batch_names),
                        _["log.batch.effect"] =
                            named(log_delta_db, d_, batch_names),
                        _["batch.membership"] = r_batch,
                        _["positions"] = r_positions,
                        _["rand.dict"] = R_kd,
                        _["rand.proj"] = Qstd_nk,
                        _["colnames"] = column_names,
                        _["rownames"] = row_names);
}

#endif
