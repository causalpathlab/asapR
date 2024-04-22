#ifndef RCPP_ASAP_PB_CBIND_HH_
#define RCPP_ASAP_PB_CBIND_HH_

#include "rcpp_asap.hh"
#include "rcpp_asap_stat.hh"
#include "rcpp_mtx_data.hh"
#include "rcpp_eigenSparse_data.hh"
#include "rcpp_asap_pb.hh"

template <typename T>
Rcpp::List
run_asap_pb_cbind(std::vector<T> &data_loaders,
                  const std::vector<std::string> &pos2row,
                  const std::vector<std::string> &columns,
                  const asap::pb::options_t &options)
{

    using namespace asap::pb;

    const Index B = data_loaders.size();
    const Index D = pos2row.size();    // dimensionality
    const Index Ntot = columns.size(); // samples

    const Index K = options.K;

    const bool do_batch_adj = options.do_batch_adj;
    const bool do_log1p = options.do_log1p;
    const bool do_down_sample = options.do_down_sample;
    const bool save_aux_data = options.save_aux_data;
    const std::size_t KNN_CELL = options.KNN_CELL;
    const std::size_t CELL_PER_SAMPLE = options.CELL_PER_SAMPLE;
    const std::size_t BATCH_ADJ_ITER = options.BATCH_ADJ_ITER;
    const double a0 = options.a0;
    const double b0 = options.b0;
    const std::size_t rseed = options.rseed;
    const bool verbose = options.verbose;
    const std::size_t NUM_THREADS =
        (options.NUM_THREADS > 0 ? options.NUM_THREADS : omp_get_max_threads());
    const Index block_size = options.BLOCK_SIZE;

    log1p_op<Mat> log1p;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    TLOG_(verbose, D << " x " << Ntot);

    ASSERT_RETL(Ntot > 0, "Ntot must be positive");

    ASSERT_RETL(Ntot > K, "Ntot should be more than K factors");

    {
        Index ntot = 0;
        for (Index b = 0; b < B; ++b) { // each batch
            ntot += data_loaders.at(b).max_col();
        }
        ASSERT_RETL(Ntot == ntot, "|column names| != columns(data)");
    }

    /////////////////////////////////////////////
    // Step 1. sample random projection matrix //
    /////////////////////////////////////////////

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);

    Mat R_kd;
    sample_random_projection(D, K, rseed, R_kd);

    Mat Q_kn = Mat::Zero(K, Ntot);

    std::vector<std::vector<Index>> batch_glob_map;
    std::vector<Index> batch_membership(Ntot);

    TLOG_(verbose, "Collecting random projection data");

    Index offset = 0;

    for (Index b = 0; b < B; ++b) {

        const Index Nb = data_loaders.at(b).max_col();
        batch_glob_map.emplace_back(std::vector<Index> {});

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index lb = 0; lb < Nb; lb += block_size) {

            const Index ub = std::min(Nb, block_size + lb);
            Mat y_dn = do_log1p ?
                data_loaders.at(b).read_reloc(lb, ub).unaryExpr(log1p) :
                data_loaders.at(b).read_reloc(lb, ub);

            Mat temp_kn = R_kd * y_dn;

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

    if (B > 1) {

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

    const std::size_t too_many_columns = 1000;

    Mat vv;

    if (Q_kn.cols() < too_many_columns) {
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

    Mat Q_nk = Q_kn.transpose();           // N x K
    Mat Qstd_nk = standardize_columns(vv); // N x K

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

            const Index Nb = data_loaders.at(b).max_col();

            TLOG_(verbose, Nb << " samples");

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
            for (Index lb = 0; lb < Nb; lb += block_size) {

                const Index ub = std::min(Nb, block_size + lb);
                const Mat y = data_loaders.at(b).read_reloc(lb, ub);

                for (Index loc = 0; loc < (ub - lb); ++loc) {
                    const Index glob = loc + lb + offset;
                    const Index s = positions.at(glob);

                    if (s < NA_POS) {
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

        TLOG_(verbose,
              "Building annoy index using random proj. results "
                  << Qstd_nk.rows() << " x " << Qstd_nk.cols());

        const Index rank = Qstd_nk.cols();

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
              "Estimating matched statics for " << Ntot << " cells across " << S
                                                << " samples...");

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
                         static_cast<std::size_t>((B - 1) * KNN_CELL));
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

                for (Index b = 0; b < B; ++b) {
                    if (a != b) {

                        std::vector<Index> neigh_index;
                        std::vector<Scalar> neigh_dist;

                        const Mat z =
                            data_loaders.at(b).read_matched_reloc(query,
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

                if (nneigh > 1) {
                    mmutil::match::normalize_weights(nneigh, cum_dist, weights);

                    for (Index k = 0; k < nneigh; ++k) {
                        w_per_cell(k) = weights.at(k);
                    }
                    zsum_ds.col(s) +=
                        z_per_cell * w_per_cell / w_per_cell.sum();
                } else {
                    zsum_ds.col(s) += z_per_cell.col(0);
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

    if (!save_aux_data) {
        Qstd_nk.resize(0, 0);
        ysum_ds.resize(0, 0);
        zsum_ds.resize(0, 0);
        delta_num_db.resize(0, 0);
    }

    TLOG_(verbose, "Done");

    return List::create(_["PB"] = named(mu_ds, d_, s_),
                        _["sum"] = named(ysum_ds, d_, s_),
                        _["matched.sum"] = named(zsum_ds, d_, s_),
                        _["sum_db"] = named_rows(delta_num_db, d_),
                        _["size"] = size_s,
                        _["prob_sb"] = named_rows(prob_bs.transpose(), s_),
                        _["size_sb"] = named_rows(n_bs.transpose(), s_),
                        _["batch.effect"] = named_rows(delta_db, d_),
                        _["log.batch.effect"] = named_rows(log_delta_db, d_),
                        _["batch.membership"] = r_batch,
                        _["positions"] = r_positions,
                        _["rand.dict"] = R_kd,
                        _["rand.proj"] = Qstd_nk,
                        _["colnames"] = columns,
                        _["rownames"] = pos2row);
}

#endif
