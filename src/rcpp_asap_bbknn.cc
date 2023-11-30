#include "rcpp_asap_bbknn.hh"

//' Reconcile multi-batch matrices by batch-balancing KNN
//'
//' @param data_nk_vec a list of sample x factor matrices
//' @param row_names_vec a list of sample x 1 names
//' @param KNN_PER_BATCH (default: 3)
//' @param BLOCK_SIZE each parallel job size (default: 100)
//' @param NUM_THREADS number of parallel threads (default: 1)
//' @param IP_DISTANCE inner product distance (default: FALSE)
//' @param verbose (default: TRUE)
//'
//' @return a list that contains:
//' \itemize{
//'  \item adjusted (N x K) matrix
//'  \item bbknn batch-balanced kNN adjacency matrix
//'  \item batches batch membership
//'  \item knn edges
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_bbknn(const std::vector<Eigen::MatrixXf> &data_nk_vec,
           const std::vector<std::vector<std::string>> &row_names_vec,
           const std::size_t KNN_PER_BATCH = 3,
           const std::size_t BLOCK_SIZE = 100,
           const std::size_t NUM_THREADS = 1,
           const bool IP_DISTANCE = false,
           const bool verbose = true)
{

    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    std::size_t rank_, b_ = 0, n_ = 0;

    ASSERT_RETL(data_nk_vec.size() == row_names_vec.size(),
                "We need row names for each data_nk");

    std::vector<std::string> global_names;

    for (std::size_t b = 0; b < data_nk_vec.size(); ++b) {
        const Eigen::MatrixXf &data_nk = data_nk_vec[b];
        const std::vector<std::string> &row_names = row_names_vec[b];
        if (b_ == 0) {
            rank_ = data_nk.cols();
        } else {
            ASSERT_RETL(rank_ == data_nk.cols(),
                        "ranks don't match across data");
        }

        ASSERT_RETL(data_nk.rows() == row_names.size(),
                    "need a name for each row");

        n_ += data_nk.rows();
        for (auto r : row_names) {
            global_names.emplace_back(r);
        }
        ++b_;
    }

    ASSERT_RETL(rank_ > 0, "zero rank data");
    ASSERT_RETL(b_ > 0 && n_ > 0, "empty data");

    const std::size_t rank = rank_;
    const std::size_t B = data_nk_vec.size();
    const std::size_t Ntot = n_;

    using annoy_index_t = Annoy::AnnoyIndex<Index,
                                            Scalar,
                                            Annoy::Euclidean,
                                            Kiss64Random,
                                            RcppAnnoyIndexThreadPolicy>;

    /////////////////////////////////
    // step 1. Build annoy indexes //
    /////////////////////////////////

    using index_ptr = std::shared_ptr<annoy_index_t>;
    std::vector<index_ptr> idx_ptr_vec;
    idx_ptr_vec.reserve(B);

    Mat V_kn(rank, Ntot);

    std::vector<std::vector<Index>> global_index;
    std::vector<Index> batches(Ntot);
    {
        V_kn.setZero();
        Index offset = 0;
        for (std::size_t b = 0; b < data_nk_vec.size(); ++b) {
            const Eigen::MatrixXf &data_nk = data_nk_vec[b];

            const std::size_t rank = data_nk.cols();
            idx_ptr_vec.emplace_back(std::make_shared<annoy_index_t>(rank));
            global_index.emplace_back(std::vector<Index> {});

            annoy_index_t &index = *idx_ptr_vec[b].get();
            std::vector<Index> &globs = global_index[b];
            globs.reserve(data_nk.rows());

            Mat vsub_kn = data_nk.transpose();
            if (IP_DISTANCE) {
                normalize_columns_inplace(vsub_kn);
            }
            std::vector<Scalar> vec(rank);

            for (Index j = 0; j < vsub_kn.cols(); ++j) {
                const Index glob = offset + j;
                batches[glob] = b;
                V_kn.col(glob) = vsub_kn.col(j);
                globs.emplace_back(glob);
                Eigen::Map<Mat>(vec.data(), rank, 1) = vsub_kn.col(j);
                index.add_item(j, vec.data()); // allocate the size up to j
            }

            index.build(50);
            TLOG_(verbose, "Populated " << vsub_kn.cols() << " items");
            offset += vsub_kn.cols();
        }
    }
    TLOG("Built " << idx_ptr_vec.size() << " ANNOY indexes for fast look-ups");

    ///////////////////////////////////////////////////
    // step 2. build mutual kNN graph across batches //
    ///////////////////////////////////////////////////

    using RNG = dqrng::xoshiro256plus;
    RNG rng;

    SpMat Wsym;

    // 2a. Randomly distribute Ntot indexes
    // 2b. For each bucket accumulate backbone
    {
        std::vector<Index> _jobs(Ntot);
        std::iota(_jobs.begin(), _jobs.end(), 0);
        std::for_each(_jobs.begin(), _jobs.end(), [&BLOCK_SIZE](auto &x) {
            return x / BLOCK_SIZE;
        });

        std::vector<std::vector<Index>> job_vec = make_index_vec_vec(_jobs);
        std::vector<std::tuple<Index, Index, Scalar>> backbone_raw;

        Index Nprocessed = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index job = 0; job < job_vec.size(); ++job) {
            const std::vector<Index> &_cells = job_vec.at(job);

            std::vector<Scalar> query(rank);

            for (Index glob_i : _cells) {
                Eigen::Map<Mat>(query.data(), rank, 1) = V_kn.col(glob_i);

                std::vector<Scalar> dist_ij;
                std::vector<Scalar> weight_ij;
                std::vector<Index> pair_ij;

                // for each batch
                for (Index b = 0; b < B; ++b) {
                    annoy_index_t &index = *idx_ptr_vec[b].get();
                    std::vector<Index> &globs = global_index[b];

                    const std::size_t nsearch =
                        std::min(KNN_PER_BATCH, globs.size());
                    std::vector<Index> neigh;
                    std::vector<Scalar> dist;

                    index.get_nns_by_vector(query.data(),
                                            nsearch,
                                            -1,
                                            &neigh,
                                            &dist);
                    for (std::size_t i = 0; i < nsearch; ++i) {
                        const Scalar d = dist.at(i);
                        const Index j = neigh.at(i);
                        const Index glob_j = globs.at(j);

                        if (glob_i != glob_j) {
                            pair_ij.emplace_back(glob_j);
                            dist_ij.emplace_back(d);
                            weight_ij.emplace_back(0);
                        }
                    }
                } // batches

                const std::size_t deg_ij = pair_ij.size();
                mmutil::match::normalize_weights(deg_ij, dist_ij, weight_ij);

#pragma omp critical
                {
                    for (std::size_t k = 0; k < deg_ij; ++k) {
                        backbone_raw.emplace_back(glob_i,
                                                  pair_ij.at(k),
                                                  weight_ij.at(k));
                    }
                }
            }
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

        } // jobs

        auto backbone_rec = mmutil::match::keep_reciprocal_knn(backbone_raw);
        SpMat W = build_eigen_sparse(backbone_rec, Ntot, Ntot);
        SpMat Wt = W.transpose();
        Wsym = (W + Wt) * 0.5;

    } // end of Wsym

    TLOG_(verbose, "Constructed BB-kNN graph");

    /////////////////////////////////////
    // step 3. Batch effect adjustment //
    /////////////////////////////////////

    Mat Vadj = V_kn;

    for (Index b = 1; b < B; ++b) {
        std::vector<Index> &globs = global_index[b];

        ColVec delta = ColVec::Zero(rank);
        Scalar denom = 0;

        for (Index j : globs) {
            for (SpMat::InnerIterator it(Wsym, j); it; ++it) {
                const Index i = it.col();          // row major
                const Index a = batches.at(i);     // batch membership
                if (a < b) {                       // mingle toward
                    const Scalar wji = it.value(); // the previous batches
                    delta += (V_kn.col(j) - Vadj.col(i)) * wji;
                    denom += wji;
                }
            }
        }

        if (denom > 1e-8) {
            delta /= denom;
        }

        for (Index j : globs) {
            Vadj.col(j) = V_kn.col(j) - delta;
        }

    } // for each batch

    TLOG_(verbose, "Successfully adjusted weights");

    Vadj.transposeInPlace();

    using namespace rcpp::util;
    using namespace Rcpp;

    List knn_list = build_sparse_list(Wsym);

    std::vector<std::vector<Index>> r_glob_idx;
    for (Index j = 0; j < global_index.size(); ++j) {
        r_glob_idx.emplace_back(convert_r_index(global_index.at(j)));
    }

    return List::create(_["adjusted"] = named_rows(Vadj, global_names),
                        _["names"] = global_names,
                        _["indexes"] = r_glob_idx,
                        _["batches"] = convert_r_index(batches),
                        _["knn"] = knn_list);
}
