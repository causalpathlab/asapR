#include "rcpp_asap_bbknn.hh"

//' Reconcile multi-batch matrices by batch-balancing KNN
//'
//' @param data_nk_vec a list of sample x factor matrices
//' @param row_names_vec a list of sample x 1 names
//' @param KNN_PER_BATCH (default: 3)
//' @param BLOCK_SIZE each parallel job size (default: 100)
//' @param NUM_THREADS number of parallel threads (default: 1)
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

    Mat V_kn(rank, Ntot);

    std::vector<std::vector<Index>> global_index;
    std::vector<Index> batches(Ntot);
    {
        V_kn.setZero();
        Index offset = 0;
        for (std::size_t b = 0; b < data_nk_vec.size(); ++b) {
            const Eigen::MatrixXf &data_nk = data_nk_vec[b];
            const std::size_t rank = data_nk.cols();
            global_index.emplace_back(std::vector<Index> {});

            std::vector<Index> &globs = global_index[b];
            globs.reserve(data_nk.rows());

            Mat vsub_kn = data_nk.transpose();
            for (Index j = 0; j < vsub_kn.cols(); ++j) {
                const Index glob = offset + j;
                batches[glob] = b;
                V_kn.col(glob) = vsub_kn.col(j);
                globs.emplace_back(glob);
            }

            TLOG_(verbose, "Populated " << vsub_kn.cols() << " items");
            offset += vsub_kn.cols();
        }
    }

    using index_ptr = std::shared_ptr<annoy_index_t>;
    std::vector<index_ptr> idx_ptr_vec;
    idx_ptr_vec.reserve(B);
    {
        using namespace asap::bbknn;
        for (std::size_t b = 0; b < data_nk_vec.size(); ++b) {
            idx_ptr_vec.emplace_back(
                build_euclidean_annoy(data_nk_vec.at(b), NUM_THREADS));
        }
    }

    TLOG("Built " << idx_ptr_vec.size() << " ANNOY indexes for fast look-ups");

    ///////////////////////////////////////////////////
    // step 2. build mutual kNN graph across batches //
    ///////////////////////////////////////////////////

    asap::bbknn::options_t options;
    options.block_size = BLOCK_SIZE;
    options.num_threads = NUM_THREADS;
    options.knn_per_batch = KNN_PER_BATCH;

    SpMat Wsym;
    {
        std::vector<std::tuple<Index, Index, Scalar>> knn_raw;
        ASSERT_RETL(asap::bbknn::build_knn(V_kn,
                                           global_index,
                                           idx_ptr_vec,
                                           options,
                                           knn_raw),
                    "Failed to build the k-MN backbone");

        auto knn_rec = mmutil::match::keep_reciprocal_knn(knn_raw);
        SpMat W = build_eigen_sparse(knn_rec, Ntot, Ntot);
        SpMat Wt = W.transpose();
        Wsym = (W + Wt) * 0.5;
    }
    TLOG_(verbose, "Constructed BB-kNN graph");

    /////////////////////////////////////
    // step 3. Batch effect adjustment //
    /////////////////////////////////////

    Mat Vadj = V_kn;

    for (Index b = 1; b < B; ++b) {
        std::vector<Index> &globs = global_index[b];

        // TODO: group-wise adjustment

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index j : globs) {

            ColVec delta = ColVec::Zero(rank);
            Scalar denom = 0;

            for (SpMat::InnerIterator it(Wsym, j); it; ++it) {
                const Index i = it.index();        // other index
                const Index a = batches.at(i);     // batch membership
                if (a < b) {                       // mingle toward
                    const Scalar wji = it.value(); // the previous batches
                    delta += (V_kn.col(j) - Vadj.col(i)) * wji;
                    denom += wji;
                }
            }

            if (denom > 1e-8) {
                delta /= denom;
            }

#pragma omp critical
            {
                ////////////////////////////////
                // May create over-adjustment //
                ////////////////////////////////

                Vadj.col(j) = V_kn.col(j) - delta;
            }
        }

    } // for each batch

    TLOG_(verbose, "Successfully adjusted weights");

    Vadj.transposeInPlace();

    SpMat Wout;
    {
        TLOG_(verbose, "Recalibrate kNN graph");
        Mat data_nk = Vadj;
        auto idx_ptr = asap::bbknn::build_euclidean_annoy(data_nk, NUM_THREADS);

        std::vector<std::tuple<Index, Index, Scalar>> knn_raw;
        Mat data_kn = data_nk.transpose();
        build_knn(data_kn, idx_ptr, options, knn_raw);

        SpMat W = build_eigen_sparse(knn_raw, Ntot, Ntot);
        SpMat Wt = W.transpose();
        Wout = (W + Wt) * 0.5;
    }

    using namespace rcpp::util;
    using namespace Rcpp;

    std::vector<std::vector<Index>> r_glob_idx;
    for (Index j = 0; j < global_index.size(); ++j) {
        r_glob_idx.emplace_back(convert_r_index(global_index.at(j)));
    }

    return List::create(_["adjusted"] = named_rows(Vadj, global_names),
                        _["names"] = global_names,
                        _["indexes"] = r_glob_idx,
                        _["batches"] = convert_r_index(batches),
                        _["knn"] = build_sparse_list(Wsym),
                        _["knn.adj"] = build_sparse_list(Wout));
}
