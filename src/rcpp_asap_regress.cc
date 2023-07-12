#include "rcpp_asap_regress.hh"

//' Reconcile multi-batch matrices by batch-balancing KNN
//'
//' @param data_nk_vec a list of sample x factor matrices
//' @param KNN_PER_BATCH (default: 10)
//' @param BLOCK_SIZE each parallel job size (default: 100)
//' @param NUM_THREADS number of parallel threads (default: 1)
//' @param verbose (default: TRUE)
//'
//' @return a list that contains:
//' \itemize{
//'  \item adjusted (N x K) matrix
//'  \item bbknn batch-balanced kNN adjacency matrix
//'  \item batches batch membership
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_adjust_bbknn(const Rcpp::List &data_nk_vec,
                  const std::size_t KNN_PER_BATCH = 10,
                  const std::size_t BLOCK_SIZE = 100,
                  const std::size_t NUM_THREADS = 1,
                  const bool verbose = true)
{

    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    std::size_t rank_, b_ = 0, n_ = 0;

    for (const Eigen::MatrixXf &data_nk : data_nk_vec) {
        if (b_ == 0) {
            rank_ = data_nk.cols();
        } else {
            ASSERT_RETL(rank_ == data_nk.cols(),
                        "ranks don't match across data");
        }
        n_ += data_nk.rows();
        ++b_;
    }

    ASSERT_RETL(rank_ > 0, "zero rank data");
    ASSERT_RETL(b_ > 0 && n_ > 0, "empty data");

    const std::size_t rank = rank_;
    const std::size_t B = data_nk_vec.size();
    const std::size_t Ntot = n_;

    using annoy_index_t = AnnoyIndex<Index,
                                     Scalar,
                                     Euclidean,
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
        for (const Eigen::MatrixXf &data_nk : data_nk_vec) {
            const std::size_t rank = data_nk.cols();
            idx_ptr_vec.emplace_back(std::make_shared<annoy_index_t>(rank));
            global_index.emplace_back(std::vector<Index> {});

            const std::size_t b = idx_ptr_vec.size() - 1;
            annoy_index_t &index = *idx_ptr_vec[b].get();
            std::vector<Index> &globs = global_index[b];
            globs.reserve(data_nk.rows());

            Mat vsub_kn = data_nk.transpose();
            normalize_columns_inplace(vsub_kn);
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

    // 2a. Randomly distribute Ntot indexes
    // 2b. For each bucket accumulate backbone

    std::vector<Index> _jobs(Ntot);
    std::iota(_jobs.begin(), _jobs.end(), 0);
    std::for_each(_jobs.begin(), _jobs.end(), [&BLOCK_SIZE](auto &x) {
        return x / BLOCK_SIZE;
    });

    std::vector<std::vector<Index>> job_vec = make_index_vec_vec(_jobs);

    std::vector<std::tuple<Index, Index, Scalar>> backbone;

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
                    backbone.emplace_back(glob_i,
                                          pair_ij.at(k),
                                          weight_ij.at(k));
                }
            }
        }
    } // jobs

    SpMat W = build_eigen_sparse(backbone, Ntot, Ntot);

    TLOG_(verbose, "Constructed BB-kNN graph");

    /////////////////////////////////////
    // step 3. Batch effect adjustment //
    /////////////////////////////////////

    Mat Vadj = V_kn;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index b = 1; b < B; ++b) {
        std::vector<Index> &globs = global_index[b];

        ColVec delta = ColVec::Zero(rank);
        Scalar denom = 0;

        for (Index j : globs) {
            for (SpMat::InnerIterator it(W, j); it; ++it) {
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

    TLOG("Successfully adjusted weights");

    Vadj.transposeInPlace();

    return Rcpp::List::create(Rcpp::_["adjusted"] = Vadj,
                              Rcpp::_["bbknn"] = W,
                              Rcpp::_["batches"] = batches);
}

//' Poisson regression to estimate factor loading
//'
//' @param Y D x N data matrix
//' @param log_x D x K log dictionary/design matrix
//' @param a0 gamma(a0, b0) (default: 1e-8)
//' @param b0 gamma(a0, b0) (default: 1)
//' @param do_log1p do log(1+y) transformation (default: FALSE)
//' @param verbose verbosity (default: false)
//'
// [[Rcpp::export]]
Rcpp::List
asap_regression(const Eigen::MatrixXf Y_,
                const Eigen::MatrixXf log_x,
                const double a0 = 1e-8,
                const double b0 = 1.0,
                const std::size_t max_iter = 10,
                const bool do_log1p = false,
                const bool verbose = true)
{

    exp_op<Mat> exp;
    at_least_one_op<Mat> at_least_one;
    softmax_op_t<Mat> softmax;
    log1p_op<Mat> log1p;

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    Mat Y_dn = do_log1p ? Y_.unaryExpr(log1p) : Y_;

    const Index D = Y_dn.rows();  // number of features
    const Index N = Y_dn.cols();  // number of samples
    const Index K = log_x.cols(); // number of topics

    Mat logX_dk = log_x;

    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Mat, RNG>;
    RNG rng;

    Mat logRho_nk(Y_dn.cols(), K), rho_nk(Y_dn.cols(), K);

    gamma_t theta_nk(Y_dn.cols(), K, a0, b0, rng); // N x K

    const ColVec Y_n = Y_dn.colwise().sum().transpose();
    const ColVec Y_n1 = Y_n.unaryExpr(at_least_one);

    // X[j,k] = sum_i X[i,k]
    RowVec Xsum = logX_dk.unaryExpr(exp).colwise().sum();
    Mat x_nk = ColVec::Ones(N) * Xsum; // n x K
    Mat R_nk = (Y_dn.transpose() * logX_dk).array().colwise() / Y_n1.array();

    if (verbose)
        TLOG("Correlation with the topics");

    RowVec tempK(K);

    for (std::size_t t = 0; t < max_iter; ++t) {

        logRho_nk = R_nk + theta_nk.log_mean();

        for (Index jj = 0; jj < Y_dn.cols(); ++jj) {
            tempK = logRho_nk.row(jj);
            logRho_nk.row(jj) = softmax.log_row(tempK);
        }
        rho_nk = logRho_nk.unaryExpr(exp);

        theta_nk.update((rho_nk.array().colwise() * Y_n.array()).matrix(),
                        x_nk);
        theta_nk.calibrate();
    }

    if (verbose)
        TLOG("Calibrated topic portions");

    return Rcpp::List::create(Rcpp::_["beta"] = logX_dk.unaryExpr(exp),
                              Rcpp::_["theta"] = theta_nk.mean(),
                              Rcpp::_["log.theta"] = theta_nk.log_mean(),
                              Rcpp::_["log.theta.sd"] = theta_nk.log_sd(),
                              Rcpp::_["corr"] = R_nk);
}

//' Topic statistics to estimate factor loading
//'
//' @param mtx_file matrix-market-formatted data file (D x N, bgzip)
//' @param row_file row names file (D x 1)
//' @param col_file column names file (N x 1)
//' @param idx_file matrix-market colum index file
//' @param log_x D x K log dictionary/design matrix
//' @param x_row_names row names log_x (D vector)
//' @param do_log1p do log(1+y) transformation
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param BLOCK_SIZE disk I/O block size (number of columns)
//'
//' @return a list that contains:
//' \itemize{
//'  \item beta dictionary matrix (row x factor)
//'  \item corr empirical correlation (column x factor)
//'  \item colsum the sum of each column (column x 1)
//' }
//'
// [[Rcpp::export]]
Rcpp::List
asap_topic_stat(const std::string mtx_file,
                const std::string row_file,
                const std::string col_file,
                const std::string idx_file,
                const Eigen::MatrixXf log_x,
                const Rcpp::StringVector &x_row_names,
                const bool do_log1p = false,
                const bool verbose = false,
                const std::size_t NUM_THREADS = 1,
                const std::size_t BLOCK_SIZE = 100)
{

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    std::vector<std::string> pos2row;

    rcpp::util::copy(x_row_names, pos2row);

    //////////////////////////////////////
    // take care of different row names //
    //////////////////////////////////////

    const Index D = pos2row.size(); // dimensionality

    ASSERT_RETL(log_x.rows() == D,
                "#Rows in the log_x matrix !=  the size of x_row_names: "
                    << log_x.rows() << " != " << D);

    std::unordered_map<std::string, Index> row2pos;
    for (Index r = 0; r < pos2row.size(); ++r) {
        row2pos[pos2row.at(r)] = r;
    }

    ASSERT_RETL(row2pos.size() == D, "Redundant row names exist");
    TLOG_(verbose, "Found " << row2pos.size() << " unique row names");

    ///////////////////////////////
    // read mtx data information //
    ///////////////////////////////

    CHK_RETL_(convert_bgzip(mtx_file),
              "mtx file " << mtx_file << " was not bgzipped.");

    mm_info_reader_t info;
    CHK_RETL_(peek_bgzf_header(mtx_file, info),
              "Failed to read the size of this mtx file:" << mtx_file);

    const Index N = info.max_col;        // number of cells
    const Index K = log_x.cols();        // number of topics
    const Index block_size = BLOCK_SIZE; // memory block size

    std::vector<std::string> coln;
    CHK_RETL_(read_vector_file(col_file, coln),
              "Failed to read the column name file: " << col_file);

    ASSERT_RETL(N == coln.size(),
                "Different #columns: " << N << " vs. " << coln.size());

    mtx_data_t data(mtx_data_t::MTX { mtx_file },
                    mtx_data_t::ROW { row_file },
                    mtx_data_t::IDX { idx_file });

    Mat logX_dk = log_x;
    TLOG_(verbose, "lnX: " << logX_dk.rows() << " x " << logX_dk.cols());

    Mat Rtot_nk(N, K);
    Mat Ytot_n(N, 1);
    Index Nprocessed = 0;

    if (verbose) {
        Rcpp::Rcerr << "Calibrating " << N << " columns..." << std::endl;
    }

    data.relocate_rows(row2pos);

    at_least_one_op<Mat> at_least_one;
    at_least_zero_op<Mat> at_least_zero;
    exp_op<Mat> exp;
    log1p_op<Mat> log1p;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {
        Index ub = std::min(N, block_size + lb);
        const SpMat y = data.read_reloc(lb, ub);

        ///////////////////////////////////////
        // do log1p transformation if needed //
        ///////////////////////////////////////

        Mat y_dn = do_log1p ? y.unaryExpr(log1p) : y;

        const Index n = y_dn.cols();

        ColVec Y_n = y_dn.colwise().sum().transpose(); // n x 1
        ColVec Y_n1 = Y_n.unaryExpr(at_least_one);     // n x 1

        ///////////////////////////
        // parameter of interest //
        ///////////////////////////

        Mat R_nk =
            (y_dn.transpose() * logX_dk).array().colwise() / Y_n1.array();

#pragma omp critical
        {
            for (Index i = 0; i < (ub - lb); ++i) {
                const Index j = i + lb;
                Rtot_nk.row(j) = R_nk.row(i);
                Ytot_n(j, 0) = Y_n(i);
            }

            Nprocessed += n;
            if (verbose) {
                Rcpp::Rcerr << "\rProcessed: " << Nprocessed << std::flush;
            } else {
                Rcpp::Rcerr << "+ " << std::flush;
                if (Nprocessed % 1000 == 0)
                    Rcpp::Rcerr << "\r" << std::flush;
            }
        } // end of omp critical
    }

    if (!verbose)
        Rcpp::Rcerr << std::endl;
    TLOG("Done");

    using namespace rcpp::util;
    using namespace Rcpp;

    std::vector<std::string> d_ = pos2row;
    std::vector<std::string> k_;
    for (std::size_t k = 1; k <= K; ++k)
        k_.push_back(std::to_string(k));
    std::vector<std::string> file_ { mtx_file };

    return List::create(_["beta"] = named(logX_dk.unaryExpr(exp), d_, k_),
                        _["corr"] = named(Rtot_nk, coln, k_),
                        _["colsum"] = named(Ytot_n, coln, file_));
}
