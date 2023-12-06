#include "rcpp_asap.hh"
#include "rcpp_asap_pb.hh"
#include "rcpp_asap_batch.hh"
#include "rcpp_asap_mtx_data.hh"
#include "RcppAnnoy.h"
#define ANNOYLIB_MULTITHREADED_BUILD 1

#ifndef RCPP_ASAP_BBKNN_HH_
#define RCPP_ASAP_BBKNN_HH_

namespace asap { namespace bbknn {

template <typename D>
using ANNOY_INDEX = Annoy::
    AnnoyIndex<Index, Scalar, D, Kiss64Random, RcppAnnoyIndexThreadPolicy>;

template <typename Derived>
std::shared_ptr<ANNOY_INDEX<Annoy::Euclidean>>
build_euclidean_annoy(const Eigen::MatrixBase<Derived> &data_nk,
                      const std::size_t num_threads = 1);

template <typename Derived>
std::shared_ptr<ANNOY_INDEX<Annoy::Angular>>
build_angular_annoy(const Eigen::MatrixBase<Derived> &data_nk,
                    const std::size_t num_threads = 1);

struct options_t {
    options_t()
        : block_size(100)
        , num_threads(1)
        , knn_per_batch(3)
        , verbose(true)
    {
    }

    Index block_size;
    std::size_t num_threads;
    std::size_t knn_per_batch;
    bool verbose;
};

template <typename Derived, typename AP>
int
build_knn(const Eigen::MatrixBase<Derived> &query_kn,
          AP &ann_ptr,
          const options_t &options,
          std::vector<std::tuple<Index, Index, Scalar>> &ret)
{

    const Index block_size = options.block_size;
    const std::size_t num_threads = options.num_threads;
    const Index knn_per_batch = options.knn_per_batch;
    const bool verbose = options.verbose;

    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;
    const Index Ntot = query_kn.cols();
    const Index rank = query_kn.rows();

    std::vector<Index> _jobs(Ntot);
    std::iota(_jobs.begin(), _jobs.end(), 0);
    std::for_each(_jobs.begin(), _jobs.end(), [&block_size](auto &x) {
        return x / block_size;
    });

    std::vector<std::vector<Index>> job_vec = make_index_vec_vec(_jobs);

    Index Nprocessed = 0;

    const auto &index = *ann_ptr.get();

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
    for (Index job = 0; job < job_vec.size(); ++job) {
        const std::vector<Index> &_cells = job_vec.at(job);
        std::vector<Scalar> query(rank);

        for (Index glob_i : _cells) {
            Eigen::Map<Mat>(query.data(), rank, 1) = query_kn.col(glob_i);

            std::vector<Scalar> dist_ij;
            std::vector<Scalar> weight_ij;
            std::vector<Index> pair_ij;

            const std::size_t nsearch =
                std::min(knn_per_batch, query_kn.cols());
            std::vector<Index> neigh;
            std::vector<Scalar> dist;

            index.get_nns_by_vector(query.data(), nsearch, -1, &neigh, &dist);

            for (std::size_t k = 0; k < nsearch; ++k) {
                const Scalar d = dist.at(k);
                const Index glob_j = neigh.at(k);

                if (glob_i != glob_j) {
                    pair_ij.emplace_back(glob_j);
                    dist_ij.emplace_back(d);
                    weight_ij.emplace_back(0);
                }
            }

            const std::size_t deg_ij = pair_ij.size();
            mmutil::match::normalize_weights(deg_ij, dist_ij, weight_ij);

#pragma omp critical
            {
                for (std::size_t k = 0; k < deg_ij; ++k) {
                    ret.emplace_back(glob_i, pair_ij.at(k), weight_ij.at(k));
                }
                Nprocessed += 1;
                if (verbose) {
                    Rcpp::Rcerr << "\rProcessed: " << Nprocessed << std::flush;
                } else {
                    Rcpp::Rcerr << "+ " << std::flush;
                    if (Nprocessed % 100 == 0)
                        Rcpp::Rcerr << "\r" << std::flush;
                }
            }
        }
    }

    if (verbose) {
        Rcpp::Rcerr << "Processed " << Nprocessed << " pairs" << std::endl;
    } else {
        Rcpp::Rcerr << std::endl;
    }
    return EXIT_SUCCESS;
}

template <typename Derived, typename APV>
int
build_knn(const Eigen::MatrixBase<Derived> &query_kn,
          const std::vector<std::vector<Index>> &batch_set,
          const APV &ann_ptr_vec,
          const options_t &options,
          std::vector<std::tuple<Index, Index, Scalar>> &ret)
{

    const Index block_size = options.block_size;
    const std::size_t num_threads = options.num_threads;
    const std::size_t knn_per_batch = options.knn_per_batch;
    const bool verbose = options.verbose;

    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;
    const Index Ntot = query_kn.cols();
    const Index rank = query_kn.rows();
    for (auto batch : batch_set) {
        for (auto j : batch) {
            ASSERT_RET(j < Ntot,
                       "This sample index [" << j << "] exceeds " << Ntot);
        }
    }

    ASSERT_RET(batch_set.size() == ann_ptr_vec.size(),
               "batch set and annoy index should be matched");

    std::vector<Index> _jobs(Ntot);
    std::iota(_jobs.begin(), _jobs.end(), 0);
    std::for_each(_jobs.begin(), _jobs.end(), [&block_size](auto &x) {
        return x / block_size;
    });

    std::vector<std::vector<Index>> job_vec = make_index_vec_vec(_jobs);

    Index Nprocessed = 0;
    const Index B = ann_ptr_vec.size();

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
    for (Index job = 0; job < job_vec.size(); ++job) {
        const std::vector<Index> &_cells = job_vec.at(job);
        std::vector<Scalar> query(rank);

        for (Index glob_i : _cells) {
            Eigen::Map<Mat>(query.data(), rank, 1) = query_kn.col(glob_i);

            std::vector<Scalar> dist_ij;
            std::vector<Scalar> weight_ij;
            std::vector<Index> pair_ij;

            // for each batch
            for (Index b = 0; b < B; ++b) {
                const auto &index = *ann_ptr_vec.at(b).get();
                const auto &globs = batch_set.at(b);

                const std::size_t nsearch =
                    std::min(knn_per_batch, globs.size());
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
            } // for each batch

            const std::size_t deg_ij = pair_ij.size();
            mmutil::match::normalize_weights(deg_ij, dist_ij, weight_ij);

#pragma omp critical
            {
                for (std::size_t k = 0; k < deg_ij; ++k) {
                    ret.emplace_back(glob_i, pair_ij.at(k), weight_ij.at(k));
                }
                Nprocessed += 1;
                if (verbose) {
                    Rcpp::Rcerr << "\rProcessed: " << Nprocessed << std::flush;
                } else {
                    Rcpp::Rcerr << "+ " << std::flush;
                    if (Nprocessed % 100 == 0)
                        Rcpp::Rcerr << "\r" << std::flush;
                }
            }

        } // for each i
    }     // for each job

    if (verbose) {
        Rcpp::Rcerr << "\nProcessed " << Nprocessed << " pairs" << std::endl;
    } else {
        Rcpp::Rcerr << std::endl;
    }
    return EXIT_SUCCESS;
}

/////////////
// details //
/////////////

template <typename ANND, typename Derived>
int
_build_annoy_index(const Eigen::MatrixBase<Derived> &data_kn,
                   std::shared_ptr<ANNOY_INDEX<ANND>> &ret,
                   const std::size_t num_threads)
{
    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;

    const std::size_t rank = data_kn.rows();

    std::vector<Scalar> vec(rank);

    using index_t = ANNOY_INDEX<ANND>;

    ret = std::make_shared<index_t>(rank);
    index_t &index = *ret.get();

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
    for (Index j = 0; j < data_kn.cols(); ++j) {
        Eigen::Map<Mat>(vec.data(), rank, 1) = data_kn.col(j);
#pragma omp critical
        {
            index.add_item(j, vec.data()); // allocate
        }
    }

    index.build(50);
    return EXIT_SUCCESS;
}

template <typename Derived>
std::shared_ptr<ANNOY_INDEX<Annoy::Euclidean>>
build_euclidean_annoy(const Eigen::MatrixBase<Derived> &data_nk,
                      const std::size_t num_threads)
{
    std::shared_ptr<ANNOY_INDEX<Annoy::Euclidean>> ret;
    Mat _data_kn = data_nk.transpose();
    _build_annoy_index(_data_kn, ret, num_threads);
    return ret;
}

template <typename Derived>
std::shared_ptr<ANNOY_INDEX<Annoy::Angular>>
build_angular_annoy(const Eigen::MatrixBase<Derived> &data_nk,
                    const std::size_t num_threads)
{
    std::shared_ptr<ANNOY_INDEX<Annoy::Angular>> ret;
    Mat _data_kn = data_nk.transpose();
    normalize_columns_inplace(_data_kn);
    _build_annoy_index(_data_kn, ret, num_threads);
    return ret;
}

}} // namespace

#endif
