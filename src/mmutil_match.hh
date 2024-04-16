#ifndef MMUTIL_MATCH_HH_
#define MMUTIL_MATCH_HH_

#include "math.hh"
#include "mmutil.hh"
#include "tuple_util.hh"

#include "RcppAnnoy.h"
#define ANNOYLIB_MULTITHREADED_BUILD 1

namespace mmutil { namespace match {

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

struct KNN {
    explicit KNN(const std::size_t _val)
        : val(_val)
    {
    }
    const std::size_t val;
};

/////////////////////////////////////////////////////////////
// @param deg_i number of elements			   //
// @param dist deg_i-vector for distance		   //
// @param weights deg_i-vector for weights		   //
// 							   //
// Since the inner-product distance is d(x,y) = (1 - x'y), //
// d = 0.5 * (x - y)'(x - y) = 0.5 * (x'x + y'y) - x'y,    //
// we have Gaussian weight w(x,y) = exp(-lambda * d(x,y))  //
/////////////////////////////////////////////////////////////
void normalize_weights(const Index deg_i,
                       std::vector<Scalar> &dist,
                       std::vector<Scalar> &weights);

template <typename D>
using ANNOY_INDEX = Annoy::
    AnnoyIndex<Index, Scalar, D, Kiss64Random, RcppAnnoyIndexThreadPolicy>;

template <typename ANND, typename Derived>
int _build_annoy_index(const Eigen::MatrixBase<Derived> &data_kn,
                       std::shared_ptr<ANNOY_INDEX<ANND>> &ret,
                       const std::size_t num_threads);

template <typename Derived>
std::shared_ptr<ANNOY_INDEX<Annoy::Euclidean>>
build_euclidean_annoy(const Eigen::MatrixBase<Derived> &data_kn,
                      const std::size_t num_threads)
{
    std::shared_ptr<ANNOY_INDEX<Annoy::Euclidean>> ret;
    _build_annoy_index(data_kn, ret, num_threads);
    return ret;
}

template <typename Derived>
std::shared_ptr<ANNOY_INDEX<Annoy::Angular>>
build_angular_annoy(const Eigen::MatrixBase<Derived> &_data_kn,
                    const std::size_t num_threads)
{
    std::shared_ptr<ANNOY_INDEX<Annoy::Angular>> ret;
    Mat data_kn = _data_kn;
    normalize_columns_inplace(data_kn);
    _build_annoy_index(_data_kn, ret, num_threads);
    return ret;
}

template <typename Dist, typename Derived>
int _match_annoy(const Eigen::MatrixBase<Derived> &_source_kn,
                 const Eigen::MatrixBase<Derived> &_target_km,
                 const options_t &options,
                 std::vector<std::tuple<Index, Index, Scalar>> &ret);

template <typename Derived>
int
match_euclidean_annoy(const Eigen::MatrixBase<Derived> &_source_kn,
                      const Eigen::MatrixBase<Derived> &_target_km,
                      const options_t &options,
                      std::vector<std::tuple<Index, Index, Scalar>> &ret)
{
    return _match_annoy<ANNOY_INDEX<Annoy::Euclidean>, Derived>(_source_kn,
                                                                _target_km,
                                                                options,
                                                                ret);
}

template <typename Derived>
int
match_angular_annoy(const Eigen::MatrixBase<Derived> &_source_kn,
                    const Eigen::MatrixBase<Derived> &_target_km,
                    const options_t &options,
                    std::vector<std::tuple<Index, Index, Scalar>> &ret)
{
    Mat source_kn = _source_kn;
    Mat target_km = _target_km;

    normalize_columns_inplace(source_kn);
    normalize_columns_inplace(target_km);

    return _match_annoy<ANNOY_INDEX<Annoy::Angular>, Derived>(source_kn,
                                                              target_km,
                                                              options,
                                                              ret);
}

template <typename Dist, typename Derived>
int
_match_annoy(const Eigen::MatrixBase<Derived> &source_kn,
             const Eigen::MatrixBase<Derived> &target_km,
             const options_t &options,
             std::vector<std::tuple<Index, Index, Scalar>> &ret)
{

    const std::size_t num_neighbours = options.knn_per_batch;
    const std::size_t num_threads = options.num_threads;
    const std::size_t block_size = options.block_size;

    std::shared_ptr<Dist> idx_source, idx_target;

    ASSERT_RET(source_kn.rows() == target_km.rows(),
               "should have the same number of rows");

    _build_annoy_index(source_kn, idx_source, num_threads);
    _build_annoy_index(target_km, idx_target, num_threads);

    const std::size_t Ntot = source_kn.cols(), Mtot = target_km.cols();
    const Index rank = source_kn.rows();

    // TLOG("Built two ANNOY indexes");

    auto distribute_jobs = [&block_size](auto N) {
        std::vector<Index> _jobs(N);
        std::iota(_jobs.begin(), _jobs.end(), 0);
        std::transform(_jobs.begin(),
                       _jobs.end(),
                       _jobs.begin(),
                       [&block_size](auto &x) { return x / block_size; });
        return make_index_vec_vec(_jobs);
    };

    //////////////////////
    // source -> target //
    //////////////////////

    auto st_jobs = distribute_jobs(Ntot);

    // TLOG("S -> T: " << st_jobs.size());

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
    for (Index job = 0; job < st_jobs.size(); ++job) {
        const std::vector<Index> &_cells = st_jobs.at(job);
        std::vector<Scalar> query(rank);

        const auto &index = *idx_target.get();

        const std::size_t nsearch = std::min(num_neighbours, Mtot);

        std::vector<Scalar> weight(nsearch);

        for (Index ii : _cells) {
            Eigen::Map<Mat>(query.data(), rank, 1) = source_kn.col(ii);

            std::vector<Scalar> dist;
            std::vector<Index> neigh;

            index.get_nns_by_vector(query.data(), nsearch, -1, &neigh, &dist);

            normalize_weights(nsearch, dist, weight);

#pragma omp critical
            {
                for (std::size_t k = 0; k < nsearch; ++k) {
                    ret.emplace_back(ii, neigh.at(k), weight.at(k));
                }
            }
        }
    }
    st_jobs.clear();

    //////////////////////
    // target -> source //
    //////////////////////

    auto ts_jobs = distribute_jobs(Mtot);

    // TLOG("T -> S: " << ts_jobs.size());

#if defined(_OPENMP)
#pragma omp parallel num_threads(num_threads)
#pragma omp for
#endif
    for (Index job = 0; job < ts_jobs.size(); ++job) {
        const std::vector<Index> &_cells = ts_jobs.at(job);
        std::vector<Scalar> query(rank);

        const auto &index = *idx_source.get();

        const std::size_t nsearch = std::min(num_neighbours, Ntot);

        std::vector<Scalar> weight(nsearch);

        for (Index ii : _cells) {
            Eigen::Map<Mat>(query.data(), rank, 1) = target_km.col(ii);

            std::vector<Scalar> dist;
            std::vector<Index> neigh;

            index.get_nns_by_vector(query.data(), nsearch, -1, &neigh, &dist);

            normalize_weights(nsearch, dist, weight);

#pragma omp critical
            {
                for (std::size_t k = 0; k < nsearch; ++k) {
                    ret.emplace_back(neigh.at(k), ii, weight.at(k));
                }
            }
        }
    }
    ts_jobs.clear();

    return EXIT_SUCCESS;
}

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

//////////////////////
// reciprocal match //
//////////////////////

std::tuple<Index, Index, Scalar>
parse_triplet(const std::tuple<Index, Index, Scalar> &tt);

std::tuple<Index, Index, Scalar>
parse_triplet(const Eigen::Triplet<Scalar> &tt);

template <typename T>
inline std::vector<T>
keep_reciprocal_knn(const std::vector<T> &knn_index, bool undirected = false)
{
    // Make sure that we could only consider reciprocal kNN pairs
    std::unordered_map<std::tuple<Index, Index>,
                       short,
                       hash_tuple::hash<std::tuple<Index, Index>>>
        edge_count;

    auto _count = [&edge_count](const auto &tt) {
        Index i, j, temp;
        std::tie(i, j, std::ignore) = parse_triplet(tt);
        if (i == j)
            return;

        if (i > j) {
            temp = i;
            i = j;
            j = temp;
        }

        if (edge_count.count({ i, j }) < 1) {
            edge_count[{ i, j }] = 1;
        } else {
            edge_count[{ i, j }] += 1;
        }
    };

    std::for_each(knn_index.begin(), knn_index.end(), _count);

    auto is_mutual = [&edge_count, &undirected](const auto &tt) {
        Index i, j, temp;
        std::tie(i, j, std::ignore) = parse_triplet(tt);
        if (i == j)
            return false;
        if (i > j) {
            temp = i;
            i = j;
            j = temp;
        }
        if (undirected)
            return (edge_count[{ i, j }] > 1) && (i <= j);
        return (edge_count[{ i, j }] > 1);
    };

    std::vector<T> reciprocal_knn_index;
    reciprocal_knn_index.reserve(knn_index.size());
    std::copy_if(knn_index.begin(),
                 knn_index.end(),
                 std::back_inserter(reciprocal_knn_index),
                 is_mutual);

    return reciprocal_knn_index;
}

}} // namespace mmutil::match

#endif
