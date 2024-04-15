#include "rcpp_asap.hh"
#include "rcpp_asap_stat.hh"
#include "rcpp_mtx_data.hh"

#ifndef RCPP_ASAP_PB_HH_
#define RCPP_ASAP_PB_HH_

namespace asap { namespace pb {

struct options_t {

    Index K;
    std::size_t rseed;
    bool verbose;
    std::size_t NUM_THREADS;
    std::size_t BLOCK_SIZE;
    bool do_batch_adj;
    bool do_log1p;
    bool do_down_sample;
    bool save_rand_proj;
    std::size_t KNN_CELL;
    std::size_t CELL_PER_SAMPLE;
    std::size_t BATCH_ADJ_ITER;
    double a0;
    double b0;
};

template <typename Derived>
void
sample_random_projection(const Index D,
                         const Index K,
                         const std::size_t rseed,
                         Eigen::MatrixBase<Derived> &_r_kd)
{

    Derived &R_kd = _r_kd.derived();
    R_kd.resize(K, D);

    using norm_dist_t = boost::random::normal_distribution<Scalar>;
    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    norm_dist_t norm_dist(0., 1.);

    auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };
    R_kd = Mat::NullaryExpr(K, D, rnorm);
}

template <typename Derived>
void
binary_shift_membership(const Eigen::MatrixBase<Derived> &RD,
                        std::vector<typename Derived::Index> &membership)
{

    const std::size_t N = RD.rows();
    const std::size_t K = RD.cols();
    IntVec bb(N);
    bb.setZero();

    for (Index k = 0; k < K; ++k) {
        auto binary_shift = [&k](const Scalar &x) -> Index {
            return x > 0. ? (1 << k) : 0;
        };
        bb += RD.col(k).unaryExpr(binary_shift);
    }

    membership.resize(N);
    for (Index j = 0; j < N; ++j) {
        membership[j] = bb(j);
    }
}

template <typename Derived, typename RNG>
void
randomly_assign_rows_to_samples(const Eigen::MatrixBase<Derived> &RD,
                                std::vector<typename Derived::Index> &positions,
                                RNG &rng,
                                const bool verbose = false,
                                const bool do_down_sample = false,
                                const std::size_t CELL_PER_SAMPLE = 0)
{

    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;

    // a. Sort them out into an implicit binary tree

    std::vector<Index> membership;
    binary_shift_membership(RD, membership);

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

    // b. actually assign cells into pseudobulk samples
    positions.clear();
    positions.resize(membership.size());

    auto _pos_op = [&pb_position](const std::size_t x) {
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
}

template <typename Derived>
int
apply_mtx_row_sd(const std::string mtx_file,
                 const std::string idx_file,
                 Eigen::MatrixBase<Derived> &_r_kd,
                 const bool verbose = false,
                 const std::size_t NUM_THREADS = 1,
                 const std::size_t BLOCK_SIZE = 100,
                 const bool do_log1p = false)
{

    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;
    using Index = typename Derived::Index;
    using Scalar = typename Derived::Scalar;
    Derived &R_kd = _r_kd.derived();

    const std::size_t D = R_kd.cols();
    ColVec row_mu(D), row_sig(D);

    std::vector<Index> mtx_idx;
    CHK_RET_(read_mmutil_index(idx_file, mtx_idx),
             "Failed to read:" << std::endl
                               << idx_file << std::endl
                               << "Consider rebuilding it." << std::endl);

    std::tie(row_mu, row_sig) = compute_row_stat(mtx_file,
                                                 mtx_idx,
                                                 BLOCK_SIZE,
                                                 do_log1p,
                                                 NUM_THREADS,
                                                 verbose);
    R_kd.array().colwise() *= row_sig.array();

    return EXIT_SUCCESS;
}

}} // asap::pb
#endif
