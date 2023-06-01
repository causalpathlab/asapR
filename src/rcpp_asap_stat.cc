#include "rcpp_asap_stat.hh"

std::tuple<Vec, Vec>
compute_row_stat(const std::string mtx_file,
                 const std::vector<Index> &mtx_idx,
                 const Index block_size,
                 const bool do_log1p = false,
                 const std::size_t NUM_THREADS = 1,
                 const bool verbose = true)
{
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    log1p_op<Mat> log1p;
    safe_sqrt_op<Mat> safe_sqrt;

    mm_info_reader_t info;
    CHK_ERR_EXIT(peek_bgzf_header(mtx_file, info),
                 "Failed to read the size: " << mtx_file);

    const Scalar eps = 1e-8;
    const Index D = info.max_row; // dimensionality
    const Index N = info.max_col; // number of cells

    ColVec mu(D), sig(D);

    mu.setZero();
    sig.setOnes();

    ColVec s1(D), s2(D);
    s1.setZero();
    s2.setZero();

    if (verbose) {
        TLOG("Collecting row-wise statistics...");
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);

        ///////////////////////////////////////
        // memory location = 0 means the end //
        ///////////////////////////////////////

        const Index lb_mem = lb < N ? mtx_idx.at(lb) : 0;
        const Index ub_mem = ub < N ? mtx_idx.at(ub) : 0;

        mmutil::stat::row_collector_t collector(do_log1p);
        collector.set_size(info.max_row, info.max_col, info.max_elem);

        CHECK(visit_bgzf_block(mtx_file, lb_mem, ub_mem, collector));

#pragma omp critical
        {
            s1 += collector.Row_S1;
            s2 += collector.Row_S2;
        }
    }

    const Scalar nn = static_cast<Scalar>(N);

    mu = s1 / nn;
    sig = (s2 / nn - mu.cwiseProduct(mu)).unaryExpr(safe_sqrt);
    sig.array() += eps;

    return std::make_tuple(mu, sig);
}
