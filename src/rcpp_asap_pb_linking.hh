#include "rcpp_asap.hh"
#include "rcpp_asap_stat.hh"
#include "rcpp_mtx_data.hh"
#include "rcpp_asap_pb.hh"

#ifndef RCPP_ASAP_PB_LINKING_HH_
#define RCPP_ASAP_PB_LINKING_HH_

template <typename T1, typename T2>
Rcpp::List
run_asap_pb_linking(T1 &data_src,
                    T2 &data_tgt,
                    const asap::pb::options_t &options)
{

    using namespace asap::pb;

    const Index K = options.K;

    // const bool save_aux_data = options.save_aux_data;
    // const bool do_batch_adj = options.do_batch_adj;
    // const std::size_t BATCH_ADJ_ITER = options.BATCH_ADJ_ITER;
    // const std::size_t KNN_CELL = options.KNN_CELL;

    const Scalar cell_norm = options.CELL_NORM;
    const bool do_log1p = options.do_log1p;
    const bool do_down_sample = options.do_down_sample;
    const std::size_t CELL_PER_SAMPLE = options.CELL_PER_SAMPLE;

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

    // data_src.row_names()
    // data_src.col_names()


    // Step 1. figure out source -> target how many sources?


    //////////////////////////////////////////////////////
    // First figure out the intersection of the columns //
    //////////////////////////////////////////////////////

    // std::vector<std::string> columns;
    // std::unordered_map<std::string, Index> col2pos;

    // rcpp::util::take_common_names(col_files,
    //                               columns,
    //                               col2pos,
    //                               false,
    //                               MAX_COL_WORD,
    //                               COL_WORD_SEP);

    // TLOG_(verbose, "Found " << col2pos.size() << " column names");

    // ASSERT_RETL(columns.size() > 0, "Empty column names!");


    // R * (E * G)


    return Rcpp::List::create();
}

#endif
