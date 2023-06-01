#include "rcpp_asap.hh"

#ifndef RCPP_ASAP_STAT_HH_
#define RCPP_ASAP_STAT_HH_

std::tuple<Vec, Vec> compute_row_stat(const std::string mtx_file,
                                      const std::vector<Index> &mtx_idx,
                                      const Index block_size,
                                      const bool do_log1p,
                                      const std::size_t NUM_THREADS,
                                      const bool verbose);

#endif
