#include "mmutil.hh"

#ifndef RCPP_UTIL_STRUCT_HH_
#define RCPP_UTIL_STRUCT_HH_

namespace rcpp { namespace util {

std::size_t pbt_num_leaves_to_nodes(const std::size_t num_leaves);

std::size_t pbt_num_depth_to_nodes(const std::size_t depth);

std::size_t pbt_num_depth_to_leaves(const std::size_t depth);

}} // end of namespace

#endif
