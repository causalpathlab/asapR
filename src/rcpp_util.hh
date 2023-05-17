// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>

#include <vector>

#ifndef RCPP_UTIL_HH_
#define RCPP_UTIL_HH_

namespace rcpp { namespace util {

std::vector<std::string> copy(const Rcpp::StringVector &r_vec);

void copy(const Rcpp::StringVector &r_vec, std::vector<std::string> &vec);

}} // namespace rcpp::util

#endif
