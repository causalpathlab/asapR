#ifndef RCPPEIG_UTIL_HH_
#define RCPPEIG_UTIL_HH_

namespace rcpp { namespace util {

std::tuple<Index, Index, Scalar>
parse_triplet(const std::tuple<Index, Index, Scalar> &tt);

std::tuple<Index, Index, Scalar>
parse_triplet(const Eigen::Triplet<Scalar> &tt);

std::vector<std::string> copy(const Rcpp::StringVector &r_vec);

void copy(const Rcpp::StringVector &r_vec, std::vector<std::string> &vec);

}} // namespace rcpp::util

#endif
