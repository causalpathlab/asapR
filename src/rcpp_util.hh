// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <vector>
#include <string>

#ifndef RCPP_UTIL_HH_
#define RCPP_UTIL_HH_

namespace rcpp { namespace util {

std::vector<std::string> copy(const Rcpp::StringVector &r_vec);

void copy(const Rcpp::StringVector &r_vec, std::vector<std::string> &vec);

template <typename Derived>
Rcpp::NumericMatrix
named(const Eigen::MatrixBase<Derived> &xx,
      const std::vector<std::string> &out_row_names,
      const std::vector<std::string> &out_col_names)
{
    Rcpp::NumericMatrix x = Rcpp::wrap(xx);
    if (xx.rows() == out_row_names.size()) {
        Rcpp::rownames(x) = Rcpp::wrap(out_row_names);
    }
    if (xx.cols() == out_col_names.size()) {
        Rcpp::colnames(x) = Rcpp::wrap(out_col_names);
    }
    return x;
}

}} // namespace rcpp::util

#endif
