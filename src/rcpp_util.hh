// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <vector>
#include <string>
#include "util.hh"

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

template <typename T>
void
convert_r_index(const std::vector<T> &cvec, std::vector<T> &rvec)
{
    rvec.resize(cvec.size());
    auto r_index = [](const T x) -> T { return x + 1; };
    std::transform(cvec.begin(), cvec.end(), rvec.begin(), r_index);
}

template <typename Derived>
void
build_sparse_mat(const Rcpp::List &in_list,
                 const std::size_t nrow,
                 const std::size_t ncol,
                 Eigen::SparseMatrixBase<Derived> &ret_)
{
    using Scalar = typename Derived::Scalar;

    Derived &ret = ret_.derived();
    ret.resize(nrow, ncol);
    ret.setZero();
    std::vector<Eigen::Triplet<Scalar>> triples;

    if (in_list.size() == 3) {
        const std::vector<std::size_t> &ii = in_list[0];
        const std::vector<std::size_t> &jj = in_list[1];
        const std::vector<Scalar> &kk = in_list[2];
        const std::size_t m = ii.size();

        if (jj.size() == m && kk.size() == m) {
            triples.reserve(m);
            for (std::size_t e = 0; e < m; ++e) {
                const std::size_t i = ii.at(e), j = jj.at(e);
                if (i < 1 || j < 1) {
                    ELOG("Assume R's 1-based indexing");
                    continue;
                }
                if (i <= nrow && j <= ncol) {
                    // 1-based -> 0-based
                    triples.emplace_back(
                        Eigen::Triplet<Scalar>(i - 1, j - 1, kk.at(e)));
                }
            }
        } else {
            WLOG("input list sizes don't match");
        }
    } else if (in_list.size() == 2) {

        const std::vector<std::size_t> &ii = in_list[0];
        const std::vector<std::size_t> &jj = in_list[1];
        const std::size_t m = ii.size();

        if (jj.size() == m) {
            triples.reserve(m);
            for (std::size_t e = 0; e < m; ++e) {
                const std::size_t i = ii.at(e), j = jj.at(e);
                if (i < 1 || j < 1) {
                    ELOG("Assume R's 1-based indexing");
                    continue;
                }
                if (i <= nrow && j <= ncol) {
                    // 1-based -> 0-based
                    triples.emplace_back(
                        Eigen::Triplet<Scalar>(i - 1, j - 1, 1.));
                }
            }
        } else {
            WLOG("input list sizes don't match");
        }
    } else {
        WLOG("Need two or three vectors in the list");
    }

    ret.reserve(triples.size());
    ret.setFromTriplets(triples.begin(), triples.end());
}

}} // namespace rcpp::util

#endif
