// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// [[Rcpp::plugins(openmp)]]
#include <omp.h>

// [[Rcpp::depends(dqrng, sitmo, BH)]]
#include <dqrng.h>
#include <dqrng_distribution.h>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <xoshiro.h>

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include <unordered_map>

#ifndef MMUTIL_HH_
#define MMUTIL_HH_

using Scalar = float;
using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
using Index = SpMat::Index;

using Mat = typename Eigen::
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using IntMat = typename Eigen::
    Matrix<std::ptrdiff_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using IntVec = typename Eigen::Matrix<std::ptrdiff_t, Eigen::Dynamic, 1>;

#ifdef __cplusplus
extern "C" {
#endif

#include "bgzf.h"
#include "kstring.h"

#ifdef __cplusplus
}
#endif

#include "util.hh"
#include "eigen_util.hh"
#include "std_util.hh"
#include "math.hh"
#include "util.hh"
#include "check.hh"
#include "io.hh"

#endif
