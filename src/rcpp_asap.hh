#ifndef _RCPP_ASAP_HH
#define _RCPP_ASAP_HH

// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include "mmutil.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "mmutil_io.hh"
#include "mmutil_stat.hh"
#include "mmutil_match.hh"

using namespace mmutil::io;
using namespace mmutil::bgzf;

#include "tuple_util.hh"
#include "svd.hh"

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>

// [[Rcpp::depends(dqrng, sitmo, BH)]]
#include <dqrng.h>
#include <dqrng_distribution.h>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <xoshiro.h>

#include "rcpp_util.hh"

#endif
