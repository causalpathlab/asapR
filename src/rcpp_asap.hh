#ifndef _RCPP_ASAP_HH
#define _RCPP_ASAP_HH

#include "mmutil.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "mmutil_index.hh"
#include "mmutil_io.hh"
#include "mmutil_stat.hh"

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

#include "dirichlet_prior.hh"
#include "poisson_nmf_model.hh"
#include "poisson_cluster_model.hh"
#include "gamma_parameter.hh"
#include "latent_matrix.hh"

#include "clustering.hh"
#include "dirichlet_prior.hh"
#include "poisson_cluster_model.hh"

#endif
