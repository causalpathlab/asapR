#include "mmutil.hh"
#include "rcpp_util.hh"

#ifndef BTREE_HH_
#define BTREE_HH_

namespace rcpp { namespace btree {

template <typename T>
struct data_traits {
    using Distrib = typename T::Distrib; // distribution tag
    using Unit = typename T::Unit;       // unit data type
};

template <typename T>
using distrib_tag = typename data_traits<T>::Distrib;

struct tag_poisson { };




}} // end of rcpp::btree

#endif
