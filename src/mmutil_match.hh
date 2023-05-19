#ifndef MMUTIL_MATCH_HH_
#define MMUTIL_MATCH_HH_

#include "math.hh"
#include "mmutil.hh"

namespace mmutil { namespace match {

struct KNN {
    explicit KNN(const std::size_t _val)
        : val(_val)
    {
    }
    const std::size_t val;
};

/**
 * @param deg_i number of elements
 * @param dist deg_i-vector for distance
 * @param weights deg_i-vector for weights

 Since the inner-product distance is d(x,y) = (1 - x'y),
 d = 0.5 * (x - y)'(x - y) = 0.5 * (x'x + y'y) - x'y,
 we have Gaussian weight w(x,y) = exp(-lambda * d(x,y))

 */
void normalize_weights(const Index deg_i,
                       std::vector<Scalar> &dist,
                       std::vector<Scalar> &weights);

}} // namespace mmutil::match

#endif
