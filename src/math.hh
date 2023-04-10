
#ifdef __cplusplus
extern "C" {
#endif

#include "fastexp.h"
#include "fastlog.h"
#include "fastgamma.h"

#ifdef __cplusplus
}
#endif

#include <cmath>

#ifndef _UTIL_MATH_HH_
#define _UTIL_MATH_HH_

/////////////////////
// log(1 + exp(x)) //
/////////////////////

template <typename T>
inline T
_softplus(const T x)
{
    const T cutoff = static_cast<T>(10.);
    const T one = static_cast<T>(1.0);
    if (x > cutoff) {
        return x + std::log1p(std::exp(-x));
    }
    return std::log(one + std::exp(x));
}

/////////////////////
// 1/(1 + exp(-x)) //
/////////////////////

template <typename T>
inline T
_sigmoid(const T x, const T pmin = 0.0, const T pmax = 1.0)
{
    const T cutoff = static_cast<T>(10.);
    const T one = static_cast<T>(1.0);

    if (x < cutoff) {
        return pmax * std::exp(x) / (one + std::exp(x));
    }

    return pmax / (one + std::exp(-x)) + pmin;
}

//////////////////////////
// log(exp(a) + exp(b)) //
//////////////////////////

template <typename T>
inline T
_log_sum_exp(const T log_a, const T log_b)
{
    if (log_a > log_b) {
        return log_a + _softplus(log_b - log_a);
    }
    return log_b + _softplus(log_a - log_b);
}

#endif
