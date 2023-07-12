#include "mmutil_match.hh"

namespace mmutil { namespace match {

void
normalize_weights(const Index deg_i,
                  std::vector<Scalar> &dist,
                  std::vector<Scalar> &weights)
{
    if (deg_i < 2) {
        weights[0] = 1.;
        return;
    }

    const float _log2 = fasterlog(2.);
    const float _di = static_cast<float>(deg_i);
    const float log2K = fasterlog(_di) / _log2;

    float lambda = 10.0;

    const float dmin = *std::min_element(dist.begin(), dist.begin() + deg_i);

    // Find lambda values by a simple line-search
    auto f = [&](const float lam) -> float {
        float rhs = 0.;
        for (Index j = 0; j < deg_i; ++j) {
            float w = fasterexp(-(dist[j] - dmin) * lam);
            rhs += w;
        }
        float lhs = log2K;
        return (lhs - rhs);
    };

    float fval = f(lambda);

    const Index max_iter = 100;

    for (Index iter = 0; iter < max_iter; ++iter) {
        float _lam = lambda;
        if (fval < 0.) {
            _lam = lambda * 1.1;
        } else {
            _lam = lambda * 0.9;
        }
        float _fval = f(_lam);
        if (std::abs(_fval) > std::abs(fval)) {
            break;
        }
        lambda = _lam;
        fval = _fval;
    }

    for (Index j = 0; j < deg_i; ++j) {
        weights[j] = fasterexp(-(dist[j] - dmin) * lambda);
    }
}

std::tuple<Index, Index, Scalar>
parse_triplet(const std::tuple<Index, Index, Scalar> &tt)
{
    return tt;
}

std::tuple<Index, Index, Scalar>
parse_triplet(const Eigen::Triplet<Scalar> &tt)
{
    return std::make_tuple(tt.row(), tt.col(), tt.value());
}

}} // namespace mmutil::match
