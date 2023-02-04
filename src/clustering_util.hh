#include "mmutil.hh"

#ifndef CLUSTERING_UTIL_HH_
#define CLUSTERING_UTIL_HH_

template <typename T>
inline T
sort_cluster_index(std::vector<T> &_membership, const T cutoff = 0)
{
    const auto N = _membership.size();
    std::vector<T> _sz = count_frequency(_membership, cutoff);
    const T kk = _sz.size();
    auto _order = std_argsort(_sz);

    std::vector<T> rename(kk, -1);
    T k_new = 0;
    for (T k : _order) {
        if (_sz.at(k) >= cutoff)
            rename[k] = k_new++;
    }

    for (std::size_t j = 0; j < N; ++j) {
        const T k_old = _membership.at(j);
        const T k_new = rename.at(k_old);
        _membership[j] = k_new;
    }

    return k_new;
}

template <typename T, typename OFS>
void
print_histogram(const std::vector<T> &nn, //
                OFS &ofs,                 //
                const T height = 50.0,    //
                const T cutoff = .01,     //
                const int ntop = 10)
{
    using std::accumulate;
    using std::ceil;
    using std::floor;
    using std::round;
    using std::setw;

    const Scalar ntot = (nn.size() <= ntop) ?
        (accumulate(nn.begin(), nn.end(), 1e-8)) :
        (accumulate(nn.begin(), nn.begin() + ntop, 1e-8));

    ofs << "<histogram>" << std::endl;

    auto _print = [&](const Index j) {
        const Scalar x = nn.at(j);
        ofs << setw(10) << (j) << " [" << setw(10) << round(x) << "] ";
        for (int i = 0; i < ceil(x / ntot * height); ++i)
            ofs << "*";
        ofs << std::endl;
    };

    auto _args = std_argsort(nn);

    if (_args.size() <= ntop) {
        std::for_each(_args.begin(), _args.end(), _print);
    } else {
        std::for_each(_args.begin(), _args.begin() + ntop, _print);
    }
    ofs << "</histogram>" << std::endl;
}

#endif
