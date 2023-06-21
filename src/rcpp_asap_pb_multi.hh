#include "rcpp_asap.hh"
#include "rcpp_asap_stat.hh"
#include "rcpp_asap_mtx_data.hh"

#ifndef RCPP_ASAP_PB_MULTI_HH_
#define RCPP_ASAP_PB_MULTI_HH_

template <typename S, typename I>
void
take_row_names(const std::vector<S> &row_files,
               std::vector<S> &pos2row,
               std::unordered_map<S, I> &row2pos,
               bool take_union)
{
    if (take_union) {
        std::unordered_set<S> _rows; // Take a unique set

        auto _insert = [&](S f) {
            std::vector<S> temp;
            CHECK(read_vector_file(f, temp));
            for (auto r : temp)
                _rows.insert(r);
        };

        std::for_each(row_files.begin(), row_files.end(), _insert);
        pos2row.reserve(_rows.size());
        std::copy(_rows.begin(), _rows.end(), std::back_inserter(pos2row));

    } else {

        std::unordered_map<S, std::size_t> nn;

        for (std::size_t j = 0; j < row_files.size(); ++j) {
            std::vector<S> temp;
            CHECK(read_vector_file(row_files[j], temp));

            for (S x : temp) {
                if (nn.count(x) == 0) {
                    nn[x] = 0;
                }
                nn[x]++;
            }
        }

        const std::size_t B = row_files.size();

        for (auto &it : nn) {
            if (it.second == B)
                pos2row.emplace_back(it.first);
        }
    }

    std::sort(pos2row.begin(), pos2row.end());
    std::unique(pos2row.begin(), pos2row.end());
    for (I r = 0; r < pos2row.size(); ++r) {
        row2pos[pos2row.at(r)] = r;
    }
}

#endif
