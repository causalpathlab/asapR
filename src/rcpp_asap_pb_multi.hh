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
            std::vector<S> vv;
            CHECK(read_vector_file(f, vv));
            const std::size_t sz = vv.size();

            std::sort(vv.begin(), vv.end());
            vv.erase(std::unique(vv.begin(), vv.end()), vv.end());
            WLOG_(vv.size() < sz, "Duplicate in \"" << f << "\"");

            for (auto r : vv)
                _rows.insert(r);
        };

        std::for_each(row_files.begin(), row_files.end(), _insert);
        pos2row.reserve(_rows.size());
        std::copy(_rows.begin(), _rows.end(), std::back_inserter(pos2row));

    } else {

        const std::size_t B = row_files.size();
        std::unordered_map<S, std::size_t> nn;

        for (std::size_t b = 0; b < B; ++b) {
            std::vector<S> vv;
            CHECK(read_vector_file(row_files[b], vv));

            const std::size_t sz = vv.size();
            std::sort(vv.begin(), vv.end());
            vv.erase(std::unique(vv.begin(), vv.end()), vv.end());
            WLOG_(vv.size() < sz, "Duplicate in \"" << row_files.at(b) << "\"");

            for (S x : vv) {
                if (nn.count(x) == 0) {
                    nn[x] = 1;
                } else {
                    nn[x] = nn[x] + 1;
                }
            }
        }

        pos2row.reserve(nn.size());

        for (auto &it : nn) {
            if (it.second >= B) {
                pos2row.emplace_back(it.first);
            }
        }
    }

    std::sort(pos2row.begin(), pos2row.end());
    for (I r = 0; r < pos2row.size(); ++r) {
        const S &s = pos2row.at(r);
        row2pos[s] = r;
    }
}

#endif
