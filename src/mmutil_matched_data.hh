#ifndef MMUTIL_MATCHED_DATA_HH_
#define MMUTIL_MATCHED_DATA_HH_

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "util.hh"
#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_match.hh"

#include "RcppAnnoy.h"

namespace mmutil { namespace match {

#define ANNOYLIB_MULTITHREADED_BUILD 1

struct data_loader_t {

    using annoy_t = Scalar;
    using annoy_index_t = AnnoyIndex<Index,
                                     Scalar,
                                     Euclidean,
                                     Kiss64Random,
                                     RcppAnnoyIndexThreadPolicy>;

    using str_vec_t = std::vector<std::string>;
    using idx_vec_t = std::vector<Index>;
    using num_vec_t = std::vector<Scalar>;

    explicit data_loader_t(const std::string _mtx_file,
                           const idx_vec_t &_mtx_idx_tab,
                           const KNN _knn)
        : mtx_file(_mtx_file)
        , mtx_idx_tab(_mtx_idx_tab)
        , knn(_knn.val)
    {
        mmutil::io::mm_info_reader_t info;
        CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
        D = info.max_row;
        Nsample = info.max_col;
        Nexposure = 0;
    }

    int set_exposure_info(const str_vec_t &);

    Mat read_counterfactual(const idx_vec_t &cells_j);
    Mat read(const idx_vec_t &cells_j);
    Mat read(const Index lb, const Index ub);

    Index num_exposure() const;
    Index exposure_group(const Index j) const;

    const std::string mtx_file;
    const idx_vec_t &mtx_idx_tab;
    const str_vec_t &get_exposure_names() const;
    const idx_vec_t &get_exposure_mapping() const;

    int build_annoy_index(const Mat _Q_kn, const std::size_t NUM_THREADS);

private:
    const std::size_t knn;

    Index D;
    Index Nsample;
    Index Nexposure;

    str_vec_t exposure_id_name; // exposure names
    idx_vec_t exposure_map;     // map: col -> exposure index
    std::vector<idx_vec_t> exposure_index_set;
    Mat Q_kn; // rank x column matching data

    std::vector<std::shared_ptr<annoy_index_t>> annoy_indexes;
};

}} // namespace mmutil::match
#endif
