#ifndef MMUTIL_MATCHED_DATA_HH_
#define MMUTIL_MATCHED_DATA_HH_

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "util.hh"
#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_match.hh"

namespace mmutil { namespace match {

struct data_loader_t {

    using vs_type = hnswlib::InnerProductSpace;
    using str_vec_t = std::vector<std::string>;
    using idx_vec_t = std::vector<Index>;
    using num_vec_t = std::vector<Scalar>;

    explicit data_loader_t(const std::string _mtx_file,
                           const idx_vec_t &_mtx_idx_tab,
                           const KNN _knn,
                           const BILINK _bilink,
                           const NNLIST _nnlist)
        : mtx_file(_mtx_file)
        , mtx_idx_tab(_mtx_idx_tab)
        , knn(_knn.val)
        , param_bilink(_bilink.val)
        , param_nnlist(_nnlist.val)
    {
        mmutil::io::mm_info_reader_t info;
        CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
        CHECK(mtx_idx_tab.size() == info.max_col);
        D = info.max_row;
        Nsample = info.max_col;
        Nexposure = 0;
    }

    int set_exposure_info(const str_vec_t &);

    int build_dictionary(const Mat, const std::size_t NUM_THREADS);

    Mat read_counterfactual(const idx_vec_t &cells_j);
    Mat read(const idx_vec_t &cells_j);
    Mat read(const Index lb, const Index ub);

    Index num_exposure() const;
    Index exposure_group(const Index j) const;

    const std::string mtx_file;
    const idx_vec_t &mtx_idx_tab;

private:
    const std::size_t knn;
    std::size_t param_bilink;
    std::size_t param_nnlist;

    Index D;
    Index Nsample;
    Index Nexposure;

    str_vec_t exposure_id_name; // exposure names
    idx_vec_t exposure_map;     // map: col -> exposure index
    std::vector<idx_vec_t> exposure_index_set;
    Mat Q_kn; // rank x column matching data

    std::vector<std::shared_ptr<vs_type>> vs_vec_exposure;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_exposure;
};

}} // namespace mmutil::match
#endif
