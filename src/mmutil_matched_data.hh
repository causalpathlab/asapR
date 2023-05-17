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
                           const str_vec_t &_mtx_cols,
                           const str_vec_t &_cols,
                           const KNN _knn,
                           const BILINK _bilink,
                           const NNLIST _nnlist)
        : mtx_file(_mtx_file)
        , mtx_idx_tab(_mtx_idx_tab)
        , mtx_cols(_mtx_cols)
        , cols(_cols)
        , Nsample(mtx_cols.size())
        , knn(_knn.val)
        , param_bilink(_bilink.val)
        , param_nnlist(_nnlist.val)
    {
        mmutil::io::mm_info_reader_t info;
        CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
        D = info.max_row;
        ASSERT(Nsample == info.max_col, "|mtx_cols| != mtx's max_col");
        Nexposure = 0;
    }

    void set_exposure_info(const str_vec_t &);

    int build_dictionary(const Rcpp::NumericMatrix r_V,
                         const std::size_t NUM_THREADS);

    Mat read_counterfactual(const idx_vec_t &cells_j);

    Mat read(const idx_vec_t &cells_j);

    Index num_exposure() const;

    const std::string mtx_file;
    const idx_vec_t &mtx_idx_tab;
    const str_vec_t &mtx_cols;
    const str_vec_t &cols;
    const Index Nsample;

private:
    Index D;
    Index Nexposure;

    str_vec_t exposure_id_name; // exposure names
    idx_vec_t exposure_map;     // map: col -> exposure index
    std::vector<idx_vec_t> exposure_index_set;
    Mat Vt; // rank x column matching data

    std::vector<std::shared_ptr<vs_type>> vs_vec_exposure;
    std::vector<std::shared_ptr<KnnAlg>> knn_lookup_exposure;

    const std::size_t knn;
    std::size_t param_bilink;
    std::size_t param_nnlist;
};

}} // namespace mmutil::match
#endif
