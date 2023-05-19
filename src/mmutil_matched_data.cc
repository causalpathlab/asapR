#include "mmutil_matched_data.hh"

namespace mmutil { namespace match {

Mat
data_loader_t::read_counterfactual(const data_loader_t::idx_vec_t &cells_j)
{
    const std::size_t rank = Q_kn.rows();

    const float *mass = Q_kn.data();
    const Index n_j = cells_j.size();
    Mat y = read(cells_j);
    Mat y0(D, n_j);
    y0.setZero();

    std::vector<Scalar> query(rank);

    for (Index jth = 0; jth < n_j; ++jth) {    // For each cell j
        const Index _cell_j = cells_j.at(jth); //
        const Index tj =
            exposure_map.at(_cell_j); // Exposure group for this cell j
        const std::size_t n_j = cells_j.size(); // number of cells

        idx_vec_t counterfactual_neigh;
        num_vec_t dist_neigh, weights_neigh;

        idx_vec_t neighbor_index;
        num_vec_t neighbor_dist;

        ///////////////////////////////////////////////
        // Search neighbours in the other conditions //
        ///////////////////////////////////////////////

        Eigen::Map<Mat>(query.data(), rank, 1) = Q_kn.col(_cell_j);

        for (Index ti = 0; ti < Nexposure; ++ti) {

            if (ti == tj) // skip the same condition
                continue; //

            const idx_vec_t &cells_i = exposure_index_set.at(ti);
            const std::size_t n_i = cells_i.size();
            const std::size_t nsearch = std::min(knn, n_i);

            annoy_index_t &index = *annoy_indexes[ti].get();

            index.get_nns_by_vector(query.data(),
                                    nsearch,
                                    -1,
                                    &neighbor_index,
                                    &neighbor_dist);

            for (Index k = 0; k < nsearch; ++k) {
                const Index kk = neighbor_index.at(k);
                const Scalar dd = neighbor_dist.at(k);
                const Index _cell_i = cells_i.at(kk);
                if (_cell_j != _cell_i) {
                    counterfactual_neigh.emplace_back(_cell_i);
                    dist_neigh.emplace_back(dd);
                }
            }
        }

        // Estimate counterfactual y

        if (counterfactual_neigh.size() > 1) {
            const Index deg_ = counterfactual_neigh.size();
            const Mat y0_ = read(counterfactual_neigh);

            weights_neigh.resize(deg_);
            normalize_weights(deg_, dist_neigh, weights_neigh);
            Vec w0_ = eigen_vector(weights_neigh);
            const Scalar denom = w0_.sum(); // must be > 0
            y0.col(jth) = y0_ * w0_ / denom;

        } else if (counterfactual_neigh.size() == 1) {
            y0.col(jth) = read(counterfactual_neigh).col(0);
        }
    }
    return y0;
}

Mat
data_loader_t::read(const data_loader_t::idx_vec_t &cells_j)
{
    return Mat(mmutil::io::read_eigen_sparse_subset_col(mtx_file,
                                                        mtx_idx_tab,
                                                        cells_j));
}

Mat
data_loader_t::read(const Index lb, const Index ub)
{
    ///////////////////////////////////////
    // memory location = 0 means the end //
    ///////////////////////////////////////
    const Index lb_mem = lb < Nsample ? mtx_idx_tab[lb] : 0;
    const Index ub_mem = ub < Nsample ? mtx_idx_tab[ub] : 0;
    return Mat(mmutil::io::read_eigen_sparse_subset_col(mtx_file,
                                                        lb,
                                                        ub,
                                                        lb_mem,
                                                        ub_mem));
}

int
data_loader_t::set_exposure_info(const data_loader_t::str_vec_t &exposure)
{
    Nexposure = 0;
    ASSERT_RET(Nsample == exposure.size(), "|cols| != |exposure|");

    std::tie(exposure_map, exposure_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(exposure);

    exposure_index_set = make_index_vec_vec(exposure_map);

    Nexposure = exposure_index_set.size();

    TLOG("Found " << Nexposure << " treatment groups");

    return EXIT_SUCCESS;
}

Index
data_loader_t::num_exposure() const
{
    return Nexposure;
}

Index
data_loader_t::exposure_group(const Index j) const
{
    return exposure_map.at(j);
}

const data_loader_t::str_vec_t &
data_loader_t::get_exposure_names() const
{
    return exposure_id_name;
}
const data_loader_t::idx_vec_t &
data_loader_t::get_exposure_mapping() const
{
    return exposure_map;
}

int
data_loader_t::build_annoy_index(const Mat _Q_kn, const std::size_t NUM_THREADS)
{
    Q_kn.resize(_Q_kn.rows(), _Q_kn.cols());
    Q_kn = _Q_kn;
    normalize_columns(Q_kn);
    const std::size_t rank = Q_kn.rows();
    TLOG("Building dictionaries for each exposure ...");

    for (Index tt = 0; tt < Nexposure; ++tt) {
        const Index n_tot = exposure_index_set[tt].size();
        annoy_indexes.emplace_back(std::make_shared<annoy_index_t>(rank));
        // annoy_index_t & index = *annoy_indexes[tt].get();
    }

    for (Index tt = 0; tt < Nexposure; ++tt) {
        const Index n_tot = exposure_index_set[tt].size(); // # cells
        annoy_index_t &index = *annoy_indexes[tt].get();
        std::vector<Scalar> vec(rank);
        for (Index i = 0; i < n_tot; ++i) {
            const Index cell_j = exposure_index_set.at(tt).at(i);
            Eigen::Map<Mat>(vec.data(), rank, 1) = Q_kn.col(cell_j);
            index.add_item(i, vec.data());
        }
    }
    for (Index tt = 0; tt < Nexposure; ++tt) {
        annoy_index_t &index = *annoy_indexes[tt].get();
        index.build(50);
    }
    return EXIT_SUCCESS;
}

}} // namespace mmutil::match
