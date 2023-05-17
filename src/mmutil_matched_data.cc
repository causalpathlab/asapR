#include "mmutil_matched_data.hh"

namespace mmutil { namespace match {

Mat
data_loader_t::read_counterfactual(const data_loader_t::idx_vec_t &cells_j)
{
    const std::size_t rank = Q_kn.rows();

    float *mass = Q_kn.data();
    const Index n_j = cells_j.size();
    Mat y = read(cells_j);
    Mat y0(D, n_j);
    y0.setZero();

    for (Index jth = 0; jth < n_j; ++jth) {    // For each cell j
        const Index _cell_j = cells_j.at(jth); //
        const Index tj =
            exposure_map.at(_cell_j); // Exposure group for this cell j
        const std::size_t n_j = cells_j.size(); // number of cells

        idx_vec_t counterfactual_neigh;
        num_vec_t dist_neigh, weights_neigh;

        ///////////////////////////////////////////////
        // Search neighbours in the other conditions //
        ///////////////////////////////////////////////

        for (Index ti = 0; ti < Nexposure; ++ti) {

            if (ti == tj) // skip the same condition
                continue; //

            const idx_vec_t &cells_i = exposure_index_set.at(ti);
            KnnAlg &alg_ti = *knn_lookup_exposure[ti].get();
            const std::size_t n_i = cells_i.size();
            const std::size_t nquery = std::min(knn, n_i);

            auto pq = alg_ti.searchKnn((void *)(mass + rank * _cell_j), nquery);

            while (!pq.empty()) {
                float d = 0;                         // distance
                std::size_t k;                       // local index
                std::tie(d, k) = pq.top();           //
                const Index _cell_i = cells_i.at(k); // global index
                if (_cell_j != _cell_i) {
                    counterfactual_neigh.emplace_back(_cell_i);
                    dist_neigh.emplace_back(d);
                }
                pq.pop();
            }
        }

        ////////////////////////////////////////////////////////
        // Find optimal weights for counterfactual imputation //
        ////////////////////////////////////////////////////////

        if (counterfactual_neigh.size() > 1) {
            Mat yy = y.col(jth);
            Index deg_ = counterfactual_neigh.size();
            weights_neigh.resize(deg_);
            normalize_weights(deg_, dist_neigh, weights_neigh);
            // Vec w0_ = eigen_vector(weights_neigh);
            const Eigen::Map<Vec> w0_(weights_neigh.data(), deg_);
            const Mat y0_ = read(counterfactual_neigh);
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

int
data_loader_t::build_dictionary(const Mat _Q_kn, const std::size_t NUM_THREADS)
{

    Q_kn.resize(_Q_kn.rows(), _Q_kn.cols());
    Q_kn = _Q_kn;

    ASSERT_RET(Q_kn.rows() > 0 && Q_kn.cols() > 0, "Empty Q_kn");
    ASSERT_RET(Q_kn.cols() == Nsample, "#columns(Q) != Nsample");
    const std::size_t rank = Q_kn.rows();

    if (param_bilink >= rank) {
        param_bilink = rank - 1;
    }

    if (param_bilink < 2) {
        param_bilink = 2; // WLOG("too small M value");
    }

    if (param_nnlist <= knn) {
        param_nnlist = knn + 1; // WLOG("too small N value");
    }

    TLOG("Building dictionaries for each exposure ...");

    for (Index tt = 0; tt < Nexposure; ++tt) {

        const Index n_tot = exposure_index_set[tt].size();

        vs_vec_exposure.emplace_back(std::make_shared<vs_type>(rank));

        vs_type &VS = *vs_vec_exposure[tt].get();

        knn_lookup_exposure.emplace_back(
            std::make_shared<KnnAlg>(&VS, n_tot, param_bilink, param_nnlist));
    }

    for (Index tt = 0; tt < Nexposure; ++tt) {
        const Index n_tot = exposure_index_set[tt].size(); // # cells
        KnnAlg &alg = *knn_lookup_exposure[tt].get();      // lookup
        const float *mass = Q_kn.data();                   // raw data

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
        for (Index i = 0; i < n_tot; ++i) {
            const Index cell_j = exposure_index_set.at(tt).at(i);
            alg.addPoint((void *)(mass + rank * cell_j), i);
        }
        TLOG("Built the dictionary [" << (tt + 1) << " / " << Nexposure << "]");
    }

    ASSERT_RET(knn_lookup_exposure.size() > 0, "Failed to build look-up");

    return EXIT_SUCCESS;
}

}} // namespace mmutil::match
