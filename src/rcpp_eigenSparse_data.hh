#include "rcpp_asap.hh"

#include "RcppAnnoy.h"
#define ANNOYLIB_MULTITHREADED_BUILD 1

#ifndef RCPP_EIGENSPARSE_DATA_HH_
#define RCPP_EIGENSPARSE_DATA_HH_

struct eigenSparse_data_t {

    using annoy_index_t = Annoy::AnnoyIndex<Index,
                                            Scalar,
                                            Annoy::Euclidean,
                                            Kiss64Random,
                                            RcppAnnoyIndexThreadPolicy>;

    explicit eigenSparse_data_t(const SpMat &_data,
                                const std::vector<std::string> &_row_names)
        : data(_data)
        , sub_rows(_row_names)
    {
        has_index = false;
        has_reloc = false;

        info.max_row = data.rows();
        info.max_col = data.cols();
        info.max_elem = data.nonZeros();

        ASSERT(data.rows() == sub_rows.size(),
               "row names should match with data.rows");

        A.resize(info.max_row, info.max_row);
        A.setZero();
    }

    template <typename Derived>
    int build_index(const Eigen::MatrixBase<Derived> &Q_kn_,
                    const bool verbose = true)
    {

        ASSERT_RET(info.max_col == Q_kn_.cols(),
                   "Incompatible Q with the underlying mtx data");

        Q_kn = Q_kn_;
        normalize_columns_inplace(Q_kn);
        const std::size_t rank = Q_kn.rows();
        index_ptr = std::make_shared<annoy_index_t>(rank);
        annoy_index_t &index = *index_ptr.get();
        std::vector<Scalar> vec(rank);
        for (Index j = 0; j < Q_kn.cols(); ++j) {
            Eigen::Map<Mat>(vec.data(), rank, 1) = Q_kn.col(j);
            index.add_item(j, vec.data());
        }
        index.build(50);
        TLOG_(verbose, "Populated " << Q_kn.cols() << " items");
        has_index = true;

        return EXIT_SUCCESS;
    }

    SpMat read(const Index lb, const Index ub);
    SpMat read_reloc(const Index lb, const Index ub);

    SpMat read(const std::vector<Index> &index);
    SpMat read_reloc(const std::vector<Index> &index);

    SpMat read_matched(const std::vector<Scalar> &query,
                       const std::size_t knn,
                       std::vector<Index> &neigh_index,
                       std::vector<Scalar> &neigh_dist);

    SpMat read_matched_reloc(const std::vector<Scalar> &query,
                             const std::size_t knn,
                             std::vector<Index> &neigh_index,
                             std::vector<Scalar> &neigh_dist);

    void relocate_rows(const std::unordered_map<std::string, Index> &row2pos);

    void knn_search(const std::vector<Scalar> &query,
                    const std::size_t knn,
                    std::vector<Index> &neigh_index,
                    std::vector<Scalar> &neigh_dist);

public:
    const SpMat &data;
    const std::vector<std::string> &sub_rows;

    mm_info_reader_t info;

private:
    SpMat A;

    std::shared_ptr<annoy_index_t> index_ptr;
    Mat Q_kn;
    bool has_index;
    bool has_reloc;
};

#endif
