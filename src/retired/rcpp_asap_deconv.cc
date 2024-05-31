#include "rcpp_asap_deconv.hh"

//' Estimated deconvolved topic statistics
//'
//' @param bulk_dm Convoluted bulk data
//' @param bulk_row_names row names of bulk_data (D vector)
//' @param Ref_nk reference correlation data (N x K)
//' @param log_beta log dictionary matrix (D x K)
//' @param beta_row_names row names log_beta (D vector)
//' @param r_log_delta D x B log batch effect matrix
//' @param do_stdize_beta use standardized log_beta (default: TRUE)
//' @param do_log1p do log(1+y) transformation
//' @param verbose verbosity
//' @param NUM_THREADS number of threads in data reading
//' @param KNN_PER_SAMPLE k Nearest Neighbours per bulk sample
//'
//'
// [[Rcpp::export]]
Rcpp::List
asap_deconv_pmf_stat(
    const Eigen::MatrixXf bulk_mk,
    const Eigen::MatrixXf Ref_nk,
    const bool verbose = true,
    const std::size_t KNN_PER_SAMPLE = 15)
{

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;

    ////////////////////////////////////////////////////
    // Step 0. Compute bulk-level correlation bulk_mk //
    ////////////////////////////////////////////////////

    const std::size_t K = bulk_mk.cols();

log_beta.cols(), M = bulk_dm.cols();

    Mat bulk_mk(M, K), bulk_m(M, 1), Q_km(K, M); // query matrix K x M samples

    std::vector<std::string> bulk_pos2row;
    rcpp::util::copy(bulk_row_names, bulk_pos2row);

    ASSERT_RETL(bulk_dm.rows() == bulk_pos2row.size(),
                "bulk D x M needs " << bulk_pos2row.size() << " rows");

    eigenSparse_data_t bulk_data(bulk_dm, bulk_pos2row);

    if (r_log_delta.isNotNull()) {

        Mat log_delta = Rcpp::as<Mat>(r_log_delta);

        ASSERT_RETL(log_delta.rows() == log_beta.rows(),
                    "rows(delta) != rows(beta)");

        CHK_RETL_(run_pmf_stat_adj(bulk_data,
                                   log_beta,
                                   log_delta,
                                   pos2row,
                                   options,
                                   bulk_mk,
                                   bulk_m),
                  "failed to compute PMF stat in bulk data");

    } else {

        CHK_RETL_(run_pmf_stat(bulk_data,
                               log_beta,
                               pos2row,
                               options,
                               bulk_mk,
                               bulk_m),
                  "failed to compute PMF stat in bulk data");
    }

    Q_km = bulk_mk.transpose();
    normalize_columns_inplace(Q_km);

    TLOG_(verbose, "estimated PMF stat on the bulk data");

    ////////////////////////////////////////////////
    // Step 1. Construct ANNOY-indexed dictionary //
    ////////////////////////////////////////////////

    using annoy_index_t = Annoy::AnnoyIndex<Index,
                                            Scalar,
                                            Annoy::Euclidean,
                                            Kiss64Random,
                                            RcppAnnoyIndexThreadPolicy>;
    annoy_index_t ref_index(K);
    {
        Mat ref_kn = Ref_nk.transpose();
        normalize_columns_inplace(ref_kn);
        std::vector<Scalar> query(K);
        for (Index j = 0; j < ref_kn.cols(); ++j) {
            Eigen::Map<Mat>(query.data(), K, 1) = ref_kn.col(j);
            ref_index.add_item(j, query.data());
        }
        ref_index.build(50);
        TLOG_(verbose,
              "Added " << ref_kn.cols() << " data points in ANNOY index");
    }

    const std::size_t N = Ref_nk.rows();
    const std::size_t nsearch = std::min(KNN_PER_SAMPLE, N);

    exp_op<Mat> exp;
    softmax_op_t<Mat> softmax;

    Mat imputed_mk = Mat::Zero(M, K);
    RowVec delta_k = RowVec::Zero(K);
    Scalar denom = 0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
#endif
    for (Index s = 0; s < Q_km.cols(); ++s) {
        std::vector<Scalar> query(K);
        Eigen::Map<Mat>(query.data(), K, 1) = Q_km.col(s);
        std::vector<Index> neigh;
        std::vector<Scalar> dist, weight(nsearch, 0);

        ref_index.get_nns_by_vector(query.data(), nsearch, -1, &neigh, &dist);
        mmutil::match::normalize_weights(nsearch, dist, weight);

#pragma omp critical
        {
            for (Index j = 0; j < nsearch; ++j) {
                const Index i = dist.at(j);
                const Scalar w_ij = weight.at(j);
                imputed_mk.row(s) += Ref_nk.row(i) * w_ij;
            }
            const Scalar w_sum =
                std::accumulate(weight.begin(), weight.end(), 0.);
            imputed_mk.row(s) /= w_sum;

            delta_k += imputed_mk.row(s) - bulk_mk.row(s);
            denom += 1.;
        }
    } // end of each bulk sample

    delta_k /= denom;
    bulk_mk.rowwise() -= delta_k;

    TLOG_(verbose, "Done");

    return Rcpp::List::create(Rcpp::_["beta"] = log_beta.unaryExpr(exp),
                              Rcpp::_["corr"] = bulk_mk,
                              Rcpp::_["colsum"] = bulk_m,
                              Rcpp::_["rownames"] = pos2row,
                              Rcpp::_["discrepancy"] = delta_k);
}
