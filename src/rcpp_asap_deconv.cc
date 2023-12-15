#include "rcpp_asap_deconv.hh"

//' Topic deconvolution
//'
//' @param bulk_dm Convoluted bulk data
//' @param bulk_row_names row names of bulk_data (D vector)
//' @param log_beta log dictionary matrix (D x K)
//' @param beta_row_names row names log_beta (D vector)
//' @param Ref_nk reference correlation data (N x K)
//' @param do_stdize_beta use standardized log_beta (default: TRUE)
//'
//'
// [[Rcpp::export]]
Rcpp::List
asap_deconv_stat(const Eigen::MatrixXf bulk_dm,
                 const Rcpp::StringVector bulk_row_names,
                 const Eigen::MatrixXf log_beta,
                 const Rcpp::StringVector beta_row_names,
                 const Eigen::MatrixXf Ref_nk,
                 const std::size_t rseed = 42,
                 const bool verbose = true,
                 const std::size_t NUM_THREADS = 1,
                 const std::size_t KNN_PER_SAMPLE = 15,
                 const std::size_t BATCH_ADJ_ITER = 100,
                 const double a0 = 1,
                 const double b0 = 1)
{
    // Step 0. Compute bulk-level correlation bulk_mk
    //
    // Step 1. Find nearest neighbour cells implicated in Ref_nk
    // profiles and estimate counterfactual impute_dm
    //
    // - corr_mk --> softmax --> theta_mk
    // - We can use impute_dm = beta_dk * theta_mk
    //
    //
    // Step 2. Estimate mu_d and delta_d by the alternating adjustment
    // method
    //
    // Step 3. Regress out (t(bulk_dm) * log_delta_d) from corr_mk
    //
}
