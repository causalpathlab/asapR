// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// asap_fit_nmf
Rcpp::List asap_fit_nmf(const Eigen::MatrixXf Y, const std::size_t maxK, const std::size_t mcem, const std::size_t burnin, const std::size_t latent_iter, const std::size_t thining, const bool verbose, const bool eval_llik, const double a0, const double b0, const std::size_t rseed, const std::size_t NUM_THREADS, const bool update_loading, const bool gibbs_sampling);
RcppExport SEXP _asapR_asap_fit_nmf(SEXP YSEXP, SEXP maxKSEXP, SEXP mcemSEXP, SEXP burninSEXP, SEXP latent_iterSEXP, SEXP thiningSEXP, SEXP verboseSEXP, SEXP eval_llikSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP rseedSEXP, SEXP NUM_THREADSSEXP, SEXP update_loadingSEXP, SEXP gibbs_samplingSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type maxK(maxKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type mcem(mcemSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type latent_iter(latent_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type thining(thiningSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const bool >::type eval_llik(eval_llikSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const bool >::type update_loading(update_loadingSEXP);
    Rcpp::traits::input_parameter< const bool >::type gibbs_sampling(gibbs_samplingSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_fit_nmf(Y, maxK, mcem, burnin, latent_iter, thining, verbose, eval_llik, a0, b0, rseed, NUM_THREADS, update_loading, gibbs_sampling));
    return rcpp_result_gen;
END_RCPP
}
// asap_fit_nmf_alternate
Rcpp::List asap_fit_nmf_alternate(const Eigen::MatrixXf Y_, const std::size_t maxK, const std::size_t max_iter, const std::size_t burnin, const bool verbose, const double a0, const double b0, const bool do_log1p, const std::size_t rseed, const double EPS, const double rate_m, const double rate_v, const bool svd_init);
RcppExport SEXP _asapR_asap_fit_nmf_alternate(SEXP Y_SEXP, SEXP maxKSEXP, SEXP max_iterSEXP, SEXP burninSEXP, SEXP verboseSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP do_log1pSEXP, SEXP rseedSEXP, SEXP EPSSEXP, SEXP rate_mSEXP, SEXP rate_vSEXP, SEXP svd_initSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type Y_(Y_SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type maxK(maxKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const bool >::type do_log1p(do_log1pSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const double >::type EPS(EPSSEXP);
    Rcpp::traits::input_parameter< const double >::type rate_m(rate_mSEXP);
    Rcpp::traits::input_parameter< const double >::type rate_v(rate_vSEXP);
    Rcpp::traits::input_parameter< const bool >::type svd_init(svd_initSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_fit_nmf_alternate(Y_, maxK, max_iter, burnin, verbose, a0, b0, do_log1p, rseed, EPS, rate_m, rate_v, svd_init));
    return rcpp_result_gen;
END_RCPP
}
// asap_random_bulk_data
Rcpp::List asap_random_bulk_data(const std::string mtx_file, const std::string mtx_idx_file, const std::size_t num_factors, const Rcpp::Nullable<Rcpp::NumericMatrix> r_covar, const Rcpp::Nullable<Rcpp::StringVector> r_batch, const std::size_t rseed, const bool verbose, const std::size_t NUM_THREADS, const std::size_t BLOCK_SIZE, const bool do_normalize, const bool do_log1p, const bool do_row_std, const std::size_t KNN_CELL);
RcppExport SEXP _asapR_asap_random_bulk_data(SEXP mtx_fileSEXP, SEXP mtx_idx_fileSEXP, SEXP num_factorsSEXP, SEXP r_covarSEXP, SEXP r_batchSEXP, SEXP rseedSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP BLOCK_SIZESEXP, SEXP do_normalizeSEXP, SEXP do_log1pSEXP, SEXP do_row_stdSEXP, SEXP KNN_CELLSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type mtx_idx_file(mtx_idx_fileSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type num_factors(num_factorsSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_covar(r_covarSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::StringVector> >::type r_batch(r_batchSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    Rcpp::traits::input_parameter< const bool >::type do_normalize(do_normalizeSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_log1p(do_log1pSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_row_std(do_row_stdSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_CELL(KNN_CELLSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_random_bulk_data(mtx_file, mtx_idx_file, num_factors, r_covar, r_batch, rseed, verbose, NUM_THREADS, BLOCK_SIZE, do_normalize, do_log1p, do_row_std, KNN_CELL));
    return rcpp_result_gen;
END_RCPP
}
// asap_regression
Rcpp::List asap_regression(const Eigen::MatrixXf Y_, const Eigen::MatrixXf log_x, const double a0, const double b0, const std::size_t max_iter, const bool do_log1p, const bool verbose, const bool do_stdize_x, const bool std_topic_latent);
RcppExport SEXP _asapR_asap_regression(SEXP Y_SEXP, SEXP log_xSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP max_iterSEXP, SEXP do_log1pSEXP, SEXP verboseSEXP, SEXP do_stdize_xSEXP, SEXP std_topic_latentSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type Y_(Y_SEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type log_x(log_xSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_log1p(do_log1pSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_stdize_x(do_stdize_xSEXP);
    Rcpp::traits::input_parameter< const bool >::type std_topic_latent(std_topic_latentSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_regression(Y_, log_x, a0, b0, max_iter, do_log1p, verbose, do_stdize_x, std_topic_latent));
    return rcpp_result_gen;
END_RCPP
}
// asap_regression_mtx
Rcpp::List asap_regression_mtx(const std::string mtx_file, const std::string mtx_idx_file, const Eigen::MatrixXf log_x, const Rcpp::Nullable<Rcpp::NumericMatrix> r_batch_effect, const Rcpp::Nullable<Rcpp::IntegerVector> r_batch_membership, const Rcpp::Nullable<Rcpp::StringVector> r_x_row_names, const Rcpp::Nullable<Rcpp::StringVector> r_mtx_row_names, const Rcpp::Nullable<Rcpp::StringVector> r_taboo_names, const double a0, const double b0, const std::size_t max_iter, const bool do_log1p, const bool verbose, const std::size_t NUM_THREADS, const std::size_t BLOCK_SIZE, const double max_depth, const bool do_stdize_x, const bool std_topic_latent);
RcppExport SEXP _asapR_asap_regression_mtx(SEXP mtx_fileSEXP, SEXP mtx_idx_fileSEXP, SEXP log_xSEXP, SEXP r_batch_effectSEXP, SEXP r_batch_membershipSEXP, SEXP r_x_row_namesSEXP, SEXP r_mtx_row_namesSEXP, SEXP r_taboo_namesSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP max_iterSEXP, SEXP do_log1pSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP BLOCK_SIZESEXP, SEXP max_depthSEXP, SEXP do_stdize_xSEXP, SEXP std_topic_latentSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type mtx_idx_file(mtx_idx_fileSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type log_x(log_xSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_batch_effect(r_batch_effectSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::IntegerVector> >::type r_batch_membership(r_batch_membershipSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::StringVector> >::type r_x_row_names(r_x_row_namesSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::StringVector> >::type r_mtx_row_names(r_mtx_row_namesSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::StringVector> >::type r_taboo_names(r_taboo_namesSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_log1p(do_log1pSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    Rcpp::traits::input_parameter< const double >::type max_depth(max_depthSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_stdize_x(do_stdize_xSEXP);
    Rcpp::traits::input_parameter< const bool >::type std_topic_latent(std_topic_latentSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_regression_mtx(mtx_file, mtx_idx_file, log_x, r_batch_effect, r_batch_membership, r_x_row_names, r_mtx_row_names, r_taboo_names, a0, b0, max_iter, do_log1p, verbose, NUM_THREADS, BLOCK_SIZE, max_depth, do_stdize_x, std_topic_latent));
    return rcpp_result_gen;
END_RCPP
}
// fit_poisson_cluster_rows
Rcpp::List fit_poisson_cluster_rows(const Eigen::MatrixXf& X, const std::size_t Ltrunc, const double alpha, const double a0, const double b0, const std::size_t rseed, const std::size_t mcmc, const std::size_t burnin, const bool verbose);
RcppExport SEXP _asapR_fit_poisson_cluster_rows(SEXP XSEXP, SEXP LtruncSEXP, SEXP alphaSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP rseedSEXP, SEXP mcmcSEXP, SEXP burninSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type Ltrunc(LtruncSEXP);
    Rcpp::traits::input_parameter< const double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_poisson_cluster_rows(X, Ltrunc, alpha, a0, b0, rseed, mcmc, burnin, verbose));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_build_index
int mmutil_build_index(const std::string mtx_file, const std::string index_file);
RcppExport SEXP _asapR_mmutil_build_index(SEXP mtx_fileSEXP, SEXP index_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type index_file(index_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_build_index(mtx_file, index_file));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_read_index
Rcpp::NumericVector mmutil_read_index(const std::string index_file);
RcppExport SEXP _asapR_mmutil_read_index(SEXP index_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type index_file(index_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_read_index(index_file));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_check_index
bool mmutil_check_index(const std::string mtx_file, const Rcpp::NumericVector& index_tab);
RcppExport SEXP _asapR_mmutil_check_index(SEXP mtx_fileSEXP, SEXP index_tabSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type index_tab(index_tabSEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_check_index(mtx_file, index_tab));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_info
Rcpp::List mmutil_info(const std::string mtx_file);
RcppExport SEXP _asapR_mmutil_info(SEXP mtx_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_info(mtx_file));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_write_mtx
int mmutil_write_mtx(const Eigen::SparseMatrix<float, Eigen::ColMajor>& X, const std::string mtx_file);
RcppExport SEXP _asapR_mmutil_write_mtx(SEXP XSEXP, SEXP mtx_fileSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<float, Eigen::ColMajor>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_write_mtx(X, mtx_file));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_read_columns_sparse
Rcpp::List mmutil_read_columns_sparse(const std::string mtx_file, const Rcpp::NumericVector& memory_location, const Rcpp::NumericVector& r_column_index, const bool verbose);
RcppExport SEXP _asapR_mmutil_read_columns_sparse(SEXP mtx_fileSEXP, SEXP memory_locationSEXP, SEXP r_column_indexSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type memory_location(memory_locationSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type r_column_index(r_column_indexSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_read_columns_sparse(mtx_file, memory_location, r_column_index, verbose));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_read_columns
Rcpp::NumericMatrix mmutil_read_columns(const std::string mtx_file, const Rcpp::NumericVector& memory_location, const Rcpp::NumericVector& r_column_index, const bool verbose);
RcppExport SEXP _asapR_mmutil_read_columns(SEXP mtx_fileSEXP, SEXP memory_locationSEXP, SEXP r_column_indexSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type memory_location(memory_locationSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type r_column_index(r_column_indexSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_read_columns(mtx_file, memory_location, r_column_index, verbose));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_read_rows_columns
Rcpp::NumericMatrix mmutil_read_rows_columns(const std::string mtx_file, const Rcpp::NumericVector& memory_location, const Rcpp::NumericVector& r_row_index, const Rcpp::NumericVector& r_column_index, const bool verbose);
RcppExport SEXP _asapR_mmutil_read_rows_columns(SEXP mtx_fileSEXP, SEXP memory_locationSEXP, SEXP r_row_indexSEXP, SEXP r_column_indexSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type memory_location(memory_locationSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type r_row_index(r_row_indexSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type r_column_index(r_column_indexSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_read_rows_columns(mtx_file, memory_location, r_row_index, r_column_index, verbose));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_simulate_poisson_mixture
Rcpp::List mmutil_simulate_poisson_mixture(const Rcpp::List r_mu_list, const std::size_t Ncell, const std::string output, const float dir_alpha, const float gam_alpha, const float gam_beta, const std::size_t rseed);
RcppExport SEXP _asapR_mmutil_simulate_poisson_mixture(SEXP r_mu_listSEXP, SEXP NcellSEXP, SEXP outputSEXP, SEXP dir_alphaSEXP, SEXP gam_alphaSEXP, SEXP gam_betaSEXP, SEXP rseedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_mu_list(r_mu_listSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type Ncell(NcellSEXP);
    Rcpp::traits::input_parameter< const std::string >::type output(outputSEXP);
    Rcpp::traits::input_parameter< const float >::type dir_alpha(dir_alphaSEXP);
    Rcpp::traits::input_parameter< const float >::type gam_alpha(gam_alphaSEXP);
    Rcpp::traits::input_parameter< const float >::type gam_beta(gam_betaSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_simulate_poisson_mixture(r_mu_list, Ncell, output, dir_alpha, gam_alpha, gam_beta, rseed));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_simulate_poisson
Rcpp::List mmutil_simulate_poisson(const Eigen::MatrixXf mu, const Eigen::VectorXf rho, const std::string output, Rcpp::Nullable<Rcpp::IntegerVector> r_indv, const std::size_t rseed);
RcppExport SEXP _asapR_mmutil_simulate_poisson(SEXP muSEXP, SEXP rhoSEXP, SEXP outputSEXP, SEXP r_indvSEXP, SEXP rseedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type mu(muSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXf >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< const std::string >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::IntegerVector> >::type r_indv(r_indvSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_simulate_poisson(mu, rho, output, r_indv, rseed));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_asapR_asap_fit_nmf", (DL_FUNC) &_asapR_asap_fit_nmf, 14},
    {"_asapR_asap_fit_nmf_alternate", (DL_FUNC) &_asapR_asap_fit_nmf_alternate, 13},
    {"_asapR_asap_random_bulk_data", (DL_FUNC) &_asapR_asap_random_bulk_data, 13},
    {"_asapR_asap_regression", (DL_FUNC) &_asapR_asap_regression, 9},
    {"_asapR_asap_regression_mtx", (DL_FUNC) &_asapR_asap_regression_mtx, 18},
    {"_asapR_fit_poisson_cluster_rows", (DL_FUNC) &_asapR_fit_poisson_cluster_rows, 9},
    {"_asapR_mmutil_build_index", (DL_FUNC) &_asapR_mmutil_build_index, 2},
    {"_asapR_mmutil_read_index", (DL_FUNC) &_asapR_mmutil_read_index, 1},
    {"_asapR_mmutil_check_index", (DL_FUNC) &_asapR_mmutil_check_index, 2},
    {"_asapR_mmutil_info", (DL_FUNC) &_asapR_mmutil_info, 1},
    {"_asapR_mmutil_write_mtx", (DL_FUNC) &_asapR_mmutil_write_mtx, 2},
    {"_asapR_mmutil_read_columns_sparse", (DL_FUNC) &_asapR_mmutil_read_columns_sparse, 4},
    {"_asapR_mmutil_read_columns", (DL_FUNC) &_asapR_mmutil_read_columns, 4},
    {"_asapR_mmutil_read_rows_columns", (DL_FUNC) &_asapR_mmutil_read_rows_columns, 5},
    {"_asapR_mmutil_simulate_poisson_mixture", (DL_FUNC) &_asapR_mmutil_simulate_poisson_mixture, 7},
    {"_asapR_mmutil_simulate_poisson", (DL_FUNC) &_asapR_mmutil_simulate_poisson, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_asapR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
