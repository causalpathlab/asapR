// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// asap_fit_modular_nmf
Rcpp::List asap_fit_modular_nmf(const Eigen::MatrixXf Y, const std::size_t maxK, const std::size_t maxL, Rcpp::Nullable<Rcpp::NumericMatrix> collapsing, const std::size_t mcem, const std::size_t burnin, const std::size_t latent_iter, const std::size_t degree_iter, const std::size_t thining, const bool verbose, const bool eval_llik, const double a0, const double b0, const std::size_t rseed, const std::size_t NUM_THREADS, const bool update_loading, const bool gibbs_sampling);
RcppExport SEXP _asapR_asap_fit_modular_nmf(SEXP YSEXP, SEXP maxKSEXP, SEXP maxLSEXP, SEXP collapsingSEXP, SEXP mcemSEXP, SEXP burninSEXP, SEXP latent_iterSEXP, SEXP degree_iterSEXP, SEXP thiningSEXP, SEXP verboseSEXP, SEXP eval_llikSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP rseedSEXP, SEXP NUM_THREADSSEXP, SEXP update_loadingSEXP, SEXP gibbs_samplingSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type maxK(maxKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type maxL(maxLSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericMatrix> >::type collapsing(collapsingSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type mcem(mcemSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type latent_iter(latent_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type degree_iter(degree_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type thining(thiningSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const bool >::type eval_llik(eval_llikSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const bool >::type update_loading(update_loadingSEXP);
    Rcpp::traits::input_parameter< const bool >::type gibbs_sampling(gibbs_samplingSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_fit_modular_nmf(Y, maxK, maxL, collapsing, mcem, burnin, latent_iter, degree_iter, thining, verbose, eval_llik, a0, b0, rseed, NUM_THREADS, update_loading, gibbs_sampling));
    return rcpp_result_gen;
END_RCPP
}
// asap_fit_nmf
Rcpp::List asap_fit_nmf(const Eigen::MatrixXf Y, const std::size_t maxK, const std::size_t mcem, const std::size_t burnin, const std::size_t latent_iter, const std::size_t degree_iter, const std::size_t thining, const bool verbose, const bool eval_llik, const double a0, const double b0, const std::size_t rseed, const std::size_t NUM_THREADS, const bool update_loading, const bool gibbs_sampling);
RcppExport SEXP _asapR_asap_fit_nmf(SEXP YSEXP, SEXP maxKSEXP, SEXP mcemSEXP, SEXP burninSEXP, SEXP latent_iterSEXP, SEXP degree_iterSEXP, SEXP thiningSEXP, SEXP verboseSEXP, SEXP eval_llikSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP rseedSEXP, SEXP NUM_THREADSSEXP, SEXP update_loadingSEXP, SEXP gibbs_samplingSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type maxK(maxKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type mcem(mcemSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type latent_iter(latent_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type degree_iter(degree_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type thining(thiningSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const bool >::type eval_llik(eval_llikSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const bool >::type update_loading(update_loadingSEXP);
    Rcpp::traits::input_parameter< const bool >::type gibbs_sampling(gibbs_samplingSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_fit_nmf(Y, maxK, mcem, burnin, latent_iter, degree_iter, thining, verbose, eval_llik, a0, b0, rseed, NUM_THREADS, update_loading, gibbs_sampling));
    return rcpp_result_gen;
END_RCPP
}
// asap_random_bulk_data
Rcpp::List asap_random_bulk_data(const std::string mtx_file, const Rcpp::NumericVector& memory_location, const std::size_t num_factors, const std::size_t rseed, const bool verbose, const std::size_t NUM_THREADS, const std::size_t BLOCK_SIZE);
RcppExport SEXP _asapR_asap_random_bulk_data(SEXP mtx_fileSEXP, SEXP memory_locationSEXP, SEXP num_factorsSEXP, SEXP rseedSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP BLOCK_SIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type memory_location(memory_locationSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type num_factors(num_factorsSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    rcpp_result_gen = Rcpp::wrap(asap_random_bulk_data(mtx_file, memory_location, num_factors, rseed, verbose, NUM_THREADS, BLOCK_SIZE));
    return rcpp_result_gen;
END_RCPP
}
// asap_predict_mtx
Rcpp::List asap_predict_mtx(const std::string mtx_file, const Rcpp::NumericVector& memory_location, const Eigen::MatrixXf beta_dict, const bool do_beta_rescale, Rcpp::Nullable<Rcpp::NumericMatrix> collapsing, const std::size_t mcem, const std::size_t burnin, const std::size_t latent_iter, const std::size_t thining, const double a0, const double b0, const std::size_t rseed, const bool verbose, const std::size_t NUM_THREADS, const std::size_t BLOCK_SIZE, const bool gibbs_sampling);
RcppExport SEXP _asapR_asap_predict_mtx(SEXP mtx_fileSEXP, SEXP memory_locationSEXP, SEXP beta_dictSEXP, SEXP do_beta_rescaleSEXP, SEXP collapsingSEXP, SEXP mcemSEXP, SEXP burninSEXP, SEXP latent_iterSEXP, SEXP thiningSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP rseedSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP BLOCK_SIZESEXP, SEXP gibbs_samplingSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type memory_location(memory_locationSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type beta_dict(beta_dictSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_beta_rescale(do_beta_rescaleSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericMatrix> >::type collapsing(collapsingSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type mcem(mcemSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type latent_iter(latent_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type thining(thiningSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    Rcpp::traits::input_parameter< const bool >::type gibbs_sampling(gibbs_samplingSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_predict_mtx(mtx_file, memory_location, beta_dict, do_beta_rescale, collapsing, mcem, burnin, latent_iter, thining, a0, b0, rseed, verbose, NUM_THREADS, BLOCK_SIZE, gibbs_sampling));
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
int mmutil_check_index(const std::string mtx_file, const Rcpp::NumericVector& index_tab);
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
Rcpp::List mmutil_read_columns_sparse(const std::string mtx_file, const Rcpp::NumericVector& memory_location, const Rcpp::NumericVector& r_column_index, const bool verbose, const std::size_t NUM_THREADS, const std::size_t MIN_SIZE);
RcppExport SEXP _asapR_mmutil_read_columns_sparse(SEXP mtx_fileSEXP, SEXP memory_locationSEXP, SEXP r_column_indexSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP MIN_SIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type memory_location(memory_locationSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type r_column_index(r_column_indexSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MIN_SIZE(MIN_SIZESEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_read_columns_sparse(mtx_file, memory_location, r_column_index, verbose, NUM_THREADS, MIN_SIZE));
    return rcpp_result_gen;
END_RCPP
}
// mmutil_read_columns
Rcpp::NumericMatrix mmutil_read_columns(const std::string mtx_file, const Rcpp::NumericVector& memory_location, const Rcpp::NumericVector& r_column_index, const bool verbose, const std::size_t NUM_THREADS, const std::size_t MIN_SIZE);
RcppExport SEXP _asapR_mmutil_read_columns(SEXP mtx_fileSEXP, SEXP memory_locationSEXP, SEXP r_column_indexSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP MIN_SIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type memory_location(memory_locationSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type r_column_index(r_column_indexSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MIN_SIZE(MIN_SIZESEXP);
    rcpp_result_gen = Rcpp::wrap(mmutil_read_columns(mtx_file, memory_location, r_column_index, verbose, NUM_THREADS, MIN_SIZE));
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
    {"_asapR_asap_fit_modular_nmf", (DL_FUNC) &_asapR_asap_fit_modular_nmf, 17},
    {"_asapR_asap_fit_nmf", (DL_FUNC) &_asapR_asap_fit_nmf, 15},
    {"_asapR_asap_random_bulk_data", (DL_FUNC) &_asapR_asap_random_bulk_data, 7},
    {"_asapR_asap_predict_mtx", (DL_FUNC) &_asapR_asap_predict_mtx, 16},
    {"_asapR_fit_poisson_cluster_rows", (DL_FUNC) &_asapR_fit_poisson_cluster_rows, 9},
    {"_asapR_mmutil_build_index", (DL_FUNC) &_asapR_mmutil_build_index, 2},
    {"_asapR_mmutil_read_index", (DL_FUNC) &_asapR_mmutil_read_index, 1},
    {"_asapR_mmutil_check_index", (DL_FUNC) &_asapR_mmutil_check_index, 2},
    {"_asapR_mmutil_info", (DL_FUNC) &_asapR_mmutil_info, 1},
    {"_asapR_mmutil_write_mtx", (DL_FUNC) &_asapR_mmutil_write_mtx, 2},
    {"_asapR_mmutil_read_columns_sparse", (DL_FUNC) &_asapR_mmutil_read_columns_sparse, 6},
    {"_asapR_mmutil_read_columns", (DL_FUNC) &_asapR_mmutil_read_columns, 6},
    {"_asapR_mmutil_simulate_poisson_mixture", (DL_FUNC) &_asapR_mmutil_simulate_poisson_mixture, 7},
    {"_asapR_mmutil_simulate_poisson", (DL_FUNC) &_asapR_mmutil_simulate_poisson, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_asapR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
