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
Rcpp::List asap_fit_nmf(Rcpp::NumericMatrix& Y_, const std::size_t maxK, const std::size_t max_iter, const Rcpp::Nullable<Rcpp::List> r_A_dd_list, const Rcpp::Nullable<Rcpp::List> r_A_nn_list, const std::size_t burnin, const bool verbose, const double a0, const double b0, const bool do_log1p, const std::size_t rseed, const bool svd_init, const double EPS, const std::size_t NUM_THREADS);
RcppExport SEXP _asapR_asap_fit_nmf(SEXP Y_SEXP, SEXP maxKSEXP, SEXP max_iterSEXP, SEXP r_A_dd_listSEXP, SEXP r_A_nn_listSEXP, SEXP burninSEXP, SEXP verboseSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP do_log1pSEXP, SEXP rseedSEXP, SEXP svd_initSEXP, SEXP EPSSEXP, SEXP NUM_THREADSSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix& >::type Y_(Y_SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type maxK(maxKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::List> >::type r_A_dd_list(r_A_dd_listSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::List> >::type r_A_nn_list(r_A_nn_listSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const bool >::type do_log1p(do_log1pSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const bool >::type svd_init(svd_initSEXP);
    Rcpp::traits::input_parameter< const double >::type EPS(EPSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_fit_nmf(Y_, maxK, max_iter, r_A_dd_list, r_A_nn_list, burnin, verbose, a0, b0, do_log1p, rseed, svd_init, EPS, NUM_THREADS));
    return rcpp_result_gen;
END_RCPP
}
// asap_fit_nmf_shared_dict
Rcpp::List asap_fit_nmf_shared_dict(const std::vector<Eigen::MatrixXf>& y_dn_vec, const std::size_t maxK, const std::size_t max_iter, const std::size_t burnin, const bool verbose, const double a0, const double b0, const bool do_log1p, const std::size_t rseed, const bool svd_init, const double EPS, const std::size_t NUM_THREADS);
RcppExport SEXP _asapR_asap_fit_nmf_shared_dict(SEXP y_dn_vecSEXP, SEXP maxKSEXP, SEXP max_iterSEXP, SEXP burninSEXP, SEXP verboseSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP do_log1pSEXP, SEXP rseedSEXP, SEXP svd_initSEXP, SEXP EPSSEXP, SEXP NUM_THREADSSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<Eigen::MatrixXf>& >::type y_dn_vec(y_dn_vecSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type maxK(maxKSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const bool >::type do_log1p(do_log1pSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const bool >::type svd_init(svd_initSEXP);
    Rcpp::traits::input_parameter< const double >::type EPS(EPSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_fit_nmf_shared_dict(y_dn_vec, maxK, max_iter, burnin, verbose, a0, b0, do_log1p, rseed, svd_init, EPS, NUM_THREADS));
    return rcpp_result_gen;
END_RCPP
}
// asap_random_bulk_data
Rcpp::List asap_random_bulk_data(const std::string mtx_file, const std::string row_file, const std::string col_file, const std::string idx_file, const std::size_t num_factors, const Rcpp::Nullable<Rcpp::NumericMatrix> r_covar_n, const Rcpp::Nullable<Rcpp::NumericMatrix> r_covar_d, const Rcpp::Nullable<Rcpp::StringVector> r_batch, const std::size_t rseed, const bool verbose, const std::size_t NUM_THREADS, const std::size_t BLOCK_SIZE, const bool do_log1p, const bool do_down_sample, const std::size_t KNN_CELL, const std::size_t CELL_PER_SAMPLE, const std::size_t BATCH_ADJ_ITER, const double a0, const double b0, const std::size_t MAX_ROW_WORD, const char ROW_WORD_SEP, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _asapR_asap_random_bulk_data(SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP, SEXP idx_fileSEXP, SEXP num_factorsSEXP, SEXP r_covar_nSEXP, SEXP r_covar_dSEXP, SEXP r_batchSEXP, SEXP rseedSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP BLOCK_SIZESEXP, SEXP do_log1pSEXP, SEXP do_down_sampleSEXP, SEXP KNN_CELLSEXP, SEXP CELL_PER_SAMPLESEXP, SEXP BATCH_ADJ_ITERSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP MAX_ROW_WORDSEXP, SEXP ROW_WORD_SEPSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type col_file(col_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type idx_file(idx_fileSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type num_factors(num_factorsSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_covar_n(r_covar_nSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::NumericMatrix> >::type r_covar_d(r_covar_dSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Nullable<Rcpp::StringVector> >::type r_batch(r_batchSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    Rcpp::traits::input_parameter< const bool >::type do_log1p(do_log1pSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_down_sample(do_down_sampleSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_CELL(KNN_CELLSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type CELL_PER_SAMPLE(CELL_PER_SAMPLESEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BATCH_ADJ_ITER(BATCH_ADJ_ITERSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_ROW_WORD(MAX_ROW_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type ROW_WORD_SEP(ROW_WORD_SEPSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_random_bulk_data(mtx_file, row_file, col_file, idx_file, num_factors, r_covar_n, r_covar_d, r_batch, rseed, verbose, NUM_THREADS, BLOCK_SIZE, do_log1p, do_down_sample, KNN_CELL, CELL_PER_SAMPLE, BATCH_ADJ_ITER, a0, b0, MAX_ROW_WORD, ROW_WORD_SEP, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// asap_random_bulk_data_multi
Rcpp::List asap_random_bulk_data_multi(const std::vector<std::string> mtx_files, const std::vector<std::string> row_files, const std::vector<std::string> col_files, const std::vector<std::string> idx_files, const std::size_t num_factors, const bool take_union_rows, const std::size_t rseed, const bool verbose, const std::size_t NUM_THREADS, const std::size_t BLOCK_SIZE, const bool do_batch_adj, const bool do_log1p, const bool do_down_sample, const std::size_t KNN_CELL, const std::size_t CELL_PER_SAMPLE, const std::size_t BATCH_ADJ_ITER, const double a0, const double b0, const std::size_t MAX_ROW_WORD, const char ROW_WORD_SEP, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _asapR_asap_random_bulk_data_multi(SEXP mtx_filesSEXP, SEXP row_filesSEXP, SEXP col_filesSEXP, SEXP idx_filesSEXP, SEXP num_factorsSEXP, SEXP take_union_rowsSEXP, SEXP rseedSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP BLOCK_SIZESEXP, SEXP do_batch_adjSEXP, SEXP do_log1pSEXP, SEXP do_down_sampleSEXP, SEXP KNN_CELLSEXP, SEXP CELL_PER_SAMPLESEXP, SEXP BATCH_ADJ_ITERSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP MAX_ROW_WORDSEXP, SEXP ROW_WORD_SEPSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<std::string> >::type mtx_files(mtx_filesSEXP);
    Rcpp::traits::input_parameter< const std::vector<std::string> >::type row_files(row_filesSEXP);
    Rcpp::traits::input_parameter< const std::vector<std::string> >::type col_files(col_filesSEXP);
    Rcpp::traits::input_parameter< const std::vector<std::string> >::type idx_files(idx_filesSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type num_factors(num_factorsSEXP);
    Rcpp::traits::input_parameter< const bool >::type take_union_rows(take_union_rowsSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type rseed(rseedSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    Rcpp::traits::input_parameter< const bool >::type do_batch_adj(do_batch_adjSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_log1p(do_log1pSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_down_sample(do_down_sampleSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_CELL(KNN_CELLSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type CELL_PER_SAMPLE(CELL_PER_SAMPLESEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BATCH_ADJ_ITER(BATCH_ADJ_ITERSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_ROW_WORD(MAX_ROW_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type ROW_WORD_SEP(ROW_WORD_SEPSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_random_bulk_data_multi(mtx_files, row_files, col_files, idx_files, num_factors, take_union_rows, rseed, verbose, NUM_THREADS, BLOCK_SIZE, do_batch_adj, do_log1p, do_down_sample, KNN_CELL, CELL_PER_SAMPLE, BATCH_ADJ_ITER, a0, b0, MAX_ROW_WORD, ROW_WORD_SEP, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// asap_topic_prop
Rcpp::List asap_topic_prop(const Eigen::MatrixXf X_dk, const Eigen::MatrixXf R_nk, const Eigen::MatrixXf Y_n, const double a0, const double b0, const std::size_t max_iter, const std::size_t NUM_THREADS, const bool verbose);
RcppExport SEXP _asapR_asap_topic_prop(SEXP X_dkSEXP, SEXP R_nkSEXP, SEXP Y_nSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP max_iterSEXP, SEXP NUM_THREADSSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type X_dk(X_dkSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type R_nk(R_nkSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type Y_n(Y_nSEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_topic_prop(X_dk, R_nk, Y_n, a0, b0, max_iter, NUM_THREADS, verbose));
    return rcpp_result_gen;
END_RCPP
}
// asap_adjust_corr_bbknn
Rcpp::List asap_adjust_corr_bbknn(const std::vector<Eigen::MatrixXf>& data_nk_vec, const std::size_t KNN_PER_BATCH, const std::size_t BLOCK_SIZE, const std::size_t NUM_THREADS, const bool verbose);
RcppExport SEXP _asapR_asap_adjust_corr_bbknn(SEXP data_nk_vecSEXP, SEXP KNN_PER_BATCHSEXP, SEXP BLOCK_SIZESEXP, SEXP NUM_THREADSSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<Eigen::MatrixXf>& >::type data_nk_vec(data_nk_vecSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type KNN_PER_BATCH(KNN_PER_BATCHSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_adjust_corr_bbknn(data_nk_vec, KNN_PER_BATCH, BLOCK_SIZE, NUM_THREADS, verbose));
    return rcpp_result_gen;
END_RCPP
}
// asap_topic_stat
Rcpp::List asap_topic_stat(const std::string mtx_file, const std::string row_file, const std::string col_file, const std::string idx_file, const Eigen::MatrixXf log_x, const Rcpp::StringVector& x_row_names, const bool do_log1p, const bool verbose, const std::size_t NUM_THREADS, const std::size_t BLOCK_SIZE, const std::size_t MAX_ROW_WORD, const char ROW_WORD_SEP, const std::size_t MAX_COL_WORD, const char COL_WORD_SEP);
RcppExport SEXP _asapR_asap_topic_stat(SEXP mtx_fileSEXP, SEXP row_fileSEXP, SEXP col_fileSEXP, SEXP idx_fileSEXP, SEXP log_xSEXP, SEXP x_row_namesSEXP, SEXP do_log1pSEXP, SEXP verboseSEXP, SEXP NUM_THREADSSEXP, SEXP BLOCK_SIZESEXP, SEXP MAX_ROW_WORDSEXP, SEXP ROW_WORD_SEPSEXP, SEXP MAX_COL_WORDSEXP, SEXP COL_WORD_SEPSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string >::type mtx_file(mtx_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type row_file(row_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type col_file(col_fileSEXP);
    Rcpp::traits::input_parameter< const std::string >::type idx_file(idx_fileSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type log_x(log_xSEXP);
    Rcpp::traits::input_parameter< const Rcpp::StringVector& >::type x_row_names(x_row_namesSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_log1p(do_log1pSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type NUM_THREADS(NUM_THREADSSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type BLOCK_SIZE(BLOCK_SIZESEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_ROW_WORD(MAX_ROW_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type ROW_WORD_SEP(ROW_WORD_SEPSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type MAX_COL_WORD(MAX_COL_WORDSEXP);
    Rcpp::traits::input_parameter< const char >::type COL_WORD_SEP(COL_WORD_SEPSEXP);
    rcpp_result_gen = Rcpp::wrap(asap_topic_stat(mtx_file, row_file, col_file, idx_file, log_x, x_row_names, do_log1p, verbose, NUM_THREADS, BLOCK_SIZE, MAX_ROW_WORD, ROW_WORD_SEP, MAX_COL_WORD, COL_WORD_SEP));
    return rcpp_result_gen;
END_RCPP
}
// asap_regression
Rcpp::List asap_regression(const Eigen::MatrixXf Y_, const Eigen::MatrixXf log_x, const double a0, const double b0, const std::size_t max_iter, const bool do_log1p, const bool verbose);
RcppExport SEXP _asapR_asap_regression(SEXP Y_SEXP, SEXP log_xSEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP max_iterSEXP, SEXP do_log1pSEXP, SEXP verboseSEXP) {
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
    rcpp_result_gen = Rcpp::wrap(asap_regression(Y_, log_x, a0, b0, max_iter, do_log1p, verbose));
    return rcpp_result_gen;
END_RCPP
}
// stretch_matrix_columns
Rcpp::NumericMatrix stretch_matrix_columns(const Eigen::MatrixXf Y, const double qq_min, const double qq_max, const double std_min, const double std_max, const bool verbose);
RcppExport SEXP _asapR_stretch_matrix_columns(SEXP YSEXP, SEXP qq_minSEXP, SEXP qq_maxSEXP, SEXP std_minSEXP, SEXP std_maxSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXf >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const double >::type qq_min(qq_minSEXP);
    Rcpp::traits::input_parameter< const double >::type qq_max(qq_maxSEXP);
    Rcpp::traits::input_parameter< const double >::type std_min(std_minSEXP);
    Rcpp::traits::input_parameter< const double >::type std_max(std_maxSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(stretch_matrix_columns(Y, qq_min, qq_max, std_min, std_max, verbose));
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
// decompose_network
Rcpp::List decompose_network(const Eigen::SparseMatrix<double, Eigen::ColMajor>& A_dd, const Eigen::MatrixXd& beta_dt, const double cutoff, const bool verbose);
RcppExport SEXP _asapR_decompose_network(SEXP A_ddSEXP, SEXP beta_dtSEXP, SEXP cutoffSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<double, Eigen::ColMajor>& >::type A_dd(A_ddSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type beta_dt(beta_dtSEXP);
    Rcpp::traits::input_parameter< const double >::type cutoff(cutoffSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(decompose_network(A_dd, beta_dt, cutoff, verbose));
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
    {"_asapR_asap_fit_nmf_shared_dict", (DL_FUNC) &_asapR_asap_fit_nmf_shared_dict, 12},
    {"_asapR_asap_random_bulk_data", (DL_FUNC) &_asapR_asap_random_bulk_data, 23},
    {"_asapR_asap_random_bulk_data_multi", (DL_FUNC) &_asapR_asap_random_bulk_data_multi, 22},
    {"_asapR_asap_topic_prop", (DL_FUNC) &_asapR_asap_topic_prop, 8},
    {"_asapR_asap_adjust_corr_bbknn", (DL_FUNC) &_asapR_asap_adjust_corr_bbknn, 5},
    {"_asapR_asap_topic_stat", (DL_FUNC) &_asapR_asap_topic_stat, 14},
    {"_asapR_asap_regression", (DL_FUNC) &_asapR_asap_regression, 7},
    {"_asapR_stretch_matrix_columns", (DL_FUNC) &_asapR_stretch_matrix_columns, 6},
    {"_asapR_fit_poisson_cluster_rows", (DL_FUNC) &_asapR_fit_poisson_cluster_rows, 9},
    {"_asapR_decompose_network", (DL_FUNC) &_asapR_decompose_network, 4},
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
