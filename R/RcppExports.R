# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' Non-negative matrix factorization with the row modules
#'
#' @param Y data matrix (gene x sample)
#' @param maxK maximum number of factors
#' @param maxL maximum number of row modules
#' @param collapsing L x row collapsing matrix (L < gene)
#' @param mcem number of Monte Carl Expectation Maximization
#' @param burnin burn-in period
#' @param latent_iter latent sampling steps
#' @param thining thining interval in record keeping
#' @param verbose verbosity
#' @param eval_llik evaluate log-likelihood
#' @param a0 gamma(a0, b0)
#' @param b0 gamma(a0, b0)
#' @param rseed random seed
#' @param NUM_THREADS number of parallel jobs
#'
asap_fit_modular_nmf <- function(Y, maxK, maxL, collapsing = NULL, mcem = 100L, burnin = 10L, latent_iter = 10L, thining = 3L, verbose = TRUE, eval_llik = TRUE, a0 = 1., b0 = 1., rseed = 42L, NUM_THREADS = 1L, update_loading = TRUE, gibbs_sampling = FALSE) {
    .Call('_asapR_asap_fit_modular_nmf', PACKAGE = 'asapR', Y, maxK, maxL, collapsing, mcem, burnin, latent_iter, thining, verbose, eval_llik, a0, b0, rseed, NUM_THREADS, update_loading, gibbs_sampling)
}

#' Non-negative matrix factorization
#'
#' @param Y data matrix (gene x sample)
#' @param maxK maximum number of factors
#' @param mcem number of Monte Carl Expectation Maximization
#' @param burnin burn-in period
#' @param latent_iter latent sampling steps
#' @param thining thining interval in record keeping
#' @param verbose verbosity
#' @param eval_llik evaluate log-likelihood
#' @param a0 gamma(a0, b0)
#' @param b0 gamma(a0, b0)
#' @param rseed random seed
#' @param NUM_THREADS number of parallel jobs
#'
asap_fit_nmf <- function(Y, maxK, mcem = 100L, burnin = 10L, latent_iter = 10L, thining = 3L, verbose = TRUE, eval_llik = TRUE, a0 = 1., b0 = 1., rseed = 42L, NUM_THREADS = 1L, update_loading = TRUE, gibbs_sampling = FALSE) {
    .Call('_asapR_asap_fit_nmf', PACKAGE = 'asapR', Y, maxK, mcem, burnin, latent_iter, thining, verbose, eval_llik, a0, b0, rseed, NUM_THREADS, update_loading, gibbs_sampling)
}

#' A quick NMF estimation based on alternating Poisson regressions
#'
#' @param Y_dn non-negative data matrix (gene x sample)
#' @param maxK maximum number of factors
#' @param max_iter max number of optimization steps
#' @param min_iter min number of optimization steps
#' @param burnin number of initiation steps
#' @param verbose verbosity
#' @param a0 gamma(a0, b0) default: a0 = 1
#' @param b0 gamma(a0, b0) default: b0 = 1
#' @param rseed random seed (default: 1337)
#'
#' @return a list that contains:
#'  \itemize{
#'   \item log.likelihood log-likelihood trace
#'   \item beta dictionary (gene x factor)
#'   \item log.beta log-dictionary (gene x factor)
#'   \item theta loading (sample x factor)
#'   \item log.theta log-loading (sample x factor)
#'   \item log.phi auxiliary variables (gene x factor)
#'   \item log.rho auxiliary variables (sample x factor)
#' }
#'
#'
asap_fit_nmf_alternate <- function(Y_dn, maxK, max_iter = 100L, burnin = 10L, verbose = TRUE, a0 = 1, b0 = 1, rseed = 1337L, EPS = 1e-6, rate_m = 1, rate_v = 1, svd_init = TRUE) {
    .Call('_asapR_asap_fit_nmf_alternate', PACKAGE = 'asapR', Y_dn, maxK, max_iter, burnin, verbose, a0, b0, rseed, EPS, rate_m, rate_v, svd_init)
}

#' Predict NMF loading -- this may be slow for high-dim data
#'
#' @param mtx_file matrix-market-formatted data file (bgzip)
#' @param memory_location column indexing for the mtx
#' @param beta_dict row x factor dictionary (beta) matrix
#' @param do_beta_rescale rescale the columns of the beta matrix
#' @param collapsing r x row collapsing matrix (r < row)
#' @param mcem number of Monte Carlo Expectation Maximization
#' @param burnin burn-in period
#' @param latent_iter latent sampling steps
#' @param thining thining interval in record keeping
#' @param a0 gamma(a0, b0)
#' @param b0 gamma(a0, b0)
#' @param rseed random seed
#' @param verbose verbosity
#' @param NUM_THREADS number of threads in data reading
#' @param BLOCK_SIZE disk I/O block size (number of columns)
#'
asap_predict_mtx <- function(mtx_file, memory_location, beta_dict, do_beta_rescale = TRUE, collapsing = NULL, mcem = 100L, burnin = 10L, latent_iter = 10L, thining = 3L, a0 = 1., b0 = 1., rseed = 42L, verbose = FALSE, NUM_THREADS = 1L, BLOCK_SIZE = 100L, gibbs_sampling = FALSE) {
    .Call('_asapR_asap_predict_mtx', PACKAGE = 'asapR', mtx_file, memory_location, beta_dict, do_beta_rescale, collapsing, mcem, burnin, latent_iter, thining, a0, b0, rseed, verbose, NUM_THREADS, BLOCK_SIZE, gibbs_sampling)
}

#' Generate approximate pseudo-bulk data by random projections
#'
#' @param mtx_file matrix-market-formatted data file (bgzip)
#' @param memory_location column indexing for the mtx
#' @param num_factors a desired number of random factors
#' @param rseed random seed
#' @param verbose verbosity
#' @param NUM_THREADS number of threads in data reading
#' @param BLOCK_SIZE disk I/O block size (number of columns)
#' @param do_normalize normalize each column after random projection
#' @param do_log1p log(x + 1) transformation (default: FALSE)
#' @param do_row_std rowwise standardization (default: FALSE)
#'
asap_random_bulk_data <- function(mtx_file, memory_location, num_factors, rseed = 42L, verbose = FALSE, NUM_THREADS = 1L, BLOCK_SIZE = 100L, do_normalize = FALSE, do_log1p = FALSE, do_row_std = FALSE) {
    .Call('_asapR_asap_random_bulk_data', PACKAGE = 'asapR', mtx_file, memory_location, num_factors, rseed, verbose, NUM_THREADS, BLOCK_SIZE, do_normalize, do_log1p, do_row_std)
}

#' Poisson regression to estimate factor loading
#'
#' @param Y D x N data matrix
#' @param log_x D x K log dictionary/design matrix
#' @param a0 gamma(a0, b0)
#' @param b0 gamma(a0, b0)
#' @param verbose verbosity
#' @param do_stdize do the standardization of log_x
#'
asap_regression <- function(Y_dn, log_x, a0 = 1., b0 = 1., max_iter = 10L, verbose = FALSE, do_stdize_x = FALSE, std_topic_latent = FALSE) {
    .Call('_asapR_asap_regression', PACKAGE = 'asapR', Y_dn, log_x, a0, b0, max_iter, verbose, do_stdize_x, std_topic_latent)
}

#' Poisson regression to estimate factor loading
#'
#' @param mtx_file matrix-market-formatted data file (bgzip)
#' @param memory_location column indexing for the mtx
#' @param log_x D x K log dictionary/design matrix
#' @param r_x_row_names (default: NULL)
#' @param r_mtx_row_names (default: NULL)
#' @param a0 gamma(a0, b0)
#' @param b0 gamma(a0, b0)
#' @param verbose verbosity
#' @param NUM_THREADS number of threads in data reading
#' @param BLOCK_SIZE disk I/O block size (number of columns)
#' @param do_stdize do the standardization of log_x
#'
asap_regression_mtx <- function(mtx_file, memory_location, log_x, r_x_row_names = NULL, r_mtx_row_names = NULL, a0 = 1., b0 = 1., max_iter = 10L, verbose = FALSE, NUM_THREADS = 1L, BLOCK_SIZE = 100L, do_stdize_x = FALSE, std_topic_latent = FALSE) {
    .Call('_asapR_asap_regression_mtx', PACKAGE = 'asapR', mtx_file, memory_location, log_x, r_x_row_names, r_mtx_row_names, a0, b0, max_iter, verbose, NUM_THREADS, BLOCK_SIZE, do_stdize_x, std_topic_latent)
}

#' Clustering the rows of a count data matrix
#'
#' @param X data matrix
#' @param Ltrunc DPM truncation level
#' @param alpha DPM parameter
#' @param a0 prior ~ Gamma(a0, b0) (default: 1e-2)
#' @param b0 prior ~ Gamma(a0, b0) (default: 1e-4)
#' @param rseed random seed (default: 42)
#' @param mcmc number of MCMC iterations (default: 100)
#' @param burnin number iterations to discard (default: 10)
#' @param verbose verbosity
#'
fit_poisson_cluster_rows <- function(X, Ltrunc, alpha = 1, a0 = 1e-2, b0 = 1e-4, rseed = 42L, mcmc = 100L, burnin = 10L, verbose = TRUE) {
    .Call('_asapR_fit_poisson_cluster_rows', PACKAGE = 'asapR', X, Ltrunc, alpha, a0, b0, rseed, mcmc, burnin, verbose)
}

#' Create an index file for a given MTX
#'
#' @param mtx_file data file
#' @param index_file index file
#'
#' @usage mmutil_build_index(mtx_file, index_file)
#'
#' @return EXIT_SUCCESS or EXIT_FAILURE
#'
mmutil_build_index <- function(mtx_file, index_file = "") {
    .Call('_asapR_mmutil_build_index', PACKAGE = 'asapR', mtx_file, index_file)
}

#' Read an index file to R
#'
#' @param index_file index file
#'
#' @return a vector column index (a vector of memory locations)
#'
mmutil_read_index <- function(index_file) {
    .Call('_asapR_mmutil_read_index', PACKAGE = 'asapR', index_file)
}

#' Check if the index tab is valid
#'
#' @param mtx_file data file
#' @param index_tab index tab (a vector of memory locations)
#'
#' @return EXIT_SUCCESS or EXIT_FAILURE
#'
mmutil_check_index <- function(mtx_file, index_tab) {
    .Call('_asapR_mmutil_check_index', PACKAGE = 'asapR', mtx_file, index_tab)
}

#' Just read the header information
#'
#' @param mtx_file data file
#'
#' @return info
#'
mmutil_info <- function(mtx_file) {
    .Call('_asapR_mmutil_info', PACKAGE = 'asapR', mtx_file)
}

#' Write down sparse matrix to the disk
#' @param X sparse matrix
#' @param mtx_file file name
#'
#' @return EXIT_SUCCESS or EXIT_FAILURE
mmutil_write_mtx <- function(X, mtx_file) {
    .Call('_asapR_mmutil_write_mtx', PACKAGE = 'asapR', X, mtx_file)
}

#' Read a subset of columns from the data matrix
#' @param mtx_file data file
#' @param memory_location column -> memory location
#' @param r_column_index column indexes to retrieve (1-based)
#'
#' @return lists of rows, columns, values
#'
mmutil_read_columns_sparse <- function(mtx_file, memory_location, r_column_index, verbose = FALSE) {
    .Call('_asapR_mmutil_read_columns_sparse', PACKAGE = 'asapR', mtx_file, memory_location, r_column_index, verbose)
}

#' Read a subset of columns from the data matrix
#' @param mtx_file data file
#' @param memory_location column -> memory location
#' @param r_column_index column indexes to retrieve (1-based)
#'
#' @return a dense sub-matrix
#'
#' @examples
#'
#' rr <- rgamma(100, 1, 1) # one hundred cells
#' mm <- matrix(rgamma(10 * 3, 1, 1), 10, 3)
#' data.hdr <- "test_sim"
#' .files <- asapR::mmutil_simulate_poisson(mm, rr, data.hdr)
#' data.file <- .files$mtx
#' idx.file <- .files$idx
#' mtx.idx <- asapR::mmutil_read_index(idx.file)
#' Y <- as.matrix(Matrix::readMM(data.file))
#' col.pos <- c(1,13,77) # 1-based
#' yy <- asapR::mmutil_read_columns(
#'                  data.file, mtx.idx, col.pos)
#' all(Y[, col.pos, drop = FALSE] == yy)
#' print(head(Y[, col.pos, drop = FALSE]))
#' print(head(yy))
#' unlink(list.files(pattern = data.hdr))
#'
mmutil_read_columns <- function(mtx_file, memory_location, r_column_index, verbose = FALSE) {
    .Call('_asapR_mmutil_read_columns', PACKAGE = 'asapR', mtx_file, memory_location, r_column_index, verbose)
}

#' Read a subset of rows and columns from the data matrix
#' @param mtx_file data file
#' @param memory_location column -> memory location
#' @param r_row_index row indexes to retrieve (1-based)
#' @param r_column_index column indexes to retrieve (1-based)
#' @param verbose verbosity
#'
#' @return a dense sub-matrix
#'
#' @examples
#'
#' rr <- rgamma(100, 1, 1) # one hundred cells
#' mm <- matrix(rgamma(10 * 3, 1, 1), 10, 3)
#' data.hdr <- "test_sim"
#' .files <- asapR::mmutil_simulate_poisson(mm, rr, data.hdr)
#' data.file <- .files$mtx
#' idx.file <- .files$idx
#' mtx.idx <- asapR::mmutil_read_index(idx.file)
#' Y <- as.matrix(Matrix::readMM(data.file))
#' col.pos <- c(1,13,77) # 1-based
#' row.pos <- 1:10
#' yy <- asapR::mmutil_read_rows_columns(
#'                  data.file, mtx.idx, row.pos, col.pos)
#' all(Y[, col.pos, drop = FALSE] == yy)
#' print(head(Y[, col.pos, drop = FALSE]))
#' print(head(yy))
#' print(tail(yy))
#' unlink(list.files(pattern = data.hdr))
#'
mmutil_read_rows_columns <- function(mtx_file, memory_location, r_row_index, r_column_index, verbose = FALSE) {
    .Call('_asapR_mmutil_read_rows_columns', PACKAGE = 'asapR', mtx_file, memory_location, r_row_index, r_column_index, verbose)
}

#' Simulate sparse counting data with a mixture of Poisson parameters
#'
#'
#' @param r_mu_list a list of gene x individual matrices
#' @param Ncell the total number of cells (may not make it if too sparse)
#' @param output a file header string for output files
#' @param dir_alpha a parameter for Dirichlet(alpha * [1, ..., 1])
#' @param gam_alpha a parameter for Gamma(alpha, beta)
#' @param gam_beta a parameter for Gamma(alpha, beta)
#' @param rseed random seed
#'
mmutil_simulate_poisson_mixture <- function(r_mu_list, Ncell, output, dir_alpha = 1.0, gam_alpha = 2.0, gam_beta = 2.0, rseed = 42L) {
    .Call('_asapR_mmutil_simulate_poisson_mixture', PACKAGE = 'asapR', r_mu_list, Ncell, output, dir_alpha, gam_alpha, gam_beta, rseed)
}

#' Simulation Poisson data based on Mu
#'
#' M= num. of features and n= num. of indv
#'
#' @param mu depth-adjusted mean matrix (M x n)
#' @param rho column depth vector (N x 1), N= num. of cells
#' @param output header for ${output}.{mtx.gz,cols.gz,indv.gz}
#' @param r_indv N x 1 individual membership (1-based, [1 .. n])
#' @param rseed random seed
#'
#' @return a list of file names: {output}.{mtx,rows,cols}.gz
#'
mmutil_simulate_poisson <- function(mu, rho, output, r_indv = NULL, rseed = 42L) {
    .Call('_asapR_mmutil_simulate_poisson', PACKAGE = 'asapR', mu, rho, output, r_indv, rseed)
}
