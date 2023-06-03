#' Fit a topic model by ASAP
#'
#' @param mtx.file data matrix file in a matrix market format
#' @param k number of topics
#' @param num.proj the number of pseudobulk projection steps (default: 1)
#' @param em.step Monte Carlo EM steps (default: 100)
#' @param max.pb.size maximum pseudobulk size (default: 1000)
#' @param covar sample x var covariate matrix (default: NULL)
#' @param batch batch information for each sample (default: NULL)
#' @param .burnin burn-in period in the record keeping (default: 0)
#' @param .reg.steps number of steps in regression analysis (default: 10)
#' @param .reg.stdize standardize X in regression (default: TRUE)
#' @param .stdize.cols standardize columns after log (default: TRUE)
#' @param a0 Gamma(a0, b0) prior (default: 1)
#' @param b0 Gamma(a0, b0) prior (default: 1)
#' @param index.file a file for column indexes (default: "{mtx.file}.index")
#' @param verbose verbosity
#' @param num.threads number of threads (default: 1)
#' @param block.size a block size for disk I/O (default: 100)
#' @param eval.llik evaluate log-likelihood trace in PMF (default: TRUE)
#' @param svd.init (default: FALSE)
#' @param do.log1p (default: FALSE)
#' @param .rand.seed random seed (default: 42)
#'
#' @return a list that contains:
#'  \itemize{
#'   \item `beta.rescaled` rescaled dictionary (feature x factor)
#'   \item `theta.rescaled` rescaled loading (sample x factor)
#'   \item `beta` dictionary (feature x factor)
#'   \item `log.beta` log-dictionary (feature x factor)
#'   \item `theta` loading (sample x factor)
#'   \item `log.theta` log-loading (sample x factor)
#'   \item `Y` pseudobulk (PB) data (feature x pseudobulk samples)
#'   \item `nmf` non-negative matrix factorization results
#'   \itemize{
#'     \item `log.likelihood` log-likelihood trace
#'     \item `beta` dictionary of PB data
#'     \item `log.beta` log dictionary of PB data
#'     \item `theta` factor loadings of PB data
#'     \item `log.theta` log factor loadings of PB data
#'     \item `log.phi` auxiliary variable (feature x factor)
#'     \item `log.rho` auxiliary variable (sample x factor)
#'     \item `beta.rescaled` rescaled dictionary
#'     \item `theta.rescaled` rescaled factor loading
#'     \item `depth`
#'   }
#'   \item `corr` sample x factor correlation matrix
#' }
#'
fit.topic.asap <- function(mtx.file,
                           k,
                           pb.factors = 10,
                           num.proj = 1,
                           em.step = 100,
                           max.pb.size = 1000,
                           .burnin = 0,
                           .reg.steps = 10,
                           .reg.stdize = TRUE,
                           .stdize.cols = FALSE,
                           a0 = 1,
                           b0 = 1,
                           index.file = paste0(mtx.file, ".index"),
                           verbose = TRUE,
                           num.threads = 1,
                           block.size = 100,
                           .rand.seed = 42,
                           .eps = 1e-6,
                           svd.init = FALSE,
                           do.log1p = FALSE,
                           covar = NULL,
                           batch = NULL){

    if(!file.exists(index.file)){
        mmutil_build_index(mtx.file, index.file)
    }

    message("Phase I: Create random pseudo-bulk data")

    Y <- NULL
    rand.proj <- NULL
    rand.positions <- NULL
    batch.effect <- NULL

    for(r in 1:num.proj){
        .pb <- asap_random_bulk_data(mtx_file = mtx.file,
                                     mtx_idx_file = index.file,
                                     num_factors = pb.factors,
                                     r_covar = covar,
                                     r_batch = batch,
                                     rseed = .rand.seed + (r - 1),
                                     verbose = verbose,
                                     NUM_THREADS = num.threads,
                                     BLOCK_SIZE = block.size,
                                     do_log1p = do.log1p,
                                     do_row_std = FALSE)

        stopifnot(!is.null(.pb$PB))

        Y <- cbind(Y, .pb$PB)

        rand.proj <- cbind(rand.proj, .pb$rand.proj)
        rand.positions <- c(rand.positions, .pb$positions)
        if(nrow(.pb$log.batch.effect) == nrow(Y)){
            batch.effect <- cbind(batch.effect, .pb$log.batch.effect)
        }
        message("Constructed random pseudo-bulk samples: Y ", nrow(Y), " x ", ncol(Y))
    }

    if(ncol(Y) > max.pb.size){
        .cols <- sample(ncol(Y), max.pb.size)
        Y <- Y[, .cols, drop = FALSE]
    }

    if(.stdize.cols){
        message("Stretching out each columns by log-standardization")
        Y <- stretch_matrix_columns(Y)
    }

    message("Phase II: Perform Poisson matrix factorization on the Y")

    .nmf <- asap_fit_nmf_alternate(Y,
                                   maxK = k,
                                   max_iter = em.step,
                                   burnin = .burnin,
                                   verbose = verbose,
                                   a0 = a0,
                                   b0 = b0,
                                   do_log1p = do.log1p,
                                   rseed = .rand.seed,
                                   svd_init = svd.init,
                                   EPS = .eps,
                                   NUM_THREADS = num.threads)

    .multinom <- pmf2topic(.nmf$beta, .nmf$theta)
    .nmf$beta.rescaled <- .multinom$beta
    .nmf$theta.rescaled <- .multinom$prop
    .nmf$depth <- .multinom$depth

    message("Phase III: Topic proportions of the columns in the original data")

    log.x <- .nmf$log.beta

    asap <- asap_regression_mtx(mtx_file = mtx.file,
                                mtx_idx_file = index.file,
                                log_x = log.x,
                                r_batch_effect = batch.effect,
                                r_x_row_names = NULL,
                                r_mtx_row_names = NULL,
                                r_taboo_names = NULL,
                                a0 = a0, b0 = b0,
                                max_iter = .reg.steps,
                                do_log1p = do.log1p,
                                verbose = verbose,
                                NUM_THREADS = num.threads,
                                BLOCK_SIZE = block.size,
                                do_stdize_x = .reg.stdize)

    message("Normalizing the estimated model parameters")

    .multinom <- pmf2topic(asap$beta, asap$theta)

    asap$beta.rescaled <- .multinom$beta
    asap$theta.rescaled <- .multinom$prop
    asap$depth <- .multinom$depth
    asap$nmf <- .nmf
    asap$Y <- Y

    asap$rand.proj <- rand.proj
    asap$rand.positions <- rand.positions
    asap$batch.effect <- batch.effect

    return(asap)
}

#' Convert Poisson Matrix Factorization to Multinomial Topic model.
#' The same idea was first coined by `fastTopics` paper.
#'
#' @details
#' E[Y|.beta, .theta] = `.beta %*% t(.theta)`
#'
#' @param .beta D x K dictionary matrix
#' @param .theta N x K sample-wise factor loading matrix
#'
#' @return a list of (beta, prop, depth), where the beta is a re-scaled dictionary matrix, each row of the prop matrix corresponds to a mixing proportion per column, and the depth parameter gauges a sequencing depth for each column
#'
pmf2topic <- function(.beta, .theta, eps=1e-8) {

    uu <- pmax(apply(.beta, 2, sum), eps)
    beta <- sweep(.beta, 2, uu, `/`)
    prop <- sweep(.theta, 2, uu, `*`)
    zz <- pmax(apply(t(prop), 2, sum), eps)
    prop <- sweep(prop, 1, zz, `/`)

    list(beta = beta, prop = prop, depth = zz)
}
