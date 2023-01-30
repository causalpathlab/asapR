#' Fit a topic model by ASAP
#'
#' @param mtx.file data matrix file in a matrix market format
#' @param k number of topics
#' @param num.proj the number of pseudobulk projection steps
#' @param em.step Monte Carlo EM steps (default: 100)
#' @param e.step Num of E-step MCMC steps (default: 1)
#' @param deg.step Num of Degree calibration steps (default: 0)
#' @param .burnin burn-in period in the record keeping (default: 10)
#' @param .thining thining for the record keeping (default: 3)
#' @param a0 Gamma(a0, b0) prior (default: 1)
#' @param b0 Gamma(a0, b0) prior (default: 1)
#' @param index.file a file for column indexes (default: "{mtx.file}.index")
#' @param verbose verbosity
#' @param num.threads number of threads (default: 1)
#' @param block.size a block size for disk I/O (default: 100)
#' @param eval.llik evaluate log-likelihood trace in PMF (default: FALSE)
#' @param .rand.seed random seed (default: 42)
#' @param .sample.col.row take column-wise MH step (default: TRUE)
#'
#'
fit.asap <- function(mtx.file,
                     k,
                     pb.factors = k,
                     num.proj = 3,
                     em.step = 100,
                     e.step = 1,
                     deg.step = 0,
                     .burnin = 10,
                     .thining = 3,
                     a0 = 1,
                     b0 = 1,
                     index.file = paste0(mtx.file, ".index"),
                     verbose = TRUE,
                     num.threads = 1,
                     block.size = 100,
                     eval.llik = FALSE,
                     .rand.seed = 42,
                     .sample.col.row = TRUE){

    if(!file.exists(index.file)){
        mmutil_build_index(mtx.file, index.file)
    }

    .index <- mmutil_read_index(index.file)
    if(length(.index) < 1){
        message("Failed to read the indexing file: ", index.file)
        return(NULL)
    }

    message("Phase I: Create random pseudo-bulk data")

    Y <- NULL

    for(r in 1:num.proj){
        .pb <- asap_random_bulk_data(mtx.file,
                                     .index,
                                     num_factors = pb.factors,
                                     rseed = .rand.seed + r,
                                     verbose = verbose,
                                     NUM_THREADS = num.threads,
                                     BLOCK_SIZE = block.size)
        Y <- cbind(Y, .pb$PB)
    }

    message("Phase II: Perform Poisson matrix factorization")

    .nmf <- asap_fit_nmf(Y,
                         maxK = k,
                         mcem = em.step,
                         burnin = .burnin,
                         do_sample_col_row = .sample.col.row,
                         latent_iter = e.step,
                         degree_iter = deg.step,
                         thining = .thining,
                         verbose = verbose,
                         eval_llik = eval.llik,
                         a0=a0,
                         b0=b0,
                         rseed = .rand.seed,
                         NUM_THREADS = num.threads)

    message("Phase III: The loading parameters for the original column vectors")

    beta <- .nmf$row$mean
    uu <- apply(beta, 2, sum)
    beta <- sweep(beta, 2, uu, `/`)

    asap <- asap_predict_mtx(mtx.file,
                             .index,
                             beta,
                             mcem = em.step,
                             burnin = .burnin,
                             thining = .thining,
                             a0 = a0,
                             b0 = b0,
                             rseed = .rand.seed,
                             verbose = verbose,
                             NUM_THREADS = num.threads,
                             BLOCK_SIZE = block.size)

    message("normalizing the estimated model parameters")

    ## The same idea borrowed from `fastTopics`
    uu <- apply(beta, 2, sum)
    beta <- sweep(beta, 2, uu, `/`)
    prop <- sweep(asap$theta, 2, uu, `*`)
    zz <- apply(t(prop), 2, sum)
    prop <- sweep(prop, 1, zz, `/`)

    asap$beta <- beta
    asap$prop <- prop
    asap$depth <- zz
    asap$nmf <- .nmf

    return(asap)
}
