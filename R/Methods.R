#' Fit a topic model by ASAP
#'
#' @param mtx.file data matrix file in a matrix market format
#' @param k number of topics
#' @param num.proj the number of pseudobulk projection steps
#' @param em.step Monte Carlo EM steps (default: 100)
#' @param max.pb.size maximum pseudobulk size (default: 1000)
#' @param .burnin burn-in period in the record keeping (default: 10)
#' @param .thining thining for the record keeping (default: 3)
#' @param a0 Gamma(a0, b0) prior (default: 1)
#' @param b0 Gamma(a0, b0) prior (default: 1)
#' @param index.file a file for column indexes (default: "{mtx.file}.index")
#' @param verbose verbosity
#' @param num.threads number of threads (default: 1)
#' @param block.size a block size for disk I/O (default: 100)
#' @param eval.llik evaluate log-likelihood trace in PMF (default: TRUE)
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
                           .burnin = 10,
                           .thining = 3,
                           a0 = 1,
                           b0 = 1,
                           index.file = paste0(mtx.file, ".index"),
                           verbose = TRUE,
                           num.threads = 1,
                           block.size = 100,
                           .rand.seed = 42){

    if(!file.exists(index.file)){
        mmutil_build_index(mtx.file, index.file)
    }

    mtx.index <- mmutil_read_index(index.file)
    if(length(mtx.index) < 1){
        message("Failed to read the indexing file: ", index.file)
        return(NULL)
    }

    message("Phase I: Create random pseudo-bulk data")

    Y <- NULL
    .pb.out <- NULL

    for(r in 1:num.proj){
        .pb <- asap_random_bulk_data(mtx_file = mtx.file,
                                     memory_location = mtx.index,
                                     num_factors = pb.factors,
                                     rseed = .rand.seed + r,
                                     verbose = verbose,
                                     NUM_THREADS = num.threads,
                                     BLOCK_SIZE = block.size)
        Y <- cbind(Y, .pb$PB)

        message("Found Random Pseudobulk: Y ", nrow(Y), " x ", ncol(Y))
    }

    if(ncol(Y) > max.pb.size){
        Y <- Y[, sample(ncol(Y), max.pb.size), drop = FALSE]
    }

    message("Phase II: Perform Poisson matrix factorization ...")

    .nmf <- asap_fit_nmf_alternate(Y,
                                   maxK = k,
                                   max_iter = em.step,
                                   burnin = .burnin,
                                   verbose = verbose,
                                   a0=a0,
                                   b0=b0,
                                   rseed = .rand.seed,
                                   EPS=1e-4)

    .multinom <- pmf2topic(.nmf$beta, .nmf$theta)
    .nmf$beta.rescaled <- .multinom$beta
    .nmf$theta.rescaled <- .multinom$prop
    .nmf$depth <- .multinom$depth

    message("Phase III: Calibrating the loading parameters of the original data")

    asap <- asap_regression_mtx(mtx_file = mtx.file,
                                memory_location = mtx.index,
                                log_x = .nmf$log.beta,
                                a0 = a0, b0 = b0,
                                max_iter = em.step,
                                verbose = verbose,
                                NUM_THREADS = num.threads,
                                BLOCK_SIZE = block.size)

    message("normalizing the estimated model parameters")

    .multinom <- pmf2topic(asap$beta, asap$theta)

    asap$beta.rescaled <- .multinom$beta
    asap$theta.rescaled <- .multinom$prop
    asap$depth <- .multinom$depth
    asap$nmf <- .nmf
    asap$Y <- Y
    return(asap)
}


#' Fit a topic model by full column-wise iterations
#'
#' @param mtx.file data matrix file in a matrix market format
#' @param k number of topics
#' @param em.step Monte Carlo EM steps (default: 100)
#' @param e.step Num of E-step MCMC steps (default: 1)
#' @param .burnin burn-in period in the record keeping (default: 10)
#' @param .thining thining for the record keeping (default: 3)
#' @param a0 Gamma(a0, b0) prior (default: 1)
#' @param b0 Gamma(a0, b0) prior (default: 1)
#' @param index.file a file for column indexes (default: "{mtx.file}.index")
#' @param verbose verbosity
#' @param num.threads number of threads (default: 1)
#' @param eval.llik evaluate log-likelihood trace in PMF (default: FALSE)
#' @param .rand.seed random seed (default: 42)
#' @param .sample.col.row Take column-wise MH step (default: TRUE)
#'
#'
fit.topic.full <- function(mtx.file,
                           k,
                           em.step = 100,
                           e.step = 1,
                           .burnin = 10,
                           .thining = 3,
                           a0 = 1,
                           b0 = 1,
                           index.file = paste0(mtx.file, ".index"),
                           verbose = TRUE,
                           num.threads = 1,
                           eval.llik = FALSE,
                           .rand.seed = 42,
                           .sample.col.row = TRUE){

    message("Reading in the full data ...")

    Y <- read.mtx.dense(mtx.file = mtx.file,
                        memory.idx.file = index.file,
                        verbose = verbose,
                        num.threads = num.threads)

    message("Start estimating a PMF model in the full data ...")

    ret <- asap_fit_nmf(Y,
                        maxK = k,
                        mcem = em.step,
                        burnin = .burnin,
                        latent_iter = e.step,
                        thining = .thining,
                        verbose = verbose,
                        eval_llik = eval.llik,
                        a0=a0,
                        b0=b0,
                        rseed = .rand.seed,
                        NUM_THREADS = num.threads)

    message("normalizing the estimated model parameters")

    .multinom <- pmf2topic(ret$dict$mean, ret$column$mean)

    ret$beta <- .multinom$beta
    ret$prop <- .multinom$prop
    ret$depth <- .multinom$depth

    return(ret)
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
pmf2topic <- function(.beta, .theta) {

    uu <- apply(.beta, 2, sum)
    beta <- sweep(.beta, 2, uu, `/`)
    prop <- sweep(.theta, 2, uu, `*`)
    zz <- apply(t(prop), 2, sum)
    prop <- sweep(prop, 1, zz, `/`)

    list(beta = beta, prop = prop, depth = zz)
}
