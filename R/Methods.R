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
#' @param do.collapse.rows collapse rows to speed up the final NMF
#' @param .beta.rescale rescale beta in the final prediction step
#' @param .collapsing.discrete do the row collapsing after discretization
#' @param .collapsing.level collapsed dimension (default: 300)
#' @param .collapsing.dpm (default: 1)
#' @param .collapsing.mcmc (default: 100)
#' @param a0 Gamma(a0, b0) prior (default: 1e-2)
#' @param b0 Gamma(a0, b0) prior (default: 1e-4)
#' @param index.file a file for column indexes (default: "{mtx.file}.index")
#' @param verbose verbosity
#' @param num.threads number of threads (default: 1)
#' @param block.size a block size for disk I/O (default: 100)
#' @param eval.llik evaluate log-likelihood trace in PMF (default: TRUE)
#' @param .rand.seed random seed (default: 42)
#'
fit.topic.asap <- function(mtx.file,
                           k,
                           pb.factors = k,
                           num.proj = 3,
                           em.step = 100,
                           e.step = 1,
                           deg.step = 0,
                           .burnin = 10,
                           .thining = 3,
                           do.collapse.rows = TRUE,
                           .beta.rescale = TRUE,
                           .collapse.discrete = FALSE,
                           .collapsing.level = 300,
                           .collapsing.dpm = 1.,
                           .collapsing.mcmc = 100,
                           a0 = 1e-2,
                           b0 = 1e-4,
                           index.file = paste0(mtx.file, ".index"),
                           verbose = TRUE,
                           num.threads = 1,
                           block.size = 100,
                           eval.llik = TRUE,
                           .rand.seed = 42){

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

    message("Phase II: Perform Poisson matrix factorization ...")

    .nmf <- asap_fit_nmf(Y,
                         maxK = k,
                         mcem = em.step,
                         burnin = .burnin,
                         latent_iter = e.step,
                         degree_iter = deg.step,
                         thining = .thining,
                         verbose = verbose,
                         eval_llik = eval.llik,
                         a0=a0,
                         b0=b0,
                         rseed = .rand.seed,
                         NUM_THREADS = num.threads)

    .multinom <- pmf2topic(.nmf$row$mean, .nmf$column$mean)
    .nmf$beta <- .multinom$beta
    .nmf$prop <- .multinom$prop
    .nmf$depth <- .multinom$depth


    if(do.collapse.rows){

        message("Clustering to reduce dimensions...")

        B <- .nmf$beta * nrow(.nmf$beta)
        clustering <- fit_poisson_cluster_rows(B,
                                               Ltrunc = .collapsing.level,
                                               alpha = .collapsing.dpm,
                                               a0 = a0,
                                               b0 = b0,
                                               rseed = .rand.seed,
                                               mcmc = .collapsing.mcmc,
                                               burnin = .burnin,
                                               verbose = verbose)

        collapsing <- t(clustering$latent$mean)

        if(.collapse.discrete){
            collapsing <- apply(collapsing, 2,
                                function(z){
                                    k <- which.max(z)
                                    ret <- z * 0
                                    ret[k] <- 1
                                    return(ret)
                                })
        }

        .size <- pmax(apply(collapsing, 1, sum), 1)
        collapsing <- sweep(collapsing, 1, .size, `/`)

    } else {
        clustering <- NULL
        collapsing <- NULL
    }

    message("Phase III: Calibrating the loading parameters of the original data")

    asap <- asap_predict_mtx(mtx.file,
                             .index,
                             beta_dict = .nmf$row$mean,
                             do_beta_rescale = .beta.rescale,
                             collapsing = collapsing,
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

    .multinom <- pmf2topic(asap$beta, asap$theta)

    asap$beta <- .multinom$beta
    asap$prop <- .multinom$prop
    asap$depth <- .multinom$depth
    asap$nmf <- .nmf
    asap$clustering <- clustering
    return(asap)
}


#' Fit a topic model by full column-wise iterations
#'
#' @param mtx.file data matrix file in a matrix market format
#' @param k number of topics
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
#' @param eval.llik evaluate log-likelihood trace in PMF (default: FALSE)
#' @param .rand.seed random seed (default: 42)
#' @param .sample.col.row Take column-wise MH step (default: TRUE)
#'
#'
fit.topic.full <- function(mtx.file,
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
                        degree_iter = deg.step,
                        thining = .thining,
                        verbose = verbose,
                        eval_llik = eval.llik,
                        a0=a0,
                        b0=b0,
                        rseed = .rand.seed,
                        NUM_THREADS = num.threads)

    message("normalizing the estimated model parameters")

    .multinom <- pmf2topic(ret$row$mean, ret$column$mean)

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
