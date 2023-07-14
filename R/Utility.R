#' Create 2D projection of topic proportions
#' @param .prop cell/sample x topic/factor proportion matrix
#' @param max.angle maximum angle
#' @return xy
project.proportions <- function(.prop, max.angle=2*pi){
    K <- ncol(.prop)

    max.angle <- min(max.angle, 2*pi)

    .angles <- seq(max.angle/K, max.angle, max.angle/K)
    .x <- sin(.angles)
    .y <- cos(.angles)

    xx <- apply(sweep(t(.prop), 1, .x, `*`), 2, sum)
    yy <- apply(sweep(t(.prop), 1, .y, `*`), 2, sum)

    cbind(xx,yy)
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
