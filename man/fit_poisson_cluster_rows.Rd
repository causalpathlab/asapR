% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{fit_poisson_cluster_rows}
\alias{fit_poisson_cluster_rows}
\title{Clustering the rows of a count data matrix}
\usage{
fit_poisson_cluster_rows(
  X,
  Ltrunc,
  alpha = 1,
  a0 = 0.01,
  b0 = 1e-04,
  rseed = 42L,
  mcmc = 100L,
  burnin = 10L,
  verbose = TRUE
)
}
\arguments{
\item{X}{data matrix}

\item{Ltrunc}{DPM truncation level}

\item{alpha}{DPM parameter}

\item{a0}{prior ~ Gamma(a0, b0) (default: 1e-2)}

\item{b0}{prior ~ Gamma(a0, b0) (default: 1e-4)}

\item{rseed}{random seed (default: 42)}

\item{mcmc}{number of MCMC iterations (default: 100)}

\item{burnin}{number iterations to discard (default: 10)}

\item{verbose}{verbosity}
}
\description{
Clustering the rows of a count data matrix
}
