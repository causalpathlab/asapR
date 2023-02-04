#include "rcpp_cluster.hh"

//' Clustering the rows of a count data matrix
//'
//' @param X data matrix
//' @param Ltrunc DPM truncation level
//' @param alpha DPM parameter
//' @param a0 prior ~ Gamma(a0, b0) (default: 1e-2)
//' @param b0 prior ~ Gamma(a0, b0) (default: 1e-4)
//' @param rseed random seed (default: 42)
//' @param mcmc number of MCMC iterations (default: 100)
//' @param burnin number iterations to discard (default: 10)
//' @param verbose verbosity
//'
// [[Rcpp::export]]
Rcpp::List
fit_poisson_cluster_rows(const Eigen::MatrixXf &X,
                         const std::size_t Ltrunc,
                         const double alpha = 1,
                         const double a0 = 1e-2,
                         const double b0 = 1e-4,
                         const std::size_t rseed = 42,
                         const std::size_t mcmc = 100,
                         const std::size_t burnin = 10,
                         const bool verbose = true)
{
    using F = poisson_component_t<Mat>;
    using F0 = trunc_dpm_t<Mat>;

    clustering_status_t<Mat> status(X.rows(), X.cols(), Ltrunc);

    clustering_by_lcvi<F, F0>(status,
                              X,
                              Ltrunc,
                              alpha,
                              a0,
                              b0,
                              rseed,
                              mcmc,
                              burnin,
                              verbose);

    auto _summary = [](running_stat_t<Mat> &stat) {
        return Rcpp::List::create(Rcpp::_["mean"] = stat.mean(),
                                  Rcpp::_["sd"] = stat.sd());
    };

    return Rcpp::List::create(Rcpp::_["elbo"] = status.elbo,
                              Rcpp::_["latent"] = _summary(status.latent),
                              Rcpp::_["parameter"] = _summary(status.parameter));
}
