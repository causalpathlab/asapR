#include "rcpp_asap_pmf_seq.hh"

//' 
//'
//' @param y_dn_vec a list of non-negative data matrices (gene x sample)
//' @param maxK maximum number of factors
//' @param max_iter max number of optimization steps
//' @param min_iter min number of optimization steps
//' @param burnin number of initiation steps (default: 50)
//' @param verbose verbosity
//' @param a0 gamma(a0, b0) default: a0 = 1
//' @param b0 gamma(a0, b0) default: b0 = 1
//' @param do_scale scale each column by standard deviation (default: TRUE)
//' @param do_log1p do log(1+y) transformation
//' @param rseed random seed (default: 1337)
//' @param EPS (default: 1e-8)
//' 
// [[Rcpp::export]]
Rcpp::List
asap_fit_pmf_seq_shared(const std::vector<Eigen::MatrixXf> y_dn_vec,
                        const std::size_t maxK,
                        const std::size_t max_iter = 100,
                        const std::size_t burnin = 0,
                        const bool verbose = true,
                        const double a0 = 1,
                        const double b0 = 1,
                        const bool do_log1p = false,
                        const std::size_t rseed = 1337,
                        const double EPS = 1e-8,
                        const std::size_t NUM_THREADS = 0)
{

    // Y(0) ~ beta(0) * theta
    // Y(t) ~ beta(t) * prod_s=0^(t-1) beta(s) * theta

    return Rcpp::List::create();
}
