#include "mmutil.hh"
#include "rcpp_util.hh"

//' Collapse N x N adjacency network into S x S
//'
//' @param W_nn_list adjacency list
//' @param r_positions collapsing positions
//' @param N number of vertices
//' @param S number of meta-vertices
//' @param verbose verbosity
//'
//' @return a list of {rows, columns, weights}
//'
// [[Rcpp::export]]
Rcpp::List
collapse_network(const Rcpp::List W_nn_list,
                 const Rcpp::IntegerVector r_positions,
                 const std::size_t N,
                 const std::size_t S,
                 const bool verbose = true)
{
    SpMat W_nn;
    rcpp::util::build_sparse_mat(W_nn_list, N, N, W_nn);
    TLOG_(verbose, "kNN graph W: " << W_nn.rows() << " x " << W_nn.cols());

    // R is 1-based; make it 0-based
    std::vector<Index> positions;
    rcpp::util::convert_c_index(r_positions, positions);

    // 1. build Z (N x S)
    const Index NA_POS = S;
    Mat Z_ns = Mat::Zero(N, S);
    for (Index j = 0; j < positions.size(); ++j) {
        const Index s = positions.at(j);
        if (s >= 0 && s < NA_POS) {
            Z_ns(j, s) = 1.;
        }
    }

    const Scalar EPS = 1e-4, ONE = 1.0;
    // 2. Z' * W * Z
    SpMat W_ss(S, S);
    W_ss = (Z_ns.transpose() * W_nn * Z_ns).sparseView(ONE, EPS);

    return rcpp::util::build_sparse_list(W_ss);
}
