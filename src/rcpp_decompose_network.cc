#include "mmutil.hh"

//' Assign best topic membership for the edges
//'
//' @param A_dd D x D adjacency matrix
//' @param beta_dt D x T node propensity matrix
//' @param cutoff A[i,j] cutoff (default: 1e-8)
//' @param verbose verbosity
//'
// [[Rcpp::export]]
Rcpp::List
decompose_network(const Eigen::SparseMatrix<double, Eigen::ColMajor> A_dd,
                  const Eigen::MatrixXd beta_dt,
                  const double cutoff = 1e-8,
                  const bool verbose = true)
{
    const std::size_t D = A_dd.rows();
    ASSERT_RETL(beta_dt.rows() == D && beta_dt.rows() == A_dd.cols(),
                "rows(A) and rows(beta) don't match");

    const std::size_t m = A_dd.nonZeros();
    std::vector<std::size_t> rows;
    rows.reserve(m);
    std::vector<std::size_t> cols;
    cols.reserve(m);
    std::vector<std::size_t> ks;
    ks.reserve(m);
    std::vector<double> weights;
    weights.reserve(m);

    using sp_mat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
    for (std::size_t j = 0; j < A_dd.outerSize(); ++j) {
        for (sp_mat::InnerIterator it(A_dd, j); it; ++it) {
            const std::size_t i = it.index();
            if (it.value() > cutoff) {
                std::size_t k;
                beta_dt.row(j).cwiseProduct(beta_dt.row(i)).maxCoeff(&k);
                rows.emplace_back(i + 1); // 1-based
                cols.emplace_back(j + 1); // 1-based
                ks.emplace_back(k + 1);   // 1-based
                weights.emplace_back(it.value());
            }
        }
    }

    return Rcpp::List::create(Rcpp::_["row"] = rows,
                              Rcpp::_["col"] = cols,
                              Rcpp::_["weight"] = weights,
                              Rcpp::_["topic"] = ks);
}
