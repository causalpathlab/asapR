#include "rcpp_asap_util.hh"

//' Stretch non-negative matrix
//'
//' @param Y non-negative data matrix(gene x sample)
//' @param qq_min min quantile (default: 0.01)
//' @param qq_max min quantile (default: 0.99)
//' @param std_min min after standardization of log (default: -8)
//' @param std_max max after standardization of log (default: 8)
//' @param verbose speak more (default: TRUE)
//'
// [[Rcpp::export]]
Rcpp::NumericMatrix
stretch_matrix_columns(const Eigen::MatrixXf Y,
                       const double qq_min = 0.01,
                       const double qq_max = 0.99,
                       const double std_min = -8,
                       const double std_max = 8,
                       const bool verbose = true)
{
    std::vector<Scalar> quantile(Y.size());
    Eigen::Map<Eigen::MatrixXf>(quantile.data(), Y.rows(), Y.cols()) = Y;

    TLOG_(verbose, "n = " << quantile.size() << " elements");

    const double nn = quantile.size();
    const std::size_t q1 = std::ceil(qq_min * nn);
    const std::size_t q2 = std::floor(qq_max * nn);

    Scalar lb, ub;

    if (q1 < q2 && q2 < quantile.size()) {

        std::nth_element(quantile.begin(),
                         quantile.begin() + q1,
                         quantile.end());

        lb = quantile[q1];

        std::nth_element(quantile.begin() + q1 + 1,
                         quantile.begin() + q2,
                         quantile.end());

        ub = quantile[q2];
    } else {
        lb = *std::min_element(quantile.begin(), quantile.end());
        ub = *std::max_element(quantile.begin(), quantile.end());
    }

    if (lb < 0) {
        WLOG("Found negative values");
        return Rcpp::NumericMatrix(Rcpp::wrap(Y));
    }

    TLOG_(verbose,
          "Shrink the raw values "
              << "between " << lb << " and " << ub << ".");

    Mat ret;
    if (lb < ub) {
        clamp_op<Mat> clamp_y(lb, ub);
        ret = Y.unaryExpr(clamp_y).eval();
    } else {
        ret = Y;
    }

    const Scalar pseudo_count = 1.0 / static_cast<Scalar>(Y.rows());

    if (lb < pseudo_count) {
        ret.array() += pseudo_count;
    }

    safe_log_op<Mat> log(1e-8);
    exp_op<Mat> exp;
    ret = ret.unaryExpr(log).eval();
    stdizer_t<Mat> std(ret);
    std.colwise();

    TLOG_(verbose,
          "After standardization we have: "
              << "[" << ret.minCoeff() << ", " << ret.maxCoeff() << "]");

    clamp_op<Mat> clamp_std(std_min, std_max);
    ret = ret.unaryExpr(clamp_std).unaryExpr(exp).eval();

    return Rcpp::NumericMatrix(Rcpp::wrap(ret));
}
