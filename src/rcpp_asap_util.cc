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
asap_stretch_nn_matrix_columns(const Eigen::MatrixXf Y,
                               const double qq_min = 0.01,
                               const double qq_max = 0.99,
                               const double std_min = -8,
                               const double std_max = 8,
                               const bool verbose = true)
{

    if (Y.rows() < 1 || Y.cols() < 1) {
        TLOG_(verbose, "Empty matrix: " << Y.rows() << " x " << Y.cols());
        return Rcpp::NumericMatrix(Rcpp::wrap(Y));
    }

    if (Y.minCoeff() < 0) {
        TLOG_(verbose, "Found negative values");
        return Rcpp::NumericMatrix(Rcpp::wrap(Y));
    }

    const Scalar pseudo_count = 1.0 / static_cast<Scalar>(Y.rows());

    safe_log_op<Mat> log(pseudo_count);

    Mat logYstd = Y.unaryExpr(log).eval();

    asap::util::stretch_matrix_columns_inplace(logYstd, qq_min, qq_max, verbose);

    clamp_op<Mat> clamp_std(std_min, std_max);

    exp_op<Mat> exp;
    return Rcpp::NumericMatrix(
        Rcpp::wrap(logYstd.unaryExpr(clamp_std).unaryExpr(exp)));
}
