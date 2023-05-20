#include "rcpp_asap_batch.hh"

int
batch_info_t::check()
{

    const Index max_batch =
        *std::max_element(membership.begin(), membership.end());
    const Index min_batch =
        *std::min_element(membership.begin(), membership.end());

    ASSERT_RET(max_batch < B && min_batch >= 0,
               "[" << min_batch << ", " << max_batch << "]"
                   << " vs. ncol(batch_effect) = " << B);

    ASSERT_RET(membership.size() == N,
               "Must have batch membership for each cell");

    ASSERT_RET(delta_db.rows() == D && delta_db.cols() > 0,
               "Found incompatible matrix" << delta_db.rows() << " x "
                                           << delta_db.cols() << " vs. " << D);

    ASSERT_RET(delta_db.minCoeff() >= 0,
               "The batch effect matrix must be positive: "
                   << delta_db.minCoeff());

    return EXIT_SUCCESS;
}

int
batch_info_t::read(
    const Rcpp::Nullable<Rcpp::NumericMatrix> r_batch_effect = R_NilValue,
    const Rcpp::Nullable<Rcpp::IntegerVector> r_batch_membership = R_NilValue)
{
    ////////////////////////////
    // read D x B bias matrix //
    ////////////////////////////
    delta_db = r_batch_effect.isNotNull() ?
        Rcpp::as<Mat>(Rcpp::NumericMatrix(r_batch_effect)) :
        Mat::Ones(D, 1);

    //////////////////////////////////
    // read N x 1 membership vector //
    //////////////////////////////////
    std::vector<Index> &batch_membership = membership;

    if (r_batch_membership.isNotNull()) {
        const Rcpp::IntegerVector _batch(r_batch_membership);
        batch_membership.reserve(_batch.size());
        for (auto b : _batch) {
            batch_membership.emplace_back(b);
        }

    } else {
        batch_membership.resize(N);
        std::fill(batch_membership.begin(), batch_membership.end(), 0);
    }

    B = delta_db.cols();

    return check();
}
