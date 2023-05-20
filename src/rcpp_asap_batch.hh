#include "rcpp_asap.hh"

#ifndef RCPP_ASAP_BATCH_HH_
#define RCPP_ASAP_BATCH_HH_

struct batch_info_t {
    explicit batch_info_t(const Index d, const Index n)
        : D(d)
        , N(n)
    {
        B = 1; // at least one batch
    }
    const Index D;
    const Index N;

    int read(const Rcpp::Nullable<Rcpp::NumericMatrix> r_batch_effect,
             const Rcpp::Nullable<Rcpp::IntegerVector> r_batch_membership);

    int check();

    Mat delta_db;
    std::vector<Index> membership;

private:
    Index B;
};

#endif
