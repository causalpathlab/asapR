#include "rcpp_asap.hh"
#include "rcpp_util.hh"
#include "rcpp_mtx_data.hh"
#include "rcpp_eigenSparse_data.hh"
#include "rcpp_asap_regression.hh"

#ifndef RCPP_ASAP_INTERACTION_NETWORK_COLS_HH_
#define RCPP_ASAP_INTERACTION_NETWORK_COLS_HH_

template <typename Data, typename Derived, typename OPT>
int
run_asap_regression_both(Data &lhs_data,
                         Data &rhs_data,
                         const Eigen::MatrixBase<Derived> &log_beta,
                         const Rcpp::Nullable<Eigen::MatrixXf> r_log_delta,
                         const std::vector<std::string> &pos2row,
                         const OPT &regOpt,
                         const bool self_interaction,
                         Eigen::MatrixBase<Derived> &lhs_kn,
                         Eigen::MatrixBase<Derived> &rhs_km)
{
    Derived &Q_lhs_kn = lhs_kn.derived();
    Derived &Q_rhs_km = rhs_km.derived();

    ASSERT_RET(lhs_data.max_row() == rhs_data.max_row(),
               "lhs and rhs should have the same number of features");

    if (r_log_delta.isNotNull()) {
        Mat log_delta = Rcpp::as<Mat>(r_log_delta);
        Mat lhs_nk, rhs_mk, Y_n, Y_m;

        ASSERT_RET(log_delta.rows() == log_beta.rows(),
                   "rows(delta) != rows(beta)");

        CHK_RET_(asap::regression::run_pmf_stat_adj(lhs_data,
                                                    log_beta,
                                                    log_delta,
                                                    pos2row,
                                                    regOpt,
                                                    lhs_nk,
                                                    Y_n),
                 "unable to compute topic statistics on the left hand side");

        Q_lhs_kn = lhs_nk.transpose();

        if (self_interaction) {
            Q_rhs_km = Q_lhs_kn;
        } else {
            CHK_RET_(
                asap::regression::run_pmf_stat_adj(rhs_data,
                                                   log_beta,
                                                   log_delta,
                                                   pos2row,
                                                   regOpt,
                                                   rhs_mk,
                                                   Y_m),
                "unable to compute topic statistics on the right hand side");
            Q_rhs_km = rhs_mk.transpose();
        }

    } else {

        Mat lhs_nk, rhs_mk, Y_n, Y_m;

        CHK_RET_(asap::regression::run_pmf_stat(lhs_data,
                                                log_beta,
                                                pos2row,
                                                regOpt,
                                                lhs_nk,
                                                Y_n),
                 "unable to compute topic statistics on the left hand side");

        Q_lhs_kn = lhs_nk.transpose();

        if (self_interaction) {
            Q_rhs_km = Q_lhs_kn;
        } else {
            CHK_RET_(
                asap::regression::run_pmf_stat(rhs_data,
                                               log_beta,
                                               pos2row,
                                               regOpt,
                                               rhs_mk,
                                               Y_m),
                "unable to compute topic statistics on the right hand side");
            Q_rhs_km = rhs_mk.transpose();
        }
    }

    return EXIT_SUCCESS;
}

#endif
