#include "mmutil.hh"

#ifndef ASAP_NMF_TRAIN_HH_
#define ASAP_NMF_TRAIN_HH_

struct train_nmf_options_t {
    std::size_t max_iter;
    std::size_t burnin;
    double eps;
    bool svd_init;
    bool verbose;
};

template <typename MODEL, typename YDATA, typename ROWNET, typename COLNET>
void
train_nmf(MODEL &model,
          const YDATA &y,
          const ROWNET &rownet,
          const COLNET &colnet,
          std::vector<Scalar> &llik_trace,
          const train_nmf_options_t &options)
{
    const std::size_t max_iter = options.max_iter;
    const std::size_t burnin = options.burnin;
    const double EPS = options.eps;
    const bool verbose = options.verbose;
    const bool svd_init = options.svd_init;

    model.precompute(y, rownet, colnet);
    TLOG_(verbose, "pre-computation");

    if (svd_init) {
        model.initialize_by_svd(y);
        TLOG_(verbose, "Initialized by SVD");
    } else {
        model.initialize_random();
        TLOG_(verbose, "Randomly initialized");
    }

    llik_trace.reserve(max_iter + burnin);

    typename MODEL::NULL_DATA null_data;

    for (std::size_t tt = 0; tt < (burnin + max_iter); ++tt) {

        model.update_by_col(y,
			    null_data,
                            colnet,
                            typename MODEL::STOCH(tt < burnin),
                            typename MODEL::STD(true));

        model.update_by_row(y,
			    null_data,
                            rownet,
                            typename MODEL::STOCH(tt < burnin),
                            typename MODEL::STD(false));

        const Scalar llik = model.log_likelihood(y);

        const Scalar diff = llik_trace.size() > 0 ?
            std::abs(llik - llik_trace.at(llik_trace.size() - 1)) /
                std::abs(llik + EPS) :
            llik;

        if (tt >= burnin) {
            llik_trace.emplace_back(llik);
        }

        TLOG_(verbose, "NMF [ " << tt << " ] " << llik << ", " << diff);

        if (tt > burnin && diff < EPS) {
            TLOG("Converged at " << tt << ", " << diff);
            break;
        }

        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            WLOG("Interruption by the user at t=" << tt);
            break;
        }
    }
    TLOG("NMF --> done");
}

#endif
