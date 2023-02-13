#include "rcpp_asap.hh"

//' Non-negative matrix factorization
//'
//' @param Y data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param mcem number of Monte Carl Expectation Maximization
//' @param burnin burn-in period
//' @param latent_iter latent sampling steps
//' @param thining thining interval in record keeping
//' @param verbose verbosity
//' @param eval_llik evaluate log-likelihood
//' @param a0 gamma(a0, b0)
//' @param b0 gamma(a0, b0)
//' @param rseed random seed
//' @param NUM_THREADS number of parallel jobs
//'
// [[Rcpp::export]]
Rcpp::List
asap_fit_nmf(const Eigen::MatrixXf Y,
             const std::size_t maxK,
             const std::size_t mcem = 100,
             const std::size_t burnin = 10,
             const std::size_t latent_iter = 10,
             const std::size_t thining = 3,
             const bool verbose = true,
             const bool eval_llik = true,
             const double a0 = 1.,
             const double b0 = 1.,
             const std::size_t rseed = 42,
             const std::size_t NUM_THREADS = 1,
             const bool update_loading = true,
             const bool gibbs_sampling = false)
{

    const Index D = Y.rows();
    const Index N = Y.cols();
    const Index K = std::min(static_cast<Index>(maxK), N);

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    using gamma_t = gamma_param_t<Mat, RNG>;

    poisson_nmf_t<Mat, RNG, gamma_t> model(D, N, K, a0, b0, rseed);
    using latent_t = latent_matrix_t<RNG>;
    latent_t aux(D, N, K, rng);

    running_stat_t<Mat> dict_stat(D, K);
    running_stat_t<Mat> log_dict_stat(D, K);
    running_stat_t<Mat> loading_stat(K, 1);
    running_stat_t<Mat> column_stat(N, K);

    model.initialize_degree(Y);

    if (verbose)
        TLOG("Randomizing auxiliary/latent matrix");

    aux.randomize();
    model.initialize_by_svd(Y);

    if (verbose)
        TLOG("Initialized model parameters");

    Scalar llik;
    std::vector<Scalar> llik_trace;

    if (eval_llik) {
        llik = model.log_likelihood(Y, aux);
        llik_trace.emplace_back(llik);
        if (verbose)
            TLOG("Initial log-likelihood: " << llik);
    }

    matrix_sampler_t<Mat, RNG> row_proposal(rng, K);

    Mat logP_DK(D, K);

    for (std::size_t t = 0; t < (mcem + burnin); ++t) {

        logP_DK = model.row_topic.log_mean().array().rowwise() +
            model.take_topic_log_loading().transpose().array();

        for (std::size_t s = 0; s < latent_iter; ++s) {

            if (gibbs_sampling) {
                aux.gibbs_sample_row_col(logP_DK,
                                         model.column_topic.log_mean(),
                                         NUM_THREADS);
            } else {
                aux.mh_sample_row_col(row_proposal.sample_logit(logP_DK),
                                      model.column_topic.log_mean(),
                                      NUM_THREADS);
            }
        }

        if (update_loading) {
            model.update_topic_loading(Y, aux);
        }
        model.update_row_topic(Y, aux);
        model.update_column_topic(Y, aux);
        model.update_degree(Y);

        if (eval_llik && t % thining == 0) {
            llik = model.log_likelihood(Y, aux);
            llik_trace.emplace_back(llik);
        }

        if (verbose && eval_llik) {
            TLOG("NMF MCEM: " << t << " " << llik);
        } else {
            if (t > 0 && t % 10 == 0)
                Rcpp::Rcerr << ". " << std::flush;
            if (t > 0 && t % 100 == 0)
                Rcpp::Rcerr << std::endl;
        }

        if (t >= burnin && t % thining == 0) {
            loading_stat(model.topic_loading.mean());
            dict_stat(model.row_topic.mean());
            log_dict_stat(model.row_topic.log_mean());
            column_stat(model.column_topic.mean());
        }

        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            WLOG("Interruption by a user at t=" << t);
            break;
        }
    }

    Rcpp::Rcerr << std::endl;

    auto _summary = [](running_stat_t<Mat> &stat) {
        return Rcpp::List::create(Rcpp::_["mean"] = stat.mean(),
                                  Rcpp::_["sd"] = stat.sd());
    };

    Rcpp::List deg_out =
        Rcpp::List::create(Rcpp::_["row"] = model.row_degree.mean(),
                           Rcpp::_["column"] = model.column_degree.mean());

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["degree"] = deg_out,
                              Rcpp::_["dict"] = _summary(dict_stat),
                              Rcpp::_["log.dict"] = _summary(log_dict_stat),
                              Rcpp::_["loading"] = _summary(loading_stat),
                              Rcpp::_["column"] = _summary(column_stat));

    return Rcpp::List::create();
}
