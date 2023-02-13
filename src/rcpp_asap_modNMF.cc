#include "rcpp_asap.hh"

//' Non-negative matrix factorization with the row modules
//'
//' @param Y data matrix (gene x sample)
//' @param maxK maximum number of factors
//' @param maxL maximum number of row modules
//' @param collapsing L x row collapsing matrix (L < gene)
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
asap_fit_modular_nmf(
    const Eigen::MatrixXf Y,
    const std::size_t maxK,
    const std::size_t maxL,
    Rcpp::Nullable<Rcpp::NumericMatrix> collapsing = R_NilValue,
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

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    using gamma_t = gamma_param_t<Mat, RNG>;

    const Index D = Y.rows();
    const Index N = Y.cols();
    const Index K = std::min(static_cast<Index>(maxK), N);
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;
    const ColVec y_sum = Y.rowwise().sum(); // D x 1

    Mat C_DL;
    if (collapsing.isNotNull()) {
        C_DL = Rcpp::as<Mat>(Rcpp::NumericMatrix(collapsing));
        ASSERT_RETL(C_DL.rows() == D, "incompatible collapsing matrix");
    }

    const Index L = collapsing.isNotNull() ? C_DL.cols() : maxL;

    Mat lnG_NL(N, L), tempNL(N, L), logP_DL(D, L);
    logP_DL.setZero();

    matrix_sampler_t<Mat, RNG> row_clust_sampler(rng, L);
    using Idx = matrix_sampler_t<Mat, RNG>::IndexVec;

    /////////////////////////////////////////
    // initialization of clustering matrix //
    /////////////////////////////////////////

    if (collapsing.isNull()) {

        if (verbose)
            TLOG("Initialization of module membership...");

        using norm_dist_t = boost::random::normal_distribution<Scalar>;
        norm_dist_t norm_dist(0., 1.);
        auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };

        const Index depth = std::ceil(fasterlog(L) / fasterlog(2));

        Mat R = Mat::NullaryExpr(Y.cols(), depth, rnorm);
        Mat Q = Y * R;

        const std::size_t lu_iter = 5;          // this should be good
        RandomizedSVD<Mat> svd(depth, lu_iter); //

        normalize_columns(Q);
        svd.compute(Q);
        Mat rsign = standardize(svd.matrixU()); // D x depth
        IntVec bb(Y.rows());
        bb.setZero();

        for (Index k = 0; k < depth; ++k) {
            auto binary_shift = [&k](const Scalar &x) -> Index {
                return x > 0. ? (1 << k) : 0;
            };
            bb += rsign.col(k).unaryExpr(binary_shift);
        }

        C_DL.resize(Y.rows(), L);
        C_DL.setZero();
        for (Index i = 0; i < Y.rows(); ++i) {
            C_DL(i, bb(i) % L) = 1;
        }
    }

    poisson_modular_nmf_t<Mat, RNG, gamma_t> model(D, N, K, L, a0, b0, rseed);
    using latent_t = latent_matrix_t<RNG>;
    latent_t Z_NL(N, L, K, rng);

    //////////////////////////
    // degree distributions //
    //////////////////////////

    model.initialize_degree(Y);

    if (verbose)
        TLOG("Randomizing auxiliary/latent matrix");

    Z_NL.randomize();
    model.initialize_by_svd(Y);

    if (verbose)
        TLOG("Initialized model parameters");

    Scalar llik;
    std::vector<Scalar> llik_trace;

    if (eval_llik) {
        llik = model.log_likelihood(Y, C_DL, Z_NL);
        llik_trace.emplace_back(llik);
        if (verbose)
            TLOG("Initial log-likelihood: " << llik);
    }

    running_stat_t<Mat> dict_stat(D, K);
    running_stat_t<Mat> row_stat(D, L);
    running_stat_t<Mat> C_stat(D, L);
    running_stat_t<Mat> middle_stat(L, K);
    running_stat_t<Mat> column_stat(N, K);

    Mat logP_LK(L, K);

    matrix_sampler_t<Mat, RNG> col_proposal(rng, K);

    for (std::size_t t = 0; t < (mcem + burnin); ++t) {

        //////////////////////////////
        // Update the middle factor //
        //////////////////////////////

        logP_LK = model.middle_LK.log_mean().array().rowwise() +
            model.get_loading_logK().transpose().array();

        for (Index s = 0; s < latent_iter; ++s) {

            if (gibbs_sampling) {
                Z_NL.gibbs_sample_row_col(model.column_NK.log_mean(),
                                          logP_LK,
                                          NUM_THREADS);
            } else {

                Z_NL.mh_sample_col_row(col_proposal.sample_logit(logP_LK),
                                       model.column_NK.log_mean(),
                                       NUM_THREADS);
            }
        }

        if (update_loading) {
            model.update_loading_K(Y, C_DL, Z_NL);
        }
        model.update_middle_topic(Y, C_DL, Z_NL);

        ///////////////////////////////
        // update the column factors //
        ///////////////////////////////

        logP_LK = model.middle_LK.log_mean().array().rowwise() +
            model.get_loading_logK().transpose().array();

        for (Index s = 0; s < latent_iter; ++s) {

            if (gibbs_sampling) {
                Z_NL.gibbs_sample_row_col(model.column_NK.log_mean(),
                                          logP_LK,
                                          NUM_THREADS);
            } else {

                Z_NL.mh_sample_col_row(col_proposal.sample_logit(logP_LK),
                                       model.column_NK.log_mean(),
                                       NUM_THREADS);
            }
        }

        if (update_loading) {
            model.update_loading_K(Y, C_DL, Z_NL);
        }

        model.update_column_topic(Y, C_DL, Z_NL);

        /////////////////////////////////////
        // latent variables for clustering //
        /////////////////////////////////////

        lnG_NL.setZero();

        for (Index k = 0; k < K; ++k) {
            tempNL.setZero();
            tempNL.colwise() += model.column_NK.log_mean().col(k);
            tempNL.rowwise() += model.middle_LK.log_mean().col(k).transpose();
            lnG_NL += Z_NL.slice_k(tempNL, k);
        }

        logP_DL = (Y * lnG_NL).array().colwise() / y_sum.array();

        {
            C_DL.setZero();
            const Idx &sampled = row_clust_sampler.sample_logit(logP_DL);
            for (Index i = 0; i < D; ++i) {
                C_DL(i, sampled.at(i)) = 1.;
            }
        }

        model.update_row_topic(Y, C_DL, Z_NL);

        model.update_degree(Y);

        if (eval_llik) {
            llik = model.log_likelihood(Y, C_DL, Z_NL);
            llik_trace.emplace_back(llik);
        }

        if (verbose && eval_llik) {
            TLOG("modNMF MCEM: " << t << " " << llik);
        } else {
            if (t > 0 && t % 10 == 0)
                Rcpp::Rcerr << ". " << std::flush;
            if (t > 0 && t % 100 == 0)
                Rcpp::Rcerr << std::endl;
        }

        if (t >= burnin && t % thining == 0) {
            C_stat(C_DL);
            dict_stat(model.row_DL.mean() * model.middle_LK.mean());
            row_stat(model.row_DL.mean());
            middle_stat(model.middle_LK.mean());
            column_stat(model.column_NK.mean());
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
        Rcpp::List::create(Rcpp::_["row"] = model.row_D.mean(),
                           Rcpp::_["column"] = model.column_N.mean());

    return Rcpp::List::create(Rcpp::_["log.likelihood"] = llik_trace,
                              Rcpp::_["degree"] = deg_out,
                              Rcpp::_["row.clust"] = _summary(C_stat),
                              Rcpp::_["dict"] = _summary(dict_stat),
                              Rcpp::_["row"] = _summary(row_stat),
                              Rcpp::_["middle"] = _summary(middle_stat),
                              Rcpp::_["column"] = _summary(column_stat),
                              Rcpp::_["C"] = C_DL);
}
