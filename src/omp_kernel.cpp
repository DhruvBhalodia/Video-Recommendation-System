#include "mf_common.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <fstream>
#include <limits>


#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif

using namespace std;

float predict_rating(const MFModel& model,
                     uint32_t      u,
                     uint32_t      i)
{
    if (u >= model.n_users || i >= model.n_items) {
        return model.global_mean;
    }

    const int k = model.k;
    const float *U = model.U.data();
    const float *V = model.V.data();
    const float bu = model.bu[u];
    const float bi = model.bi[i];

    const uint32_t u_offset = u * k;
    const uint32_t i_offset = i * k;

    float dot = 0.0f;
    for (int f = 0; f < k; ++f) {
        dot += U[u_offset + f] * V[i_offset + f];
    }

    float pred = model.global_mean + bu + bi + dot;
    return pred;
}

double compute_rmse(const vector<Rating>& data,
                    const MFModel&        model)
{
    if (data.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double se = 0.0;       
    uint64_t cnt = 0;

    for (const auto& r : data) {
        float pred = predict_rating(model, r.u, r.i);
        double diff = static_cast<double>(r.r) - static_cast<double>(pred);
        se += diff * diff;
        ++cnt;
    }

    return std::sqrt(se / static_cast<double>(cnt));
}

MFModel train_omp(const Dataset& ds, const TrainConfig& cfg)
{
    const uint32_t n_users = ds.n_users;
    const uint32_t n_items = ds.n_items;
    const int      k       = cfg.k;

    if (n_users == 0 || n_items == 0) {
        throw runtime_error("Empty dataset: n_users or n_items is zero.");
    }
    if (k <= 0) {
        throw runtime_error("Latent dimension k must be positive.");
    }

    MFModel model;
    model.n_users     = n_users;
    model.n_items     = n_items;
    model.k           = k;
    model.global_mean = ds.global_mean;

    model.U.assign(static_cast<size_t>(n_users) * k, 0.0f);
    model.V.assign(static_cast<size_t>(n_items) * k, 0.0f);
    model.bu.assign(n_users, 0.0f);
    model.bi.assign(n_items, 0.0f);

    std::mt19937 rng(12345); 
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (auto &val : model.U) {
        val = dist(rng);
    }
    for (auto &val : model.V) {
        val = dist(rng);
    }
    vector<Rating> train = ds.train; 
    const float eta    = cfg.eta;
    const float lambda = cfg.lambda;
    const int   epochs = cfg.epochs;

    cerr << "Starting OpenMP training with "
         << n_users << " users, "
         << n_items << " items, "
         << train.size() << " train ratings, "
         << ds.test.size() << " test ratings.\n";

#ifdef _OPENMP
    cerr << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#endif
std::ofstream log_file("outputs/omp_training_log.csv");
log_file << "epoch,train_rmse,test_rmse,epoch_time_seconds,num_threads,world_size\n";

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(train.begin(), train.end(), rng);

            auto t_start = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < static_cast<int>(train.size()); ++idx) {
            const Rating& r = train[idx];
            uint32_t u = r.u;
            uint32_t i = r.i;
            float    rating = r.r;

            if (u >= n_users || i >= n_items) {
                continue; 
            }

            const uint32_t u_offset = u * k;
            const uint32_t i_offset = i * k;

            float bu = model.bu[u];
            float bi = model.bi[i];

            float dot = 0.0f;
            for (int f = 0; f < k; ++f) {
                dot += model.U[u_offset + f] * model.V[i_offset + f];
            }

            float pred = model.global_mean + bu + bi + dot;
            float err  = rating - pred;

            for (int f = 0; f < k; ++f) {
                float p_uf = model.U[u_offset + f];
                float q_if = model.V[i_offset + f];
                model.U[u_offset + f] += eta * (err * q_if - lambda * p_uf);
                model.V[i_offset + f] += eta * (err * p_uf - lambda * q_if);
            }

            model.bu[u] += eta * (err - lambda * bu);
            model.bi[i] += eta * (err - lambda * bi);
        }

        double train_rmse = compute_rmse(train, model);
        double test_rmse  = compute_rmse(ds.test, model);

        cerr << "Epoch " << epoch
             << " | Train RMSE: " << train_rmse
             << " | Test RMSE: "  << test_rmse
             << "\n";
        auto t_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> dt = t_end - t_start;
        double epoch_time = dt.count();

        int num_threads = 1;
        #ifdef _OPENMP
        num_threads = omp_get_max_threads();
        #endif

        log_file << epoch << ","
                << train_rmse << ","
                << test_rmse << ","
                << epoch_time << ","
                << num_threads << ","
                << 1         
                << "\n";

    }

    cerr << "OpenMP training completed.\n";
    return model;
}
