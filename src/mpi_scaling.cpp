#include "mf_common.h"
#include <mpi.h>
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

MFModel train_mpi(const Dataset& ds, const TrainConfig& cfg)
{
    int world_size = 1;
    int world_rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const uint32_t n_users = ds.n_users;
    const uint32_t n_items = ds.n_items;
    const int      k       = cfg.k;

    if (n_users == 0 || n_items == 0) {
        if (world_rank == 0) {
            cerr << "Error: empty dataset: n_users or n_items is zero.\n";
        }
        throw runtime_error("Empty dataset in train_mpi.");
    }
    if (k <= 0) {
        if (world_rank == 0) {
            cerr << "Error: latent dimension k must be positive.\n";
        }
        throw runtime_error("Latent dimension k must be positive.");
    }

    vector<Rating> local_train;
    vector<Rating> local_test;

    local_train.reserve(ds.train.size() / world_size + 1);
    local_test.reserve(ds.test.size() / world_size + 1);

    for (const auto& r : ds.train) {
        if ((r.u % static_cast<uint32_t>(world_size)) == static_cast<uint32_t>(world_rank)) {
            local_train.push_back(r);
        }
    }
    for (const auto& r : ds.test) {
        if ((r.u % static_cast<uint32_t>(world_size)) == static_cast<uint32_t>(world_rank)) {
            local_test.push_back(r);
        }
    }

    if (local_train.empty() && world_rank == 0) {
        cerr << "Warning: local_train is empty on rank 0. "
             << "With this partitioning some ranks may have no data.\n";
    }

    if (world_rank == 0) {
        cerr << "MPI world size: " << world_size << "\n";
        cerr << "Global users: " << n_users
             << ", items: " << n_items
             << ", global train ratings: " << ds.train.size()
             << ", global test ratings: " << ds.test.size() << "\n";
    }

    cerr << "Rank " << world_rank
         << " | local_train size: " << local_train.size()
         << ", local_test size: " << local_test.size() << "\n";

    MFModel model;
    model.n_users     = n_users;
    model.n_items     = n_items;
    model.k           = k;
    model.global_mean = ds.global_mean;

    const size_t size_U  = static_cast<size_t>(n_users) * k;
    const size_t size_V  = static_cast<size_t>(n_items) * k;
    const size_t size_bu = static_cast<size_t>(n_users);
    const size_t size_bi = static_cast<size_t>(n_items);

    model.U.assign(size_U, 0.0f);
    model.V.assign(size_V, 0.0f);
    model.bu.assign(size_bu, 0.0f);
    model.bi.assign(size_bi, 0.0f);

    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (auto& val : model.U) {
        val = dist(rng);
    }
    for (auto& val : model.V) {
        val = dist(rng);
    }
    const float eta    = cfg.eta;
    const float lambda = cfg.lambda;
    const int   epochs = cfg.epochs;

    if (world_rank == 0) {
        cerr << "Starting MPI + OpenMP training...\n";
    }

    vector<float> V_buffer(size_V, 0.0f);
    vector<float> bi_buffer(size_bi, 0.0f);

    std::mt19937 rng_local(12345u + static_cast<uint32_t>(world_rank));

    std::ofstream log_file;
    if (world_rank == 0) {
        std::string filename = "outputs/mpi_training_log_" + std::to_string(world_size) + ".csv";
        log_file.open(filename);
        if (!log_file.is_open()) {
            cerr << "Warning: could not open " << filename << " for writing.\n";
        } else {
            log_file << "epoch,train_rmse,test_rmse,epoch_time_seconds,num_threads,world_size\n";
        }
    }

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t_start = std::chrono::steady_clock::now();

        std::shuffle(local_train.begin(), local_train.end(), rng_local);

#pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < static_cast<int>(local_train.size()); ++idx) {
            const Rating& r = local_train[idx];
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

        MPI_Allreduce(model.V.data(), V_buffer.data(),
                      static_cast<int>(size_V), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(model.bi.data(), bi_buffer.data(),
                      static_cast<int>(size_bi), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        const float inv_world = 1.0f / static_cast<float>(world_size);
        for (size_t idx = 0; idx < size_V; ++idx) {
            model.V[idx] = V_buffer[idx] * inv_world;
        }
        for (size_t idx = 0; idx < size_bi; ++idx) {
            model.bi[idx] = bi_buffer[idx] * inv_world;
        }

        double   local_train_se  = 0.0;
        uint64_t local_train_cnt = 0;

        for (const auto& r : local_train) {
            float pred = predict_rating(model, r.u, r.i);
            double diff = static_cast<double>(r.r) - static_cast<double>(pred);
            local_train_se += diff * diff;
            ++local_train_cnt;
        }

        double   local_test_se  = 0.0;
        uint64_t local_test_cnt = 0;

        for (const auto& r : local_test) {
            float pred = predict_rating(model, r.u, r.i);
            double diff = static_cast<double>(r.r) - static_cast<double>(pred);
            local_test_se += diff * diff;
            ++local_test_cnt;
        }

        double   global_train_se = 0.0;
        double   global_test_se  = 0.0;
        unsigned long long global_train_cnt = 0;
        unsigned long long global_test_cnt  = 0;

        MPI_Allreduce(&local_train_se, &global_train_se, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_test_se,  &global_test_se,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        unsigned long long local_train_cnt_ull = static_cast<unsigned long long>(local_train_cnt);
        unsigned long long local_test_cnt_ull  = static_cast<unsigned long long>(local_test_cnt);

        MPI_Allreduce(&local_train_cnt_ull, &global_train_cnt, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_test_cnt_ull,  &global_test_cnt,  1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        double global_train_rmse = std::numeric_limits<double>::quiet_NaN();
        double global_test_rmse  = std::numeric_limits<double>::quiet_NaN();

        if (global_train_cnt > 0) {
            global_train_rmse = std::sqrt(global_train_se / static_cast<double>(global_train_cnt));
        }
        if (global_test_cnt > 0) {
            global_test_rmse = std::sqrt(global_test_se / static_cast<double>(global_test_cnt));
        }

        auto t_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> dt = t_end - t_start;
        double epoch_time = dt.count();

        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#endif

        if (world_rank == 0) {
            cerr << "Epoch " << epoch
                 << " | Global Train RMSE: " << global_train_rmse
                 << " | Global Test RMSE: "  << global_test_rmse
                 << " | Time: " << epoch_time << " s\n";

            if (log_file.is_open()) {
                log_file << epoch << ","
                         << global_train_rmse << ","
                         << global_test_rmse  << ","
                         << epoch_time << ","
                         << num_threads << ","
                         << world_size
                         << "\n";
            }
        }
    }

    if (world_rank == 0) {
        cerr << "MPI + OpenMP training completed.\n";
    }

    return model;
}