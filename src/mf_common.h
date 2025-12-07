#ifndef MF_COMMON_H
#define MF_COMMON_H

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>

struct Rating {
    std::uint32_t u;
    std::uint32_t i; 
    float         r;
};

struct Dataset {
    std::vector<Rating> train;
    std::vector<Rating> test;

    std::uint32_t n_users = 0;
    std::uint32_t n_items = 0;

    float global_mean = 0.0f; 
};

struct TrainConfig {
    int   k          = 50; 
    float eta        = 0.01f; 
    float lambda     = 0.05f; 
    int   epochs     = 20;    

    float test_ratio = 0.2f; 
};

struct MFModel {
    std::vector<float> U; 
    std::vector<float> V;  
    std::vector<float> bu;
    std::vector<float> bi; 

    std::uint32_t n_users = 0;
    std::uint32_t n_items = 0;
    int           k       = 0;

    float global_mean = 0.0f; 
};

Dataset load_movielens(const std::string& ratings_path,
                       float              test_ratio = 0.2f,
                       std::uint32_t      random_seed = 42);

MFModel train_omp(const Dataset& ds, const TrainConfig& cfg);

MFModel train_mpi(const Dataset& ds, const TrainConfig& cfg);

float predict_rating(const MFModel& model,
                     std::uint32_t  u,
                     std::uint32_t  i);

double compute_rmse(const std::vector<Rating>& data,
                    const MFModel&             model);

MFModel run_training_omp(const std::string& ratings_path,
                         const TrainConfig& cfg);

MFModel run_training_mpi(const std::string& ratings_path,
                         const TrainConfig& cfg);

void recommend_top_k_for_user(const MFModel& model,
                              std::uint32_t  user_id,
                              int            K);

inline void save_model(const MFModel& model, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to save model: " + path);
    }

    out.write((char*)&model.n_users, sizeof(model.n_users));
    out.write((char*)&model.n_items, sizeof(model.n_items));
    out.write((char*)&model.k, sizeof(model.k));
    out.write((char*)&model.global_mean, sizeof(model.global_mean));

    out.write((char*)model.U.data(), model.U.size() * sizeof(float));
    out.write((char*)model.V.data(), model.V.size() * sizeof(float));
    out.write((char*)model.bu.data(), model.bu.size() * sizeof(float));
    out.write((char*)model.bi.data(), model.bi.size() * sizeof(float));
}

inline MFModel load_model(const std::string& path) {
    MFModel model;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to load model: " + path);
    }

    in.read((char*)&model.n_users, sizeof(model.n_users));
    in.read((char*)&model.n_items, sizeof(model.n_items));
    in.read((char*)&model.k, sizeof(model.k));
    in.read((char*)&model.global_mean, sizeof(model.global_mean));

    model.U.resize((size_t)model.n_users * model.k);
    model.V.resize((size_t)model.n_items * model.k);
    model.bu.resize(model.n_users);
    model.bi.resize(model.n_items);

    in.read((char*)model.U.data(), model.U.size() * sizeof(float));
    in.read((char*)model.V.data(), model.V.size() * sizeof(float));
    in.read((char*)model.bu.data(), model.bu.size() * sizeof(float));
    in.read((char*)model.bi.data(), model.bi.size() * sizeof(float));

    return model;
}


#endif 