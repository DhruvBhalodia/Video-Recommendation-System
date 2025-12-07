#include "mf_common.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <fstream>
#include <sstream>

using namespace std;

static bool g_movie_meta_loaded = false;
static std::vector<std::uint32_t> g_idx_to_raw_item;
static std::unordered_map<std::uint32_t, std::string> g_movie_titles;

static void build_item_mapping_from_ratings(const std::string& ratings_path)
{
    g_idx_to_raw_item.clear();
    std::unordered_map<std::uint32_t, std::uint32_t> raw_to_idx;

    std::ifstream in(ratings_path);
    if (!in.is_open()) {
        std::cerr << "[inference] Could not open ratings file " << ratings_path
                  << " to rebuild item mapping. Will fall back to item indices.\n";
        return;
    }

    std::string line;
    bool has_double_colon = false;
    bool format_checked = false;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        if (!format_checked) {
            if (line.find("::") != std::string::npos) {
                has_double_colon = true;
            }
            format_checked = true;
        }

        std::uint32_t raw_user = 0;
        std::uint32_t raw_item = 0;
        float rating = 0.0f;

        if (has_double_colon) {
            size_t p1 = line.find("::");
            size_t p2 = (p1 == std::string::npos) ? std::string::npos : line.find("::", p1 + 2);
            if (p1 == std::string::npos || p2 == std::string::npos) {
                continue;
            }
            raw_user = static_cast<std::uint32_t>(std::stoul(line.substr(0, p1)));
            raw_item = static_cast<std::uint32_t>(std::stoul(line.substr(p1 + 2, p2 - (p1 + 2))));
        } else {
            std::stringstream ss(line);
            ss >> raw_user >> raw_item >> rating;
        }

        auto it = raw_to_idx.find(raw_item);
        if (it == raw_to_idx.end()) {
            std::uint32_t new_idx = static_cast<std::uint32_t>(g_idx_to_raw_item.size());
            raw_to_idx[raw_item] = new_idx;
            g_idx_to_raw_item.push_back(raw_item);
        }
    }

    std::cerr << "[inference] Built item mapping from ratings.dat: "
              << g_idx_to_raw_item.size() << " items.\n";
}

static void load_movie_titles(const std::string& movies_path)
{
    g_movie_titles.clear();

    std::ifstream in(movies_path);
    if (!in.is_open()) {
        std::cerr << "[inference] Could not open movies file " << movies_path
                  << ". Will only print item indices.\n";
        return;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        size_t p1 = line.find("::");
        size_t p2 = (p1 == std::string::npos) ? std::string::npos : line.find("::", p1 + 2);
        if (p1 == std::string::npos || p2 == std::string::npos) {
            continue;
        }

        std::uint32_t movie_id = static_cast<std::uint32_t>(std::stoul(line.substr(0, p1)));
        std::string title = line.substr(p1 + 2, p2 - (p1 + 2));

        g_movie_titles[movie_id] = title;
    }

    std::cerr << "[inference] Loaded " << g_movie_titles.size()
              << " movie titles from movies.dat.\n";
}

static void ensure_movie_metadata_loaded()
{
    if (g_movie_meta_loaded) return;

    const std::string ratings_path = "data/ratings.dat";
    const std::string movies_path  = "data/movies.dat";

    build_item_mapping_from_ratings(ratings_path);
    load_movie_titles(movies_path);

    g_movie_meta_loaded = true;
}

void recommend_top_k_for_user(const MFModel& model,
                              std::uint32_t  user_id,
                              int            K)
{
    if (user_id >= model.n_users) {
        std::cerr << "[inference] Invalid user_id: " << user_id
                  << " (n_users = " << model.n_users << ")\n";
        return;
    }
    if (K <= 0) {
        std::cerr << "[inference] K must be positive.\n";
        return;
    }
    if (model.n_items == 0) {
        std::cerr << "[inference] Model has zero items.\n";
        return;
    }

    ensure_movie_metadata_loaded();

    std::vector<std::pair<std::uint32_t, float>> scores;
    scores.reserve(model.n_items);

    for (std::uint32_t item = 0; item < model.n_items; ++item) {
        float pred = predict_rating(model, user_id, item);
        scores.emplace_back(item, pred);
    }

    if (K > static_cast<int>(scores.size())) {
        K = static_cast<int>(scores.size());
    }

    std::nth_element(
        scores.begin(),
        scores.begin() + K,
        scores.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second; 
        }
    );

    std::sort(
        scores.begin(),
        scores.begin() + K,
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        }
    );

    std::cout << "Top " << K << " recommended items for user " << user_id << ":\n";

    for (int idx = 0; idx < K; ++idx) {
        auto pair = scores[idx];
        std::uint32_t item_idx = pair.first;
        float score = pair.second;

        std::uint32_t raw_movie_id = item_idx;
        std::string title = "<unknown>";

        if (!g_idx_to_raw_item.empty() && item_idx < g_idx_to_raw_item.size()) {
            raw_movie_id = g_idx_to_raw_item[item_idx];
        }

        auto it = g_movie_titles.find(raw_movie_id);
        if (it != g_movie_titles.end()) {
            title = it->second;
        }

        std::cout << "  item_idx " << item_idx
                  << " | movie_id " << raw_movie_id
                  << " | title \"" << title << "\""
                  << " | predicted rating = " << score
                  << "\n";
    }
}