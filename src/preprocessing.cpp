#include "mf_common.h"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <random>
#include <stdexcept>
#include <iostream>

using namespace std;

static bool parse_ratings_line(const string& line,
                               string&       user_raw,
                               string&       item_raw,
                               float&        rating_val)
{
    if (line.empty()) return false;

    size_t pos1 = line.find("::");
    if (pos1 != string::npos) {
        size_t pos2 = line.find("::", pos1 + 2);
        if (pos2 == string::npos) return false;
        size_t pos3 = line.find("::", pos2 + 2);

        string rating_str;
        user_raw = line.substr(0, pos1);
        item_raw = line.substr(pos1 + 2, pos2 - (pos1 + 2));

        if (pos3 == string::npos) {
            rating_str = line.substr(pos2 + 2);
        } else {
            rating_str = line.substr(pos2 + 2, pos3 - (pos2 + 2));
        }

        try {
            rating_val = std::stof(rating_str);
        } catch (...) {
            return false;
        }
        return true;
    }

    istringstream iss(line);
    string rating_str;
    string timestamp_dummy;

    if (!(iss >> user_raw >> item_raw >> rating_str)) {
        return false;
    }
    iss >> timestamp_dummy;

    try {
        rating_val = std::stof(rating_str);
    } catch (...) {
        return false;
    }

    return true;
}

Dataset load_movielens(const string& ratings_path,
                       float         test_ratio,
                       uint32_t      random_seed)
{
    if (test_ratio < 0.0f || test_ratio >= 1.0f) {
        throw invalid_argument("test_ratio must be in [0.0, 1.0).");
    }

    ifstream fin(ratings_path);
    if (!fin.is_open()) {
        throw runtime_error("Failed to open ratings file: " + ratings_path);
    }

    unordered_map<string, uint32_t> user_map;
    unordered_map<string, uint32_t> item_map;

    vector<Rating> all_ratings;
    all_ratings.reserve(1000000);

    uint32_t next_user = 0;
    uint32_t next_item = 0;

    double   sum_ratings   = 0.0;
    uint64_t count_ratings = 0;

    string line;
    while (std::getline(fin, line)) {
        string user_raw, item_raw;
        float  rating_val = 0.0f;

        if (!parse_ratings_line(line, user_raw, item_raw, rating_val)) {
            static bool warned = false;
            if (!warned) {
                cerr << "Warning: encountered malformed line in ratings file, skipping.\n";
                warned = true;
            }
            continue;
        }

        uint32_t u_idx;
        auto it_u = user_map.find(user_raw);
        if (it_u == user_map.end()) {
            u_idx = next_user;
            user_map[user_raw] = next_user;
            ++next_user;
        } else {
            u_idx = it_u->second;
        }

        uint32_t i_idx;
        auto it_i = item_map.find(item_raw);
        if (it_i == item_map.end()) {
            i_idx = next_item;
            item_map[item_raw] = next_item;
            ++next_item;
        } else {
            i_idx = it_i->second;
        }

        Rating r;
        r.u = u_idx;
        r.i = i_idx;
        r.r = rating_val;

        all_ratings.push_back(r);
        sum_ratings   += rating_val;
        count_ratings += 1;
    }

    if (count_ratings == 0) {
        throw runtime_error("No valid ratings found in file: " + ratings_path);
    }

    Dataset ds;
    ds.n_users     = next_user;
    ds.n_items     = next_item;
    ds.global_mean = static_cast<float>(sum_ratings / static_cast<double>(count_ratings));

    mt19937 rng(random_seed);
    uniform_real_distribution<float> dist(0.0f, 1.0f);

    ds.train.reserve(all_ratings.size());
    ds.test.reserve(static_cast<size_t>(all_ratings.size() * test_ratio) + 1);

    for (const auto& r : all_ratings) {
        float x = dist(rng);
        if (x < test_ratio) {
            ds.test.push_back(r);
        } else {
            ds.train.push_back(r);
        }
    }

    if (ds.train.empty()) {
        throw runtime_error("Train set is empty after splitting; adjust test_ratio.");
    }
    if (ds.test.empty()) {
        cerr << "Warning: test set is empty after splitting; consider increasing test_ratio.\n";
    }

    cerr << "Loaded MovieLens ratings from: " << ratings_path << "\n";
    cerr << "Users: " << ds.n_users << ", Items: " << ds.n_items
         << ", Total ratings: " << count_ratings << "\n";
    cerr << "Train size: " << ds.train.size()
         << ", Test size: " << ds.test.size() << "\n";
    cerr << "Global mean rating: " << ds.global_mean << "\n";

    return ds;
}
