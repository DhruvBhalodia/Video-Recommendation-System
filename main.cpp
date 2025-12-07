#include "mf_common.h"

#include <mpi.h>
#include <iostream>
#include <string>

using namespace std;

static void print_usage(const char* prog)
{
    cerr << "Usage:\n";
    cerr << "  " << prog << " omp  <path_to_ratings.dat>\n";
    cerr << "  " << prog << " mpi  <path_to_ratings.dat>\n";
    cerr << "\n";
    cerr << "Example:\n";
    cerr << "  " << prog << " omp data/ratings.dat\n";
    cerr << "  mpirun -np 4 " << prog << " mpi data/ratings.dat\n";
}

static void print_config(const TrainConfig& cfg)
{
    cerr << "Config: k=" << cfg.k
         << ", eta=" << cfg.eta
         << ", lambda=" << cfg.lambda
         << ", epochs=" << cfg.epochs
         << ", test_ratio=" << cfg.test_ratio
         << "\n";
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode         = argv[1];
    std::string ratings_path = argv[2];

    TrainConfig cfg;

    if (mode == "omp") {
        cerr << "[MAIN] Running in OMP mode.\n";
        print_config(cfg);

        MFModel model = run_training_omp(ratings_path, cfg);
        save_model(model, "outputs/trained_model_omp.bin");
        cerr << "[MAIN] Model saved to outputs/trained_model.bin\n";

        if (model.n_users > 0) {
            std::uint32_t demo_user = 0;
            cout << "\n=== OMP: Top-10 recommendations for user " << demo_user << " ===\n";
            recommend_top_k_for_user(model, demo_user, 10);
        } else {
            cerr << "[MAIN] Model has zero users, cannot run recommendation demo.\n";
        }

        return 0;
    }
    else if (mode == "mpi") {
        MPI_Init(&argc, &argv);

        int world_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        if (world_rank == 0) {
            cerr << "[MAIN] Running in MPI mode with " << world_size << " ranks.\n";
            print_config(cfg);
        }

        MFModel model = run_training_mpi(ratings_path, cfg);
        save_model(model, "outputs/trained_model_mpi.bin");
        cerr << "[MAIN] Model saved to outputs/trained_model.bin\n";

        if (world_rank == 0) {
            if (model.n_users > 0) {
                std::uint32_t demo_user = 0;
                cout << "\n=== MPI: Top-10 recommendations for user " << demo_user << " ===\n";
                recommend_top_k_for_user(model, demo_user, 10);
            } else {
                cerr << "[MAIN] Model has zero users, cannot run recommendation demo.\n";
            }
        }

        MPI_Finalize();
        return 0;
    }
    else if (mode == "infer") {
        if (argc < 4) {
            cerr << "Usage: ./parallel_mf infer <model_path> <user_id>\n";
            return 1;
        }

        string model_path = argv[2];
        uint32_t user_id = stoi(argv[3]);

        MFModel model = load_model(model_path);
        recommend_top_k_for_user(model, user_id, 10);
        return 0;
    }
    else {
        cerr << "[MAIN] Unknown mode: '" << mode << "'. Use 'omp' or 'mpi'.\n";
        print_usage(argv[0]);
        return 1;
    }
}