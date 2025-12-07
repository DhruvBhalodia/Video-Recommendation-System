#include "mf_common.h"

#include <mpi.h>
#include <iostream>

using namespace std;

MFModel run_training_omp(const std::string& ratings_path,
                         const TrainConfig& cfg)
{
    cerr << "[OMP] Loading dataset from: " << ratings_path << "\n";
    Dataset ds = load_movielens(ratings_path, cfg.test_ratio, 42u);

    cerr << "[OMP] Training configuration:\n";
    cerr << "       k = "        << cfg.k
         << ", eta = "          << cfg.eta
         << ", lambda = "       << cfg.lambda
         << ", epochs = "       << cfg.epochs
         << ", test_ratio = "   << cfg.test_ratio
         << "\n";

    MFModel model = train_omp(ds, cfg);
    cerr << "[OMP] Training finished.\n";

    return model;
}

MFModel run_training_mpi(const std::string& ratings_path,
                         const TrainConfig& cfg)
{
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        cerr << "[MPI] Loading dataset on all ranks from: " << ratings_path << "\n";
        cerr << "[MPI] Training configuration:\n";
        cerr << "      k = "        << cfg.k
             << ", eta = "          << cfg.eta
             << ", lambda = "       << cfg.lambda
             << ", epochs = "       << cfg.epochs
             << ", test_ratio = "   << cfg.test_ratio
             << "\n";
    }

    Dataset ds = load_movielens(ratings_path, cfg.test_ratio, 42u);

    MFModel model = train_mpi(ds, cfg);

    if (world_rank == 0) {
        cerr << "[MPI] Training finished.\n";
    }

    return model;
}
