# Parallel Matrix Factorization for Movie Recommendations - Project Explanation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Project Flow](#project-flow)
3. [Matrix Factorization Methodology](#matrix-factorization-methodology)
4. [Parallelization Strategies](#parallelization-strategies)
5. [Processor/Thread Configurations](#processorthread-configurations)
6. [Key Components](#key-components)

---

## 🎯 Project Overview

This project implements **Matrix Factorization (MF)** for collaborative filtering-based movie recommendations using two parallelization approaches:
- **OpenMP**: Shared-memory parallelization (multi-threading on a single machine)
- **MPI**: Distributed-memory parallelization (multi-process across multiple nodes/machines)

The goal is to factorize a sparse user-item rating matrix into two lower-dimensional matrices that capture latent user preferences and item characteristics.

---

## 🔄 Project Flow

### High-Level Execution Flow

```
1. Data Loading (preprocessing.cpp)
   ↓
2. Train/Test Split
   ↓
3. Model Initialization
   ↓
4. Training Loop (OpenMP or MPI)
   ├── SGD Updates
   ├── Synchronization (MPI only)
   └── RMSE Evaluation
   ↓
5. Model Saving
   ↓
6. Inference/Recommendations
```

### Detailed Step-by-Step Flow

#### **Step 1: Entry Point (`main.cpp`)**
- User specifies mode: `omp` or `mpi`
- For MPI: Initializes MPI environment, gets world size and rank
- Calls appropriate training function

#### **Step 2: Data Preprocessing (`preprocessing.cpp`)**
- **Function**: `load_movielens()`
- Reads ratings file (format: `user::item::rating::timestamp` or space-separated)
- Maps user/item IDs to internal indices (0-based)
- Calculates global mean rating
- Splits data into train/test sets (default: 80/20 split)
- Returns `Dataset` structure with:
  - `train`: Training ratings
  - `test`: Test ratings
  - `n_users`, `n_items`: Dimensions
  - `global_mean`: Average rating

#### **Step 3: Model Initialization**
- Creates `MFModel` structure:
  - **U**: User latent factors matrix (n_users × k)
  - **V**: Item latent factors matrix (n_items × k)
  - **bu**: User bias vector (n_users)
  - **bi**: Item bias vector (n_items)
- Initializes U and V with random values from N(0, 0.1²)
- Initializes biases to 0.0

#### **Step 4: Training Loop**

**For OpenMP (`omp_kernel.cpp`):**
```
For each epoch:
  1. Shuffle training data
  2. Parallel loop over ratings (dynamic scheduling)
     - For each rating (u, i, r):
       a. Compute prediction: pred = μ + bu[u] + bi[i] + U[u]·V[i]
       b. Compute error: err = r - pred
       c. Update U[u], V[i], bu[u], bi[i] using SGD
  3. Compute train/test RMSE
  4. Log metrics
```

**For MPI (`mpi_scaling.cpp`):**
```
For each epoch:
  1. Partition data by user ID modulo world_size
  2. Each rank processes its local ratings (with OpenMP)
  3. Synchronize V and bi across all ranks (MPI_Allreduce)
  4. Average synchronized values
  5. Compute global RMSE (aggregate from all ranks)
  6. Log metrics (rank 0 only)
```

#### **Step 5: Model Persistence**
- Saves model to binary file:
  - `outputs/trained_model_omp.bin` (OpenMP)
  - `outputs/trained_model_mpi.bin` (MPI)

#### **Step 6: Inference (`inference.cpp`)**
- Loads saved model
- For a given user, predicts ratings for all items
- Returns top-K recommendations

---

## 🧮 Matrix Factorization Methodology

### Mathematical Foundation

The project implements **Stochastic Gradient Descent (SGD) for Matrix Factorization with Bias Terms**.

#### **Prediction Formula**
```
r̂_ui = μ + bu[u] + bi[i] + U[u]ᵀ · V[i]
```

Where:
- `μ` = global mean rating
- `bu[u]` = user bias (deviation from global mean)
- `bi[i]` = item bias (deviation from global mean)
- `U[u]` = user latent factor vector (k-dimensional)
- `V[i]` = item latent factor vector (k-dimensional)
- `k` = latent dimension (default: 50)

#### **Loss Function**
The algorithm minimizes the regularized squared error:
```
L = Σ(r_ui - r̂_ui)² + λ(||U[u]||² + ||V[i]||² + bu[u]² + bi[i]²)
```

#### **SGD Update Rules**

For each observed rating (u, i, r):

1. **Compute prediction error:**
   ```
   err = r - (μ + bu[u] + bi[i] + U[u]·V[i])
   ```

2. **Update user factors:**
   ```
   U[u][f] ← U[u][f] + η · (err · V[i][f] - λ · U[u][f])
   ```
   For each feature f = 0 to k-1

3. **Update item factors:**
   ```
   V[i][f] ← V[i][f] + η · (err · U[u][f] - λ · V[i][f])
   ```

4. **Update biases:**
   ```
   bu[u] ← bu[u] + η · (err - λ · bu[u])
   bi[i] ← bi[i] + η · (err - λ · bi[i])
   ```

Where:
- `η` (eta) = learning rate (default: 0.01)
- `λ` (lambda) = regularization parameter (default: 0.05)

### Key Algorithmic Features

1. **Bias Terms**: Captures user/item-specific tendencies (some users rate higher, some movies are rated higher)
2. **Regularization**: Prevents overfitting by penalizing large parameter values
3. **Stochastic Updates**: Processes one rating at a time (or in parallel batches)
4. **Shuffling**: Randomizes order each epoch to improve convergence

---

## ⚡ Parallelization Strategies

### 1. OpenMP (Shared-Memory Parallelization)

**Location**: `omp_kernel.cpp`

**Strategy**: 
- Parallelizes the loop over training ratings
- Uses `#pragma omp parallel for schedule(dynamic)`
- Each thread processes a subset of ratings independently

**Key Characteristics**:
- **Shared Memory**: All threads access the same U, V, bu, bi arrays
- **Race Conditions**: Handled by OpenMP's parallel region management
- **Scheduling**: Dynamic scheduling balances workload when ratings per user vary
- **Thread Count**: Controlled by `OMP_NUM_THREADS` environment variable

**Code Structure**:
```cpp
#pragma omp parallel for schedule(dynamic)
for (int idx = 0; idx < train.size(); ++idx) {
    // Process rating and update model
    // All threads update shared U, V, bu, bi
}
```

**Advantages**:
- Simple to implement
- Good for single-node multi-core systems
- Low overhead

**Limitations**:
- Limited to single machine
- Memory bandwidth can become bottleneck

### 2. MPI (Distributed-Memory Parallelization)

**Location**: `mpi_scaling.cpp`

**Strategy**: 
- **Data Partitioning**: Ratings distributed by user ID modulo world_size
  - Rank 0: users 0, 4, 8, 12, ...
  - Rank 1: users 1, 5, 9, 13, ...
  - Rank 2: users 2, 6, 10, 14, ...
  - Rank 3: users 3, 7, 11, 15, ...
- **Hybrid Parallelism**: MPI for inter-process, OpenMP for intra-process
- **Synchronization**: After each epoch, V and bi are synchronized using `MPI_Allreduce`

**Key Characteristics**:
- **Distributed Memory**: Each process has its own copy of model
- **User-Based Partitioning**: Each rank owns a subset of users
- **Synchronization**: Item factors (V, bi) are averaged across all ranks
- **User Factors**: U remains local to each rank (no synchronization needed due to partitioning)

**Synchronization Logic**:
```cpp
// After processing local ratings:
MPI_Allreduce(model.V.data(), V_buffer.data(), size_V, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
MPI_Allreduce(model.bi.data(), bi_buffer.data(), size_bi, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

// Average the results
for (size_t idx = 0; idx < size_V; ++idx) {
    model.V[idx] = V_buffer[idx] / world_size;
}
```

**Why This Partitioning?**
- User factors (U) don't need synchronization because each rank only updates its own users
- Item factors (V) must be synchronized because all ranks update all items
- This minimizes communication overhead

**Advantages**:
- Scales across multiple machines
- Can handle very large datasets
- Hybrid MPI+OpenMP for maximum parallelism

**Limitations**:
- Communication overhead (synchronization cost)
- More complex implementation
- Requires MPI runtime

---

## 🔧 Processor/Thread Configurations

### OpenMP Configuration

**Environment Variables**:
```bash
export OMP_NUM_THREADS=4    # Set number of threads
export OMP_PROC_BIND=true   # Bind threads to cores
export OMP_PLACES=cores     # Place threads on cores
```

**Default Behavior**:
- Uses all available CPU cores if `OMP_NUM_THREADS` not set
- Detected via `omp_get_max_threads()`

**Usage**:
```bash
./parallel_mf omp data/ratings.dat
```

**Monitoring**:
- Training log records: `num_threads` column in `outputs/omp_training_log.csv`
- Visualizations: `outputs/omp_threads_vs_time.png`

### MPI Configuration

**Command-Line Arguments**:
```bash
mpirun -np 4 ./parallel_mf mpi data/ratings.dat
```

Where:
- `-np 4`: Number of MPI processes (world_size = 4)

**Hybrid Configuration (MPI + OpenMP)**:
```bash
export OMP_NUM_THREADS=2
mpirun -np 4 ./parallel_mf mpi data/ratings.dat
```
- Total parallelism: 4 processes × 2 threads = 8 parallel workers

**Process Distribution**:
- Each MPI rank runs on a separate process (potentially different nodes)
- Within each rank, OpenMP threads share memory
- Data partitioned by: `user_id % world_size`

**Monitoring**:
- Training log: `outputs/mpi_training_log_<world_size>.csv`
- Records: `world_size`, `num_threads` per epoch
- Visualizations: 
  - `outputs/mpi_worldsize_vs_time.png`
  - `outputs/mpi_rmse_vs_epochs_ws4.png`

### Configuration Parameters (TrainConfig)

Defined in `mf_common.h`:

```cpp
struct TrainConfig {
    int   k          = 50;      // Latent dimension
    float eta        = 0.01f;    // Learning rate
    float lambda     = 0.05f;    // Regularization
    int   epochs     = 20;       // Training epochs
    float test_ratio = 0.2f;     // Test set fraction (20%)
};
```

**Tuning Guidelines**:
- **k**: Higher = more expressive, but slower and may overfit (typical: 20-100)
- **eta**: Higher = faster learning, but may diverge (typical: 0.001-0.1)
- **lambda**: Higher = more regularization, prevents overfitting (typical: 0.01-0.1)
- **epochs**: More = better convergence, but diminishing returns (typical: 10-50)

---

## 📦 Key Components

### Data Structures

1. **`Rating`**: `{u, i, r}` - user ID, item ID, rating value
2. **`Dataset`**: Contains train/test splits, dimensions, global mean
3. **`MFModel`**: Contains factor matrices (U, V) and biases (bu, bi)
4. **`TrainConfig`**: Hyperparameters

### Core Functions

1. **`load_movielens()`**: Data loading and preprocessing
2. **`train_omp()`**: OpenMP training loop
3. **`train_mpi()`**: MPI training loop
4. **`predict_rating()`**: Compute prediction for (user, item)
5. **`compute_rmse()`**: Calculate Root Mean Squared Error
6. **`recommend_top_k_for_user()`**: Generate top-K recommendations
7. **`save_model()` / `load_model()`**: Model persistence

### File Organization

```
src/
├── main.cpp          # Entry point, mode selection
├── mf_common.h       # Data structures, function declarations
├── preprocessing.cpp # Data loading
├── training.cpp      # Wrapper functions (run_training_omp/mpi)
├── omp_kernel.cpp    # OpenMP implementation
├── mpi_scaling.cpp   # MPI implementation
└── inference.cpp     # Recommendation generation
```

---

## 📊 Performance Metrics

The project tracks:
- **Train RMSE**: Error on training set (should decrease)
- **Test RMSE**: Error on test set (should decrease, indicates generalization)
- **Epoch Time**: Time per training epoch
- **Thread/Process Count**: Parallelism configuration

All metrics logged to CSV files for analysis and visualization.

---

## 🎓 Summary

This project demonstrates:
1. **Matrix Factorization** using SGD with bias terms
2. **Two parallelization paradigms**: OpenMP (shared-memory) and MPI (distributed)
3. **Hybrid parallelism**: MPI+OpenMP for maximum scalability
4. **Practical application**: Movie recommendation system

The choice between OpenMP and MPI depends on:
- **OpenMP**: Single machine, multi-core CPU
- **MPI**: Multiple machines, very large datasets, cluster environments
