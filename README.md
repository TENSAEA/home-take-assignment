# Deep Learning Parallelization: CNN Training on CIFAR-10

A comprehensive implementation comparing serial and parallel training strategies for a Convolutional Neural Network on the CIFAR-10 dataset.

## 📋 Project Overview

This project implements and evaluates:
- **Serial Baseline**: Single-threaded NumPy CNN implementation
- **MPI Parallel**: Data-parallel training using MPI (mpi4py)
- **Hybrid MPI+OpenMP**: Combined MPI data parallelism with OpenMP thread parallelism (via Numba)

## 🏗️ Architecture

```
Input (32×32×3)
    ↓
Conv2D(32 filters, 3×3) + ReLU → MaxPool(2×2)
    ↓
Conv2D(64 filters, 3×3) + ReLU → MaxPool(2×2)
    ↓
Flatten → Dense(256) + ReLU → Dense(10) + Softmax
    ↓
Output (10 classes)
```

## 📦 Requirements

```bash
# Core dependencies
pip install numpy

# For MPI parallelism
pip install mpi4py

# For hybrid OpenMP parallelism
pip install numba
```

**System Requirements:**
- Linux OS (tested on Ubuntu)
- MPI implementation (OpenMPI recommended): `sudo apt install openmpi-bin libopenmpi-dev`
- Python 3.8+

## 🚀 Quick Start

### 1. Clone and Setup
```bash
cd /home/tensu-hiwi/Documents/Projects/home-take-programming
pip install numpy mpi4py numba
```

### 2. Run Serial Baseline
```bash
python serial_cnn.py --epochs 10 --batch-size 32
```

### 3. Run MPI Parallel (4 processes)
```bash
mpirun -np 4 python parallel_cnn_mpi.py --epochs 10 --batch-size 32
```

### 4. Run Hybrid MPI+OpenMP
```bash
export OMP_NUM_THREADS=4
mpirun -np 2 python parallel_cnn_hybrid.py --epochs 10 --batch-size 32
```

### 5. Run Live Demo (for presentations)
```bash
python demo.py
```

## 📊 Running Full Experiments

```bash
# Run all experiments (serial + MPI + hybrid)
python experiments/run_experiments.py --epochs 5 --subset 5000

# Analyze results
python experiments/analyze_results.py --results-dir ./results
```

## 📁 Project Structure

```
home-take-programming/
├── serial_cnn.py              # Serial baseline implementation
├── parallel_cnn_mpi.py        # MPI data-parallel implementation
├── parallel_cnn_hybrid.py     # Hybrid MPI+OpenMP implementation
├── demo.py                    # Live demonstration script
├── utils/
│   ├── layers.py              # CNN layer implementations
│   ├── optimizers.py          # SGD and Adam optimizers
│   ├── data_loader.py         # CIFAR-10 data loading
│   └── metrics.py             # Timing and metrics utilities
├── experiments/
│   ├── run_experiments.py     # Automated experiment runner
│   └── analyze_results.py     # Results analysis and plotting
├── report/
│   ├── report.md              # Technical report (Markdown)
│   └── report.tex             # Technical report (LaTeX source)
└── README.md                  # This file
```

## ⚙️ Command Line Options

All training scripts support:
- `--epochs N`: Number of training epochs (default: 10)
- `--batch-size N`: Batch size (default: 32)
- `--lr F`: Learning rate (default: 0.01)
- `--momentum F`: SGD momentum (default: 0.9)
- `--data-dir PATH`: CIFAR-10 download directory (default: ./data)
- `--save-metrics PATH`: Path to save metrics (default: results/*.npz)
- `--subset N`: Use subset of data for quick testing

## 🔬 Parallelization Strategies

### Data Parallelism (MPI)
- Dataset is partitioned across MPI processes
- Each process computes local gradients
- Gradients synchronized via `MPI_Allreduce`
- All processes maintain identical model weights

### Hybrid MPI+OpenMP
- MPI: Data parallelism across nodes/processes
- OpenMP: Thread-level parallelism for convolutions
- Numba `prange` used for OpenMP-style loops
- Combines inter-process and intra-process parallelism

## 📈 Expected Results

| Configuration | Speedup | Efficiency |
|---------------|---------|------------|
| Serial (1 core) | 1.0x | 100% |
| MPI (2 procs) | ~1.9x | ~95% |
| MPI (4 procs) | ~3.5x | ~88% |
| Hybrid (2P×4T) | ~5.5x | ~69% |

*Results may vary based on hardware and data size.*

## 📝 Technical Report

See [report/report.tex](report/report.tex) for the professional LaTeX source or [report/report.md](report/report.md) for the Markdown version.
The report covers:
- Model architecture and design
- Parallelization approach
- Experimental methodology
- Performance analysis
- Discussion and conclusions

## 🎓 Author

**Tensae Aschalew**
ID: GSR/3976/17
Deep Learning Parallelization Project - Take-Home Assignment

## 📜 License

Educational use only.
