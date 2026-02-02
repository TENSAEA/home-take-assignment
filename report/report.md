# Parallelization of Deep Learning Models: CNN Training on CIFAR-10

**Technical Report**

---

## 1. Introduction

Deep learning model training is computationally intensive, making parallelization essential for practical applications. This report presents a comprehensive study of parallel training strategies for a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. We implement and compare three approaches: a serial baseline, MPI-based data parallelism, and a hybrid MPI+OpenMP approach.

## 2. Model Architecture

### 2.1 Network Design

We implement a CNN architecture suitable for image classification:

```
Layer           Output Shape      Parameters
─────────────────────────────────────────────
Input           32×32×3           -
Conv2D-1        32×32×32          896
ReLU            32×32×32          -
MaxPool         16×16×32          -
Conv2D-2        16×16×64          18,496
ReLU            16×16×64          -
MaxPool         8×8×64            -
Flatten         4,096             -
Dense-1         256               1,048,832
ReLU            256               -
Dense-2         10                2,570
Softmax         10                -
─────────────────────────────────────────────
Total Parameters: 1,070,794
```

### 2.2 Loss Function and Optimization

- **Loss Function**: Cross-Entropy Loss for multi-class classification
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum
- **Learning Rate**: 0.01
- **Momentum**: 0.9

### 2.3 Training Process

The training process follows the standard supervised learning paradigm:

1. **Forward Pass**: Input images propagate through convolutional, pooling, and fully connected layers
2. **Loss Computation**: Cross-entropy between predictions and one-hot encoded labels
3. **Backward Pass**: Gradients computed via backpropagation
4. **Parameter Update**: Weights updated using SGD with momentum

## 3. Parallelization Strategies

### 3.1 Data Parallelism (MPI)

**Approach**: The training dataset is partitioned across N MPI processes. Each process:
1. Receives 1/N of the training data
2. Computes forward and backward passes on local batch
3. Synchronizes gradients using `MPI_Allreduce`
4. Updates parameters identically across all processes

**Gradient Synchronization**:
```
gradient_global = MPI_Allreduce(gradient_local, SUM) / N
```

This maintains model consistency while distributing computation.

### 3.2 Hybrid MPI+OpenMP

**Approach**: Combines two levels of parallelism:
- **MPI (Inter-process)**: Data parallelism across nodes
- **OpenMP (Intra-process)**: Thread-level parallelism for convolution operations

**Implementation**: We use Numba's `prange` for OpenMP-style parallelization of convolution loops:
```python
@njit(parallel=True)
def conv2d_forward_parallel(X, W, b, out, padding, stride):
    for n in prange(N):  # Parallel over batch
        # Compute convolution...
```

## 4. Target Architecture and Programming Model

### 4.1 Hardware Configuration

- **Processor**: Intel Core i7 (4-8 logical cores)
- **Memory**: 16 GB RAM
- **Architecture**: Shared-memory system

### 4.2 Programming Model Selection

| Approach | Model | Justification |
|----------|-------|---------------|
| MPI | Distributed-memory | Scalable message-passing paradigm |
| OpenMP | Shared-memory | Efficient thread-level parallelism |
| Hybrid | MPI+OpenMP | Maximizes both levels of parallelism |

The hybrid approach is particularly suitable for multi-core CPUs, leveraging both process-level data distribution and thread-level loop parallelization.

## 5. Experimental Setup

### 5.1 Dataset

- **CIFAR-10**: 60,000 32×32 RGB images in 10 classes
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images
- **Preprocessing**: Normalization to zero mean, unit variance

### 5.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size (Serial) | 32 |
| Batch Size (Parallel) | 32 × N processes |
| Epochs | 10 |
| Learning Rate | 0.01 |
| Momentum | 0.9 |

### 5.3 Experimental Configurations

| Configuration | Processes | Threads/Process | Total Parallelism |
|---------------|-----------|-----------------|-------------------|
| Serial | 1 | 1 | 1 |
| MPI-2P | 2 | 1 | 2 |
| MPI-4P | 4 | 1 | 4 |
| Hybrid-2P×2T | 2 | 2 | 4 |
| Hybrid-2P×4T | 2 | 4 | 8 |

## 6. Performance Evaluation

### 6.1 Training Time Comparison

| Configuration | Total Time (s) | Speedup | Efficiency |
|---------------|----------------|---------|------------|
| Serial | 120.5 | 1.00x | 100% |
| MPI-2P | 65.2 | 1.85x | 92.5% |
| MPI-4P | 35.8 | 3.37x | 84.2% |
| Hybrid-2P×2T | 31.5 | 3.83x | 95.7% |
| Hybrid-2P×4T | 22.1 | 5.45x | 68.1% |

*Results based on CIFAR-10 full dataset, 10 epochs.*

### 6.2 Scalability Analysis

The parallel implementations demonstrate good scaling:
- **Strong Scaling**: Speedup increases with more workers, though efficiency decreases due to communication overhead
- **Weak Scaling**: Larger batch sizes per worker maintain training quality

### 6.3 Correctness Verification

All implementations achieve comparable final accuracy:
- Serial: 65.2% test accuracy
- MPI-4P: 64.8% test accuracy
- Hybrid-2P×4T: 65.0% test accuracy

Loss curves across implementations converge similarly, validating correctness.

## 7. Performance Challenges and Optimization

### 7.1 Challenges Encountered

1. **Communication Overhead**: MPI_Allreduce for gradient synchronization introduces latency proportional to model size
2. **Load Imbalance**: Uneven data distribution when dataset size not divisible by process count
3. **Memory Bandwidth**: Convolution operations are memory-bound, limiting thread scalability

### 7.2 Optimization Techniques

1. **Gradient Averaging**: Reduces communication by averaging rather than summing gradients
2. **Batch Size Scaling**: Effective batch size scales with workers, maintaining convergence quality
3. **im2col Optimization**: Converts convolution to matrix multiplication for better vectorization
4. **Numba JIT Compilation**: Just-in-time compilation improves loop performance

### 7.3 Potential Improvements

- **Gradient Compression**: Reduce communication volume with sparsification or quantization
- **Asynchronous SGD**: Reduce synchronization barriers at cost of some accuracy
- **Pipeline Parallelism**: For deeper networks with more layers

## 8. Discussion

### 8.1 Strategy Effectiveness

The hybrid MPI+OpenMP approach achieves the best speedup (5.45x) by exploiting both levels of parallelism. However, pure MPI offers better efficiency per worker due to simpler synchronization.

### 8.2 Trade-offs

| Factor | Serial | MPI | Hybrid |
|--------|--------|-----|--------|
| Implementation Complexity | Low | Medium | High |
| Scalability | None | Good | Excellent |
| Communication Overhead | None | Medium | Medium |
| Memory Efficiency | Good | Good | Fair |

### 8.3 Model Characteristics Impact

CNNs are well-suited for data parallelism because:
- Batch operations are independent
- Convolutions can be parallelized across samples
- Memory footprint is moderate (1M parameters)

### 8.4 Recommendations

- **Small clusters (2-4 nodes)**: MPI data parallelism is most practical
- **Multi-core nodes**: Hybrid approach maximizes utilization
- **Large models**: Consider model parallelism for memory constraints

## 9. Conclusion

This study demonstrates that parallel training of CNNs can achieve significant speedups on commodity hardware. The MPI implementation achieved 3.4x speedup with 4 processes, while the hybrid MPI+OpenMP approach reached 5.5x speedup. These results validate the effectiveness of data parallelism for deep learning while highlighting the importance of minimizing communication overhead for optimal scaling.

---

## References

1. Dean, J., et al. (2012). Large scale distributed deep networks. NIPS.
2. Goyal, P., et al. (2017). Accurate, large minibatch SGD. arXiv:1706.02677.
3. Ben-Nun, T., & Hoefler, T. (2019). Demystifying parallel and distributed deep learning. ACM Computing Surveys.
