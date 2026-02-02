#!/usr/bin/env python3
"""
Hybrid MPI+OpenMP Parallel CNN Implementation for CIFAR-10
==========================================================
Uses MPI for inter-process data parallelism and OpenMP (via Numba)
for intra-process parallelization of convolution operations.

Parallelization Strategy:
    - MPI: Data parallelism across processes (gradient synchronization)
    - OpenMP: Thread-level parallelism for convolution loops within each process

Usage:
    export OMP_NUM_THREADS=4
    mpirun -np 2 python parallel_cnn_hybrid.py --epochs 10

Author: Deep Learning Parallelization Project
"""

import argparse
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.optimizers import SGD
from utils.data_loader import load_cifar10, one_hot_encode
from utils.metrics import Timer, TrainingMetrics

# MPI imports
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available. Running without MPI.")

# Numba for OpenMP-like parallelism
try:
    from numba import njit, prange, set_num_threads, get_num_threads
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available. Running without OpenMP parallelism.")


# =============================================================================
# OpenMP-Parallelized Operations (using Numba prange)
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def conv2d_forward_parallel(X, weights_arr, b, out, padding, stride):
        """
        Parallelized 2D convolution forward pass.
        Uses OpenMP-style parallelism via Numba's prange.
        
        X: (N, H, W_dim, C_in)
        weights_arr: (C_out, C_in, kH, kW)
        b: (C_out,)
        out: (N, H_out, W_out, C_out)
        """
        N, H, W_dim, C_in = X.shape
        C_out, _, kH, kW = weights_arr.shape
        _, H_out, W_out, _ = out.shape
        
        # Parallel over batch dimension
        for n in prange(N):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride - padding
                        w_start = w_out * stride - padding
                        
                        val = b[c_out]
                        
                        for c_in in range(C_in):
                            for kh in range(kH):
                                for kw in range(kW):
                                    h_in = h_start + kh
                                    w_in = w_start + kw
                                    
                                    if 0 <= h_in < H and 0 <= w_in < W_dim:
                                        val += X[n, h_in, w_in, c_in] * weights_arr[c_out, c_in, kh, kw]
                        
                        out[n, h_out, w_out, c_out] = val
        
        return out
    
    @njit(parallel=True, cache=True)
    def conv2d_backward_parallel(dout, X, weights_arr, dW, db, dX, padding, stride):
        """
        Parallelized 2D convolution backward pass.
        
        dout: (N, H_out, W_out, C_out)
        X: (N, H, W_dim, C_in)
        weights_arr: (C_out, C_in, kH, kW)
        """
        N, H, W_dim, C_in = X.shape
        C_out, _, kH, kW = weights_arr.shape
        _, H_out, W_out, _ = dout.shape
        
        # Compute dW (parallel over output channels)
        for c_out in prange(C_out):
            for c_in in range(C_in):
                for kh in range(kH):
                    for kw in range(kW):
                        grad_sum = 0.0
                        for n in range(N):
                            for h_out in range(H_out):
                                for w_out in range(W_out):
                                    h_in = h_out * stride - padding + kh
                                    w_in = w_out * stride - padding + kw
                                    
                                    if 0 <= h_in < H and 0 <= w_in < W_dim:
                                        grad_sum += dout[n, h_out, w_out, c_out] * X[n, h_in, w_in, c_in]
                        
                        dW[c_out, c_in, kh, kw] = grad_sum
        
        # Compute db
        for c_out in prange(C_out):
            grad_sum = 0.0
            for n in range(N):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        grad_sum += dout[n, h_out, w_out, c_out]
            db[c_out] = grad_sum
        
        # Compute dX (parallel over batch)
        for n in prange(N):
            for h in range(H):
                for w in range(W_dim):
                    for c_in in range(C_in):
                        grad_sum = 0.0
                        for c_out in range(C_out):
                            for kh in range(kH):
                                for kw in range(kW):
                                    h_out = (h + padding - kh)
                                    w_out = (w + padding - kw)
                                    
                                    if h_out % stride == 0 and w_out % stride == 0:
                                        h_out = h_out // stride
                                        w_out = w_out // stride
                                        
                                        if 0 <= h_out < H_out and 0 <= w_out < W_out:
                                            grad_sum += dout[n, h_out, w_out, c_out] * weights_arr[c_out, c_in, kh, kw]
                        
                        dX[n, h, w, c_in] = grad_sum
        
        return dW, db, dX


# =============================================================================
# Hybrid Parallel Layers
# =============================================================================

class HybridConv2D:
    """
    2D Convolution Layer with OpenMP parallelization via Numba.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float64) * scale
        self.b = np.zeros(out_channels, dtype=np.float64)
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache = None
    
    def forward(self, X):
        """OpenMP-parallelized forward pass."""
        N, H, W, C = X.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        out = np.zeros((N, H_out, W_out, self.out_channels), dtype=np.float64)
        
        if NUMBA_AVAILABLE:
            X_cont = np.ascontiguousarray(X.astype(np.float64))
            conv2d_forward_parallel(X_cont, self.W, self.b, out, self.padding, self.stride)
        else:
            # Fallback to naive implementation
            out = self._forward_naive(X)
        
        self.cache = X.astype(np.float64) if not NUMBA_AVAILABLE else X_cont
        return out
    
    def _forward_naive(self, X):
        """Naive forward pass fallback."""
        N, H, W, C = X.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Pad input
        if self.padding > 0:
            X_padded = np.pad(X, ((0,0), (self.padding, self.padding), 
                                  (self.padding, self.padding), (0,0)), mode='constant')
        else:
            X_padded = X
        
        out = np.zeros((N, H_out, W_out, self.out_channels))
        
        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        window = X_padded[n, h_start:h_start+self.kernel_size, 
                                         w_start:w_start+self.kernel_size, :]
                        out[n, h, w, c_out] = np.sum(window * self.W[c_out].transpose(1, 2, 0)) + self.b[c_out]
        
        return out
    
    def backward(self, dout):
        """OpenMP-parallelized backward pass."""
        X = self.cache
        N, H, W, C = X.shape
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        dX = np.zeros_like(X)
        
        if NUMBA_AVAILABLE:
            dout_cont = np.ascontiguousarray(dout.astype(np.float64))
            conv2d_backward_parallel(dout_cont, X, self.W, self.dW, self.db, dX, 
                                     self.padding, self.stride)
        else:
            dX = self._backward_naive(dout)
        
        return dX
    
    def _backward_naive(self, dout):
        """Naive backward pass fallback."""
        X = self.cache
        N, H, W, C_in = X.shape
        _, H_out, W_out, C_out = dout.shape
        
        # Pad for backward
        if self.padding > 0:
            X_padded = np.pad(X, ((0,0), (self.padding, self.padding),
                                  (self.padding, self.padding), (0,0)), mode='constant')
        else:
            X_padded = X
        
        dX_padded = np.zeros_like(X_padded)
        self.dW = np.zeros_like(self.W)
        self.db = np.sum(dout, axis=(0, 1, 2))
        
        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        self.dW[c_out] += (X_padded[n, h_start:h_start+self.kernel_size,
                                                    w_start:w_start+self.kernel_size, :].transpose(2, 0, 1) 
                                          * dout[n, h, w, c_out])
                        
                        dX_padded[n, h_start:h_start+self.kernel_size,
                                  w_start:w_start+self.kernel_size, :] += (
                            self.W[c_out].transpose(1, 2, 0) * dout[n, h, w, c_out]
                        )
        
        if self.padding > 0:
            dX = dX_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX = dX_padded
        
        return dX
    
    def get_params(self):
        return {'W': self.W, 'b': self.b}
    
    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']
    
    def get_grads(self):
        return {'W': self.dW, 'b': self.db}


class MaxPool2D:
    """Max Pooling Layer."""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
    
    def forward(self, X):
        N, H, W, C = X.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        
        out = np.zeros((N, H_out, W_out, C))
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                window = X[:, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size, :]
                out[:, i, j, :] = np.max(window, axis=(1, 2))
        
        self.cache = X
        return out
    
    def backward(self, dout):
        X = self.cache
        N, H, W, C = X.shape
        _, H_out, W_out, _ = dout.shape
        
        dX = np.zeros_like(X)
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                window = X[:, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size, :]
                
                for n in range(N):
                    for c in range(C):
                        win = window[n, :, :, c]
                        max_idx = np.unravel_index(np.argmax(win), win.shape)
                        dX[n, h_start + max_idx[0], w_start + max_idx[1], c] += dout[n, i, j, c]
        
        return dX


class Dense:
    """Fully Connected Layer."""
    
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features).astype(np.float64) * scale
        self.b = np.zeros((1, out_features), dtype=np.float64)
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache = None
    
    def forward(self, X):
        self.cache = X
        return X @ self.W + self.b
    
    def backward(self, dout):
        X = self.cache
        self.dW = X.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.W.T
    
    def get_params(self):
        return {'W': self.W, 'b': self.b}
    
    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']
    
    def get_grads(self):
        return {'W': self.dW, 'b': self.db}


class Flatten:
    def __init__(self):
        self.shape = None
    
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.shape)


class ReLU:
    def __init__(self):
        self.cache = None
    
    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)
    
    def backward(self, dout):
        return dout * (self.cache > 0)


class Softmax:
    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)


class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        N = y_pred.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / N
    
    def backward(self, y_pred, y_true):
        N = y_pred.shape[0]
        return (y_pred - y_true) / N


# =============================================================================
# Hybrid Parallel CNN
# =============================================================================

class HybridParallelCNN:
    """
    CNN with hybrid MPI+OpenMP parallelism.
    - MPI: Data parallelism across processes
    - OpenMP: Thread-level parallelism for convolutions
    """
    
    def __init__(self, comm=None, num_threads=None):
        self.comm = comm
        if comm is not None:
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        else:
            self.rank = 0
            self.size = 1
        
        # Set OpenMP threads
        if num_threads and NUMBA_AVAILABLE:
            set_num_threads(num_threads)
            if self.rank == 0:
                print(f"OpenMP threads per process: {get_num_threads()}")
        
        np.random.seed(42)
        
        # Layers with OpenMP-parallelized convolutions
        self.conv1 = HybridConv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        self.conv2 = HybridConv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        
        self.flatten = Flatten()
        self.fc1 = Dense(in_features=8*8*64, out_features=256)
        self.relu3 = ReLU()
        self.fc2 = Dense(in_features=256, out_features=10)
        
        self.softmax = Softmax()
        self.loss_fn = CrossEntropyLoss()
        
        self.trainable_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        
        self._sync_initial_params()
    
    def _sync_initial_params(self):
        """Broadcast initial parameters from rank 0."""
        if self.comm is None:
            return
        
        for layer in self.trainable_layers:
            params = layer.get_params()
            for key in params:
                if self.rank == 0:
                    param_data = np.ascontiguousarray(params[key])
                else:
                    param_data = np.empty_like(params[key])
                
                self.comm.Bcast(param_data, root=0)
                params[key] = param_data
            layer.set_params(params)
    
    def forward(self, X):
        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        
        out = self.flatten.forward(out)
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        out = self.fc2.forward(out)
        
        return self.softmax.forward(out)
    
    def backward(self, y_pred, y_true):
        dout = self.loss_fn.backward(y_pred, y_true)
        
        dout = self.fc2.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.fc1.backward(dout)
        
        dout = self.flatten.backward(dout)
        
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)
        
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)
    
    def sync_gradients(self):
        """Synchronize gradients across MPI processes."""
        if self.comm is None:
            return
        
        for layer in self.trainable_layers:
            grads = layer.get_grads()
            
            for key in grads:
                grad_sum = np.zeros_like(grads[key])
                self.comm.Allreduce(np.ascontiguousarray(grads[key]), grad_sum, op=MPI.SUM)
                grads[key][:] = grad_sum / self.size
            
            if hasattr(layer, 'dW'):
                layer.dW = grads['W']
            if hasattr(layer, 'db'):
                layer.db = grads['b']
    
    def compute_loss(self, y_pred, y_true):
        return self.loss_fn.forward(y_pred, y_true)


def partition_data(X, y, rank, size):
    """Partition data across MPI processes."""
    n_samples = X.shape[0]
    samples_per_rank = n_samples // size
    
    start_idx = rank * samples_per_rank
    end_idx = n_samples if rank == size - 1 else start_idx + samples_per_rank
    
    return X[start_idx:end_idx], y[start_idx:end_idx]


def train_epoch_hybrid(model, X_train, y_train, optimizer, batch_size, epoch):
    """Train for one epoch with hybrid parallelism."""
    comm = model.comm
    rank = model.rank
    size = model.size
    
    np.random.seed(epoch)
    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    X_local, y_local = partition_data(X_shuffled, y_shuffled, rank, size)
    
    n_local = X_local.shape[0]
    n_batches = (n_local + batch_size - 1) // batch_size
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_local)
        
        X_batch = X_local[start:end]
        y_batch = y_local[start:end]
        y_onehot = one_hot_encode(y_batch, num_classes=10)
        
        y_pred = model.forward(X_batch)
        loss = model.compute_loss(y_pred, y_onehot)
        
        model.backward(y_pred, y_onehot)
        model.sync_gradients()
        optimizer.step(model.trainable_layers)
        
        total_loss += loss * X_batch.shape[0]
        total_correct += np.sum(np.argmax(y_pred, axis=1) == y_batch)
        total_samples += X_batch.shape[0]
    
    if comm is not None:
        global_loss = comm.allreduce(total_loss, op=MPI.SUM)
        global_correct = comm.allreduce(total_correct, op=MPI.SUM)
        global_samples = comm.allreduce(total_samples, op=MPI.SUM)
    else:
        global_loss = total_loss
        global_correct = total_correct
        global_samples = total_samples
    
    return global_loss / global_samples, global_correct / global_samples


def evaluate_hybrid(model, X_test, y_test, batch_size):
    """Evaluate model on test set."""
    if model.rank != 0:
        return 0, 0
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    n_batches = (X_test.shape[0] + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, X_test.shape[0])
        
        X_batch = X_test[start:end]
        y_batch = y_test[start:end]
        y_onehot = one_hot_encode(y_batch, num_classes=10)
        
        y_pred = model.forward(X_batch)
        loss = model.compute_loss(y_pred, y_onehot)
        
        total_loss += loss * X_batch.shape[0]
        total_correct += np.sum(np.argmax(y_pred, axis=1) == y_batch)
        total_samples += X_batch.shape[0]
    
    return total_loss / total_samples, total_correct / total_samples


def train_hybrid(model, X_train, y_train, X_test, y_test, optimizer, epochs, batch_size, verbose=True):
    """Full hybrid parallel training loop."""
    comm = model.comm
    rank = model.rank
    size = model.size
    
    metrics = TrainingMetrics()
    
    if verbose and rank == 0:
        print("\n" + "="*60)
        print("HYBRID MPI+OPENMP PARALLEL CNN TRAINING - CIFAR-10")
        print("="*60)
        print(f"MPI processes: {size}")
        if NUMBA_AVAILABLE:
            print(f"OpenMP threads per process: {get_num_threads()}")
        print(f"Total parallelism: {size * (get_num_threads() if NUMBA_AVAILABLE else 1)} workers")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Batch size per process: {batch_size}")
        print(f"Epochs: {epochs}")
        print("="*60)
    
    for epoch in range(1, epochs + 1):
        if comm is not None:
            comm.Barrier()
        
        timer = Timer()
        timer.start()
        
        if verbose and rank == 0:
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)
        
        train_loss, train_acc = train_epoch_hybrid(
            model, X_train, y_train, optimizer, batch_size, epoch
        )
        
        test_loss, test_acc = evaluate_hybrid(model, X_test, y_test, batch_size)
        
        if comm is not None:
            test_loss = comm.bcast(test_loss, root=0)
            test_acc = comm.bcast(test_acc, root=0)
        
        epoch_time = timer.stop()
        
        if rank == 0:
            metrics.add_epoch(train_loss, train_acc, test_loss, test_acc, epoch_time)
            
            if verbose:
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
                print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")
                print(f"  Epoch Time: {epoch_time:.2f}s")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Hybrid MPI+OpenMP CNN Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per process')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--num-threads', type=int, default=None, help='OpenMP threads per process')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save-metrics', type=str, default='results/hybrid_metrics.npz',
                        help='Path to save metrics')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use subset of training data')
    args = parser.parse_args()
    
    # Initialize MPI
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1
    
    if rank == 0:
        print("Loading CIFAR-10 dataset...")
    
    X_train, y_train, X_test, y_test = load_cifar10(args.data_dir)
    
    if args.subset:
        X_train = X_train[:args.subset]
        y_train = y_train[:args.subset]
        X_test = X_test[:args.subset // 5]
        y_test = y_test[:args.subset // 5]
    
    if rank == 0:
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"\nInitializing Hybrid CNN with {size} MPI processes...")
    
    model = HybridParallelCNN(comm=comm, num_threads=args.num_threads)
    optimizer = SGD(learning_rate=args.lr, momentum=args.momentum)
    
    metrics = train_hybrid(
        model, X_train, y_train, X_test, y_test,
        optimizer, args.epochs, args.batch_size
    )
    
    if rank == 0:
        metrics.print_summary()
        
        os.makedirs(os.path.dirname(args.save_metrics), exist_ok=True)
        metrics.save(args.save_metrics)
        print(f"\nMetrics saved to {args.save_metrics}")


if __name__ == '__main__':
    main()
