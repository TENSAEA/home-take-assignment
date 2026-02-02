#!/usr/bin/env python3
"""
MPI Parallel CNN Implementation for CIFAR-10 Classification
============================================================
Data-parallel training using MPI for gradient synchronization.

Parallelization Strategy:
    - Data is partitioned across MPI processes
    - Each process computes gradients on its local batch
    - Gradients are averaged using MPI AllReduce
    - Parameters are synchronized after each batch update

Usage:
    mpirun -np 4 python parallel_cnn_mpi.py --epochs 10

Author: Deep Learning Parallelization Project
"""

import argparse
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, Softmax, CrossEntropyLoss
from utils.optimizers import SGD
from utils.data_loader import load_cifar10, one_hot_encode, DataLoader
from utils.metrics import Timer, TrainingMetrics, compute_accuracy

# MPI imports
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available. Running in serial mode.")


class ParallelCNN:
    """
    CNN with MPI support for data-parallel training.
    Gradients are synchronized across processes using AllReduce.
    """
    
    def __init__(self, comm=None):
        """Initialize CNN layers with optional MPI communicator."""
        self.comm = comm
        if comm is not None:
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        else:
            self.rank = 0
            self.size = 1
        
        # Set unique random seed per rank for different weight initialization
        # But then broadcast from rank 0 to ensure identical initial weights
        np.random.seed(42)
        
        # Layer 1: Conv(3->32) + ReLU + MaxPool
        self.conv1 = Conv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        # Layer 2: Conv(32->64) + ReLU + MaxPool
        self.conv2 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        
        # Flatten
        self.flatten = Flatten()
        
        # Fully connected layers
        self.fc1 = Dense(in_features=8*8*64, out_features=256)
        self.relu3 = ReLU()
        self.fc2 = Dense(in_features=256, out_features=10)
        
        # Softmax and loss
        self.softmax = Softmax()
        self.loss_fn = CrossEntropyLoss()
        
        # Trainable layers
        self.trainable_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        
        # Synchronize initial weights across all processes
        self._sync_initial_params()
    
    def _sync_initial_params(self):
        """Broadcast initial parameters from rank 0 to all processes."""
        if self.comm is None:
            return
        
        for layer in self.trainable_layers:
            params = layer.get_params()
            for key in params:
                if self.rank == 0:
                    param_data = params[key]
                else:
                    param_data = np.empty_like(params[key])
                
                self.comm.Bcast(param_data, root=0)
                params[key] = param_data
            layer.set_params(params)
    
    def forward(self, X):
        """Forward pass through the network."""
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
        
        probs = self.softmax.forward(out)
        return probs
    
    def backward(self, y_pred, y_true):
        """Backward pass - compute local gradients."""
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
        """
        Synchronize gradients across all MPI processes using AllReduce.
        Gradients are averaged to maintain consistent updates.
        """
        if self.comm is None:
            return
        
        for layer in self.trainable_layers:
            grads = layer.get_grads()
            
            for key in grads:
                # AllReduce sum, then divide by number of processes
                grad_sum = np.zeros_like(grads[key])
                self.comm.Allreduce(grads[key], grad_sum, op=MPI.SUM)
                grads[key][:] = grad_sum / self.size
            
            # Update layer gradients
            if hasattr(layer, 'dW'):
                layer.dW = grads['W']
            if hasattr(layer, 'db'):
                layer.db = grads['b']
    
    def compute_loss(self, y_pred, y_true):
        """Compute cross-entropy loss."""
        return self.loss_fn.forward(y_pred, y_true)


def partition_data(X, y, rank, size):
    """
    Partition data across MPI processes.
    Each process gets a contiguous chunk of the data.
    """
    n_samples = X.shape[0]
    samples_per_rank = n_samples // size
    
    start_idx = rank * samples_per_rank
    if rank == size - 1:
        # Last rank gets remaining samples
        end_idx = n_samples
    else:
        end_idx = start_idx + samples_per_rank
    
    return X[start_idx:end_idx], y[start_idx:end_idx]


def train_epoch_parallel(model, X_train, y_train, optimizer, batch_size, epoch, verbose=True):
    """Train for one epoch with MPI synchronization."""
    comm = model.comm
    rank = model.rank
    size = model.size
    
    # Shuffle data (same shuffle across all ranks for consistency)
    np.random.seed(epoch)  # Same seed ensures consistent shuffle
    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Partition data for this rank
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
        
        # Forward pass
        y_pred = model.forward(X_batch)
        
        # Compute local loss
        loss = model.compute_loss(y_pred, y_onehot)
        
        # Backward pass (compute local gradients)
        model.backward(y_pred, y_onehot)
        
        # Synchronize gradients across processes
        model.sync_gradients()
        
        # Update parameters (all ranks update identically)
        optimizer.step(model.trainable_layers)
        
        # Track local metrics
        total_loss += loss * X_batch.shape[0]
        total_correct += np.sum(np.argmax(y_pred, axis=1) == y_batch)
        total_samples += X_batch.shape[0]
    
    # Aggregate metrics across all ranks
    if comm is not None:
        global_loss = comm.allreduce(total_loss, op=MPI.SUM)
        global_correct = comm.allreduce(total_correct, op=MPI.SUM)
        global_samples = comm.allreduce(total_samples, op=MPI.SUM)
    else:
        global_loss = total_loss
        global_correct = total_correct
        global_samples = total_samples
    
    avg_loss = global_loss / global_samples
    accuracy = global_correct / global_samples
    
    return avg_loss, accuracy


def evaluate_parallel(model, X_test, y_test, batch_size):
    """Evaluate model on test set (only rank 0 evaluates)."""
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
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train_parallel(model, X_train, y_train, X_test, y_test, optimizer, epochs, batch_size, verbose=True):
    """Full parallel training loop."""
    comm = model.comm
    rank = model.rank
    size = model.size
    
    metrics = TrainingMetrics()
    
    if verbose and rank == 0:
        print("\n" + "="*60)
        print("MPI PARALLEL CNN TRAINING - CIFAR-10")
        print("="*60)
        print(f"Number of MPI processes: {size}")
        print(f"Total training samples: {X_train.shape[0]}")
        print(f"Samples per process: ~{X_train.shape[0] // size}")
        print(f"Batch size per process: {batch_size}")
        print(f"Effective batch size: {batch_size * size}")
        print(f"Epochs: {epochs}")
        print("="*60)
    
    for epoch in range(1, epochs + 1):
        if comm is not None:
            comm.Barrier()  # Sync before timing
        
        timer = Timer()
        timer.start()
        
        if verbose and rank == 0:
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch_parallel(
            model, X_train, y_train, optimizer, batch_size, epoch, verbose
        )
        
        # Evaluate (rank 0 only)
        test_loss, test_acc = evaluate_parallel(model, X_test, y_test, batch_size)
        
        # Broadcast test metrics to all ranks
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
    parser = argparse.ArgumentParser(description='MPI Parallel CNN Training on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per process')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save-metrics', type=str, default='results/mpi_metrics.npz',
                        help='Path to save metrics')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use subset of training data (for quick testing)')
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
    
    # Load data (all ranks load full data, then partition)
    if rank == 0:
        print("Loading CIFAR-10 dataset...")
    
    X_train, y_train, X_test, y_test = load_cifar10(args.data_dir)
    
    # Use subset if specified
    if args.subset:
        X_train = X_train[:args.subset]
        y_train = y_train[:args.subset]
        X_test = X_test[:args.subset // 5]
        y_test = y_test[:args.subset // 5]
    
    if rank == 0:
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
    
    # Initialize model and optimizer
    if rank == 0:
        print(f"\nInitializing Parallel CNN model with {size} MPI processes...")
    
    model = ParallelCNN(comm=comm)
    optimizer = SGD(learning_rate=args.lr, momentum=args.momentum)
    
    # Train
    metrics = train_parallel(
        model, X_train, y_train, X_test, y_test,
        optimizer, args.epochs, args.batch_size
    )
    
    # Print summary and save metrics (rank 0 only)
    if rank == 0:
        metrics.print_summary()
        
        os.makedirs(os.path.dirname(args.save_metrics), exist_ok=True)
        metrics.save(args.save_metrics)
        print(f"\nMetrics saved to {args.save_metrics}")


if __name__ == '__main__':
    main()
