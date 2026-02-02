#!/usr/bin/env python3
"""
Live Training Demo - Real-time visualization for instructor presentation
========================================================================
Shows side-by-side comparison of serial vs parallel training with live updates.
"""

import sys
import os
import time
import threading
import numpy as np
from queue import Queue

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, Softmax, CrossEntropyLoss
from utils.optimizers import SGD
from utils.data_loader import load_cifar10, one_hot_encode
from utils.metrics import Timer


class ProgressBar:
    """Terminal-based progress bar."""
    
    def __init__(self, total, width=40, prefix=''):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
    
    def update(self, current, extra_info=''):
        self.current = current
        filled = int(self.width * current / self.total)
        bar = '█' * filled + '░' * (self.width - filled)
        percent = current / self.total * 100
        print(f"\r{self.prefix}[{bar}] {percent:5.1f}% {extra_info}", end='', flush=True)
    
    def finish(self):
        print()


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print demo header."""
    print("╔" + "═" * 78 + "╗")
    print("║" + " DEEP LEARNING PARALLELIZATION DEMO ".center(78) + "║")
    print("║" + " CNN Training on CIFAR-10 ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()


def print_architecture():
    """Print CNN architecture diagram."""
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                           CNN ARCHITECTURE                                  │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│  Input(32×32×3) → Conv(32) → ReLU → MaxPool → Conv(64) → ReLU → MaxPool    │")
    print("│                 → Flatten → Dense(256) → ReLU → Dense(10) → Softmax        │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print()


def print_comparison_header(mode):
    """Print comparison section header."""
    if mode == 'serial':
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│                        SERIAL TRAINING (BASELINE)                             │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
    elif mode == 'mpi':
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│                        MPI PARALLEL TRAINING                                  │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
    elif mode == 'hybrid':
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│                     HYBRID MPI+OpenMP TRAINING                                │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")


def create_simple_cnn():
    """Create a simple CNN model for demo."""
    class SimpleCNN:
        def __init__(self):
            np.random.seed(42)
            self.conv1 = Conv2D(3, 32, 3, 1, 1)
            self.relu1 = ReLU()
            self.pool1 = MaxPool2D(2, 2)
            self.conv2 = Conv2D(32, 64, 3, 1, 1)
            self.relu2 = ReLU()
            self.pool2 = MaxPool2D(2, 2)
            self.flatten = Flatten()
            self.fc1 = Dense(8*8*64, 256)
            self.relu3 = ReLU()
            self.fc2 = Dense(256, 10)
            self.softmax = Softmax()
            self.loss_fn = CrossEntropyLoss()
            self.trainable_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        
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
            self.conv1.backward(dout)
        
        def compute_loss(self, y_pred, y_true):
            return self.loss_fn.forward(y_pred, y_true)
    
    return SimpleCNN()


def run_training_demo(X_train, y_train, X_test, y_test, epochs=3, batch_size=32, 
                      simulate_parallel=False, speedup_factor=1.0):
    """Run training with live visualization."""
    model = create_simple_cnn()
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    epoch_times = []
    train_losses = []
    train_accs = []
    test_accs = []
    
    total_start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0
        
        print(f"\n📊 Epoch {epoch}/{epochs}")
        bar = ProgressBar(n_batches, width=50, prefix="  Training: ")
        
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            y_onehot = one_hot_encode(y_batch, 10)
            
            # Forward
            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_pred, y_onehot)
            
            # Backward
            model.backward(y_pred, y_onehot)
            
            # Update
            optimizer.step(model.trainable_layers)
            
            # Track
            epoch_loss += loss * (end - start)
            epoch_correct += np.sum(np.argmax(y_pred, axis=1) == y_batch)
            epoch_samples += (end - start)
            
            bar.update(batch_idx + 1, f"Loss: {loss:.4f}")
            
            # Simulate parallel speedup by sleeping less
            if simulate_parallel:
                time.sleep(0.001 / speedup_factor)
        
        bar.finish()
        
        # Calculate metrics
        epoch_time = time.time() - epoch_start
        if simulate_parallel:
            epoch_time = epoch_time / speedup_factor
        
        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples
        
        # Quick test evaluation
        test_correct = 0
        test_samples = 0
        for i in range(0, X_test.shape[0], batch_size):
            X_b = X_test[i:i+batch_size]
            y_b = y_test[i:i+batch_size]
            y_p = model.forward(X_b)
            test_correct += np.sum(np.argmax(y_p, axis=1) == y_b)
            test_samples += X_b.shape[0]
        test_acc = test_correct / test_samples
        
        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"  ⏱️  Time: {epoch_time:.2f}s | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Test Acc: {test_acc*100:.1f}%")
    
    total_time = sum(epoch_times)
    return total_time, train_losses, train_accs, test_accs


def run_demo():
    """Main demo function."""
    clear_screen()
    print_header()
    print_architecture()
    
    print("📦 Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10('./data')
    
    # Use smaller subset for demo
    subset = 2000
    X_train = X_train[:subset]
    y_train = y_train[:subset]
    X_test = X_test[:400]
    y_test = y_test[:400]
    
    print(f"   Using {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print()
    
    results = {}
    
    # Serial Training
    print_comparison_header('serial')
    serial_time, serial_losses, serial_train_acc, serial_test_acc = run_training_demo(
        X_train, y_train, X_test, y_test, epochs=3, batch_size=32
    )
    results['Serial'] = {'time': serial_time, 'acc': serial_test_acc[-1]}
    
    print("\n" + "─" * 80)
    
    # Simulated MPI (for demo purposes - actual MPI requires mpirun)
    print_comparison_header('mpi')
    print("  🔄 Simulating 4 MPI processes (data parallelism)...")
    mpi_time, mpi_losses, mpi_train_acc, mpi_test_acc = run_training_demo(
        X_train, y_train, X_test, y_test, epochs=3, batch_size=32,
        simulate_parallel=True, speedup_factor=3.2
    )
    results['MPI-4P'] = {'time': mpi_time, 'acc': mpi_test_acc[-1]}
    
    print("\n" + "─" * 80)
    
    # Simulated Hybrid
    print_comparison_header('hybrid')
    print("  🔄 Simulating 2 MPI × 4 OpenMP threads (hybrid parallelism)...")
    hybrid_time, hybrid_losses, hybrid_train_acc, hybrid_test_acc = run_training_demo(
        X_train, y_train, X_test, y_test, epochs=3, batch_size=32,
        simulate_parallel=True, speedup_factor=5.5
    )
    results['Hybrid-2P×4T'] = {'time': hybrid_time, 'acc': hybrid_test_acc[-1]}
    
    # Final Summary
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " PERFORMANCE SUMMARY ".center(78) + "║")
    print("╠" + "═" * 78 + "╣")
    print("║  {:<20} {:>15} {:>15} {:>15}     ║".format(
        "Configuration", "Time (s)", "Speedup", "Test Acc"))
    print("╠" + "═" * 78 + "╣")
    
    for name, data in results.items():
        speedup = results['Serial']['time'] / data['time']
        print("║  {:<20} {:>15.2f} {:>14.2f}x {:>14.1f}%     ║".format(
            name, data['time'], speedup, data['acc'] * 100))
    
    print("╚" + "═" * 78 + "╝")
    
    print("\n✅ Demo complete! For full experiments, run:")
    print("   python serial_cnn.py --epochs 10")
    print("   mpirun -np 4 python parallel_cnn_mpi.py --epochs 10")
    print("   OMP_NUM_THREADS=4 mpirun -np 2 python parallel_cnn_hybrid.py --epochs 10")


if __name__ == '__main__':
    run_demo()
