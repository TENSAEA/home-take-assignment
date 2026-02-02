#!/usr/bin/env python3
"""
Serial CNN Implementation for CIFAR-10 Classification
======================================================
This is the baseline serial implementation for performance comparison.

Architecture:
    Input (32x32x3) -> Conv2D(32) -> ReLU -> MaxPool 
    -> Conv2D(64) -> ReLU -> MaxPool -> Flatten 
    -> Dense(256) -> ReLU -> Dense(10) -> Softmax

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


class CNN:
    """Convolutional Neural Network for CIFAR-10 classification."""
    
    def __init__(self):
        """Initialize CNN layers."""
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
        # After 2 pooling layers: 32x32 -> 16x16 -> 8x8, with 64 channels = 8*8*64 = 4096
        self.fc1 = Dense(in_features=8*8*64, out_features=256)
        self.relu3 = ReLU()
        self.fc2 = Dense(in_features=256, out_features=10)
        
        # Softmax and loss
        self.softmax = Softmax()
        self.loss_fn = CrossEntropyLoss()
        
        # List of layers for optimizer
        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten,
            self.fc1, self.relu3,
            self.fc2
        ]
        
        # Trainable layers
        self.trainable_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input batch (N, 32, 32, 3)
        
        Returns:
            Softmax probabilities (N, 10)
        """
        # Conv Block 1
        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        
        # Conv Block 2
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        
        # Flatten and FC layers
        out = self.flatten.forward(out)
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        out = self.fc2.forward(out)
        
        # Softmax
        probs = self.softmax.forward(out)
        return probs
    
    def backward(self, y_pred, y_true):
        """
        Backward pass - compute gradients for all layers.
        
        Args:
            y_pred: Predicted probabilities (N, 10)
            y_true: One-hot encoded labels (N, 10)
        """
        # Loss gradient (softmax + cross-entropy combined)
        dout = self.loss_fn.backward(y_pred, y_true)
        
        # Backprop through FC layers (reverse order)
        dout = self.fc2.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.fc1.backward(dout)
        
        # Backprop through flatten
        dout = self.flatten.backward(dout)
        
        # Backprop through Conv Block 2
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)
        
        # Backprop through Conv Block 1
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)
    
    def compute_loss(self, y_pred, y_true):
        """Compute cross-entropy loss."""
        return self.loss_fn.forward(y_pred, y_true)
    
    def get_all_params(self):
        """Get all trainable parameters as flat arrays."""
        params = {}
        for i, layer in enumerate(self.trainable_layers):
            layer_params = layer.get_params()
            for key, val in layer_params.items():
                params[f'layer{i}_{key}'] = val.copy()
        return params
    
    def set_all_params(self, params):
        """Set all trainable parameters from flat arrays."""
        for i, layer in enumerate(self.trainable_layers):
            layer_params = {}
            for key in layer.get_params().keys():
                layer_params[key] = params[f'layer{i}_{key}']
            layer.set_params(layer_params)
    
    def get_all_grads(self):
        """Get all gradients as flat arrays."""
        grads = {}
        for i, layer in enumerate(self.trainable_layers):
            layer_grads = layer.get_grads()
            for key, val in layer_grads.items():
                grads[f'layer{i}_{key}'] = val.copy()
        return grads


def train_epoch(model, train_loader, optimizer, epoch, verbose=True):
    """Train for one epoch."""
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        # One-hot encode labels
        y_onehot = one_hot_encode(y_batch, num_classes=10)
        
        # Forward pass
        y_pred = model.forward(X_batch)
        
        # Compute loss
        loss = model.compute_loss(y_pred, y_onehot)
        
        # Backward pass
        model.backward(y_pred, y_onehot)
        
        # Update parameters
        optimizer.step(model.trainable_layers)
        
        # Track metrics
        total_loss += loss * X_batch.shape[0]
        total_correct += np.sum(np.argmax(y_pred, axis=1) == y_batch)
        total_samples += X_batch.shape[0]
        
        if verbose and (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss:.4f}, "
                  f"Acc: {total_correct/total_samples*100:.2f}%")
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, test_loader):
    """Evaluate model on test set."""
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for X_batch, y_batch in test_loader:
        y_onehot = one_hot_encode(y_batch, num_classes=10)
        
        # Forward pass only
        y_pred = model.forward(X_batch)
        loss = model.compute_loss(y_pred, y_onehot)
        
        total_loss += loss * X_batch.shape[0]
        total_correct += np.sum(np.argmax(y_pred, axis=1) == y_batch)
        total_samples += X_batch.shape[0]
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train(model, train_loader, test_loader, optimizer, epochs, verbose=True):
    """Full training loop."""
    metrics = TrainingMetrics()
    
    if verbose:
        print("\n" + "="*60)
        print("SERIAL CNN TRAINING - CIFAR-10")
        print("="*60)
        print(f"Training samples: {len(train_loader.X)}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Epochs: {epochs}")
        print("="*60)
    
    for epoch in range(1, epochs + 1):
        timer = Timer()
        timer.start()
        
        if verbose:
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch, verbose)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader)
        
        epoch_time = timer.stop()
        metrics.add_epoch(train_loss, train_acc, test_loss, test_acc, epoch_time)
        
        if verbose:
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")
            print(f"  Epoch Time: {epoch_time:.2f}s")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Serial CNN Training on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save-metrics', type=str, default='results/serial_metrics.npz',
                        help='Path to save metrics')
    parser.add_argument('--subset', type=int, default=None, 
                        help='Use subset of training data (for quick testing)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10(args.data_dir)
    
    # Use subset if specified
    if args.subset:
        X_train = X_train[:args.subset]
        y_train = y_train[:args.subset]
        X_test = X_test[:args.subset // 5]
        y_test = y_test[:args.subset // 5]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create data loaders
    train_loader = DataLoader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(X_test, y_test, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model and optimizer
    print("\nInitializing CNN model...")
    model = CNN()
    optimizer = SGD(learning_rate=args.lr, momentum=args.momentum)
    
    # Train
    metrics = train(model, train_loader, test_loader, optimizer, args.epochs)
    
    # Print summary
    metrics.print_summary()
    
    # Save metrics
    os.makedirs(os.path.dirname(args.save_metrics), exist_ok=True)
    metrics.save(args.save_metrics)
    print(f"\nMetrics saved to {args.save_metrics}")


if __name__ == '__main__':
    main()
