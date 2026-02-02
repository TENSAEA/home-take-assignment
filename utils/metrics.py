"""
Metrics and Timing Utilities
"""

import time
import numpy as np


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class TrainingMetrics:
    """Track training metrics over epochs."""
    
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.epoch_times = []
        self.total_time = 0
    
    def add_epoch(self, train_loss, train_acc, test_loss, test_acc, epoch_time):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        self.epoch_times.append(epoch_time)
        self.total_time += epoch_time
    
    def get_summary(self):
        return {
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_train_acc': self.train_accuracies[-1] if self.train_accuracies else None,
            'final_test_loss': self.test_losses[-1] if self.test_losses else None,
            'final_test_acc': self.test_accuracies[-1] if self.test_accuracies else None,
            'total_time': self.total_time,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else None,
            'num_epochs': len(self.train_losses)
        }
    
    def print_summary(self):
        summary = self.get_summary()
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Total Epochs: {summary['num_epochs']}")
        print(f"Total Training Time: {summary['total_time']:.2f}s")
        print(f"Average Epoch Time: {summary['avg_epoch_time']:.2f}s")
        print(f"Final Train Loss: {summary['final_train_loss']:.4f}")
        print(f"Final Train Accuracy: {summary['final_train_acc']*100:.2f}%")
        print(f"Final Test Loss: {summary['final_test_loss']:.4f}")
        print(f"Final Test Accuracy: {summary['final_test_acc']*100:.2f}%")
        print("="*50)
    
    def save(self, filepath):
        """Save metrics to numpy file."""
        np.savez(filepath,
                 train_losses=self.train_losses,
                 train_accuracies=self.train_accuracies,
                 test_losses=self.test_losses,
                 test_accuracies=self.test_accuracies,
                 epoch_times=self.epoch_times,
                 total_time=self.total_time)
    
    @classmethod
    def load(cls, filepath):
        """Load metrics from numpy file."""
        data = np.load(filepath)
        metrics = cls()
        metrics.train_losses = data['train_losses'].tolist()
        metrics.train_accuracies = data['train_accuracies'].tolist()
        metrics.test_losses = data['test_losses'].tolist()
        metrics.test_accuracies = data['test_accuracies'].tolist()
        metrics.epoch_times = data['epoch_times'].tolist()
        metrics.total_time = float(data['total_time'])
        return metrics


def compute_accuracy(y_pred, y_true):
    """
    Compute classification accuracy.
    
    y_pred: (N, num_classes) - predicted probabilities
    y_true: (N,) or (N, num_classes) - true labels
    """
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(pred_labels == y_true)
