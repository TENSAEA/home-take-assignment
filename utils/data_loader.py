"""
CIFAR-10 Data Loader
Downloads and preprocesses CIFAR-10 dataset for training.
"""

import numpy as np
import os
import pickle
import urllib.request
import tarfile


def download_cifar10(data_dir='./data'):
    """Download CIFAR-10 dataset if not already present."""
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    extract_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    
    if os.path.exists(extract_dir):
        return extract_dir
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(filename):
        print(f"Downloading CIFAR-10 dataset (~170MB)...")
        
        def progress_hook(count, block_size, total_size):
            percent = count * block_size * 100 / total_size
            mb_downloaded = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        
        urllib.request.urlretrieve(url, filename, progress_hook)
        print("\nDownload complete.")
    
    print("Extracting CIFAR-10 dataset...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(data_dir)
    print("Extraction complete.")
    
    return extract_dir


def load_cifar10_batch(batch_file):
    """Load a single CIFAR-10 batch file."""
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    data = batch[b'data']
    labels = batch[b'labels']
    
    # Reshape to (N, 3, 32, 32) and transpose to (N, 32, 32, 3)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return data, np.array(labels)


def load_cifar10(data_dir='./data', normalize=True):
    """
    Load the full CIFAR-10 dataset.
    
    Returns:
        X_train, y_train, X_test, y_test: Training and test data
    """
    extract_dir = download_cifar10(data_dir)
    
    # Load training batches
    X_train_list = []
    y_train_list = []
    
    for i in range(1, 6):
        batch_file = os.path.join(extract_dir, f'data_batch_{i}')
        X, y = load_cifar10_batch(batch_file)
        X_train_list.append(X)
        y_train_list.append(y)
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # Load test batch
    test_file = os.path.join(extract_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(test_file)
    
    # Normalize to [0, 1]
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        # Standardize using training mean and std
        mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
        std = X_train.std(axis=(0, 1, 2), keepdims=True)
        X_train = (X_train - mean) / (std + 1e-8)
        X_test = (X_test - mean) / (std + 1e-8)
    
    return X_train, y_train, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """One-hot encode labels."""
    return np.eye(num_classes)[y]


class DataLoader:
    """Mini-batch data loader with shuffling."""
    
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        
    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]
    
    def __len__(self):
        return self.n_batches


if __name__ == '__main__':
    # Test data loading
    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10()
    print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")
    print(f"Classes: {np.unique(y_train)}")
