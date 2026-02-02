"""
CNN Layer Implementations (Pure NumPy)
Includes forward and backward passes for each layer.
"""

import numpy as np


class Conv2D:
    """
    2D Convolution Layer with padding support.
    Uses im2col for efficient implementation.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels, 1))
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Cache for backward pass
        self.cache = None
        
    def im2col(self, X, kernel_h, kernel_w, stride, pad):
        """Convert image to column matrix for efficient convolution."""
        N, H, W, C = X.shape
        
        out_h = (H + 2 * pad - kernel_h) // stride + 1
        out_w = (W + 2 * pad - kernel_w) // stride + 1
        
        # Pad input
        X_padded = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
        
        # im2col
        col = np.zeros((N, out_h, out_w, kernel_h, kernel_w, C))
        
        for y in range(kernel_h):
            y_max = y + stride * out_h
            for x in range(kernel_w):
                x_max = x + stride * out_w
                col[:, :, :, y, x, :] = X_padded[:, y:y_max:stride, x:x_max:stride, :]
        
        col = col.reshape(N * out_h * out_w, -1)
        return col, out_h, out_w
    
    def col2im(self, col, X_shape, kernel_h, kernel_w, stride, pad):
        """Convert column matrix back to image."""
        N, H, W, C = X_shape
        out_h = (H + 2 * pad - kernel_h) // stride + 1
        out_w = (W + 2 * pad - kernel_w) // stride + 1
        
        col = col.reshape(N, out_h, out_w, kernel_h, kernel_w, C)
        
        H_padded, W_padded = H + 2 * pad, W + 2 * pad
        X_padded = np.zeros((N, H_padded, W_padded, C))
        
        for y in range(kernel_h):
            y_max = y + stride * out_h
            for x in range(kernel_w):
                x_max = x + stride * out_w
                X_padded[:, y:y_max:stride, x:x_max:stride, :] += col[:, :, :, y, x, :]
        
        if pad > 0:
            return X_padded[:, pad:-pad, pad:-pad, :]
        return X_padded
    
    def forward(self, X):
        """
        Forward pass.
        X: (N, H, W, C) - batch of images
        Returns: (N, H_out, W_out, out_channels)
        """
        N, H, W, C = X.shape
        
        col, out_h, out_w = self.im2col(X, self.kernel_size, self.kernel_size, 
                                         self.stride, self.padding)
        
        # Reshape weights: (out_channels, in_channels, kh, kw) -> (out_channels, in_channels*kh*kw)
        W_col = self.W.reshape(self.out_channels, -1)
        
        # Convolution as matrix multiplication
        out = col @ W_col.T + self.b.T  # (N*out_h*out_w, out_channels)
        out = out.reshape(N, out_h, out_w, self.out_channels)
        
        self.cache = (X, col)
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        dout: (N, H_out, W_out, out_channels)
        Returns: dX (same shape as input X)
        """
        X, col = self.cache
        N, H, W, C = X.shape
        
        # Reshape dout
        dout_reshaped = dout.reshape(-1, self.out_channels)  # (N*out_h*out_w, out_channels)
        
        # Gradient of weights
        W_col = self.W.reshape(self.out_channels, -1)
        self.dW = (dout_reshaped.T @ col).reshape(self.W.shape)
        self.db = dout_reshaped.sum(axis=0, keepdims=True).T
        
        # Gradient of input
        dcol = dout_reshaped @ W_col  # (N*out_h*out_w, in_channels*kh*kw)
        dX = self.col2im(dcol, X.shape, self.kernel_size, self.kernel_size,
                         self.stride, self.padding)
        
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
        """
        Forward pass.
        X: (N, H, W, C)
        Returns: (N, H_out, W_out, C)
        """
        N, H, W, C = X.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        out = np.zeros((N, out_h, out_w, C))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                window = X[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = np.max(window, axis=(1, 2))
        
        self.cache = X
        return out
    
    def backward(self, dout):
        """
        Backward pass - route gradients to max locations.
        """
        X = self.cache
        N, H, W, C = X.shape
        _, out_h, out_w, _ = dout.shape
        
        dX = np.zeros_like(X)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                window = X[:, h_start:h_end, w_start:w_end, :]
                window_reshaped = window.reshape(N, -1, C)
                max_idx = np.argmax(window_reshaped, axis=1)
                
                # Create mask for max positions
                mask = np.zeros_like(window_reshaped)
                for n in range(N):
                    for c in range(C):
                        mask[n, max_idx[n, c], c] = 1
                
                mask = mask.reshape(window.shape)
                dX[:, h_start:h_end, w_start:w_end, :] += mask * dout[:, i:i+1, j:j+1, :]
        
        return dX


class Dense:
    """Fully Connected Layer."""
    
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # He initialization
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        self.cache = None
    
    def forward(self, X):
        """
        Forward pass.
        X: (N, in_features)
        Returns: (N, out_features)
        """
        self.cache = X
        return X @ self.W + self.b
    
    def backward(self, dout):
        """
        Backward pass.
        dout: (N, out_features)
        Returns: dX (N, in_features)
        """
        X = self.cache
        
        self.dW = X.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        dX = dout @ self.W.T
        
        return dX
    
    def get_params(self):
        return {'W': self.W, 'b': self.b}
    
    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']
    
    def get_grads(self):
        return {'W': self.dW, 'b': self.db}


class Flatten:
    """Flatten layer to convert from (N, H, W, C) to (N, H*W*C)."""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.input_shape)


class ReLU:
    """ReLU activation function."""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)
    
    def backward(self, dout):
        X = self.cache
        return dout * (X > 0)


class Softmax:
    """Softmax activation for classification output."""
    
    def forward(self, X):
        # Numerically stable softmax
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)


class CrossEntropyLoss:
    """Cross-entropy loss for classification."""
    
    def forward(self, y_pred, y_true):
        """
        y_pred: (N, num_classes) - softmax probabilities
        y_true: (N, num_classes) - one-hot labels
        Returns: scalar loss
        """
        N = y_pred.shape[0]
        # Clip for numerical stability
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / N
        return loss
    
    def backward(self, y_pred, y_true):
        """
        Gradient of softmax + cross-entropy combined.
        Returns: (N, num_classes)
        """
        N = y_pred.shape[0]
        return (y_pred - y_true) / N
