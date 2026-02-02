#!/usr/bin/env python3
"""
Unit Tests for CNN Layers
=========================
Tests forward/backward passes for all layer implementations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import numpy as np
from utils.layers import Conv2D, MaxPool2D, Dense, Flatten, ReLU, Softmax, CrossEntropyLoss


def test_conv2d_forward():
    """Test Conv2D forward pass output shapes."""
    print("Testing Conv2D forward... ", end="")
    
    conv = Conv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    X = np.random.randn(4, 32, 32, 3).astype(np.float32)  # Batch of 4 images
    
    out = conv.forward(X)
    
    assert out.shape == (4, 32, 32, 32), f"Expected (4, 32, 32, 32), got {out.shape}"
    print("✓ PASSED")


def test_conv2d_backward():
    """Test Conv2D backward pass gradient shapes."""
    print("Testing Conv2D backward... ", end="")
    
    conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    X = np.random.randn(2, 8, 8, 3).astype(np.float32)
    
    out = conv.forward(X)
    dout = np.random.randn(*out.shape)
    dX = conv.backward(dout)
    
    assert dX.shape == X.shape, f"Expected {X.shape}, got {dX.shape}"
    assert conv.dW.shape == conv.W.shape, f"dW shape mismatch"
    assert conv.db.shape == conv.b.shape, f"db shape mismatch"
    print("✓ PASSED")


def test_maxpool_forward():
    """Test MaxPool2D forward pass."""
    print("Testing MaxPool2D forward... ", end="")
    
    pool = MaxPool2D(pool_size=2, stride=2)
    X = np.random.randn(4, 16, 16, 32).astype(np.float32)
    
    out = pool.forward(X)
    
    assert out.shape == (4, 8, 8, 32), f"Expected (4, 8, 8, 32), got {out.shape}"
    print("✓ PASSED")


def test_maxpool_backward():
    """Test MaxPool2D backward pass."""
    print("Testing MaxPool2D backward... ", end="")
    
    pool = MaxPool2D(pool_size=2, stride=2)
    X = np.random.randn(2, 8, 8, 16).astype(np.float32)
    
    out = pool.forward(X)
    dout = np.random.randn(*out.shape)
    dX = pool.backward(dout)
    
    assert dX.shape == X.shape, f"Expected {X.shape}, got {dX.shape}"
    print("✓ PASSED")


def test_dense_forward():
    """Test Dense layer forward pass."""
    print("Testing Dense forward... ", end="")
    
    dense = Dense(in_features=256, out_features=10)
    X = np.random.randn(4, 256).astype(np.float32)
    
    out = dense.forward(X)
    
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"
    print("✓ PASSED")


def test_dense_backward():
    """Test Dense layer backward pass."""
    print("Testing Dense backward... ", end="")
    
    dense = Dense(in_features=128, out_features=64)
    X = np.random.randn(2, 128).astype(np.float32)
    
    out = dense.forward(X)
    dout = np.random.randn(*out.shape)
    dX = dense.backward(dout)
    
    assert dX.shape == X.shape, f"Expected {X.shape}, got {dX.shape}"
    assert dense.dW.shape == dense.W.shape
    print("✓ PASSED")


def test_relu_forward_backward():
    """Test ReLU activation."""
    print("Testing ReLU... ", end="")
    
    relu = ReLU()
    X = np.array([[-1, 0, 1], [2, -2, 0.5]])
    
    out = relu.forward(X)
    expected = np.array([[0, 0, 1], [2, 0, 0.5]])
    
    assert np.allclose(out, expected), "ReLU forward failed"
    
    dout = np.ones_like(out)
    dX = relu.backward(dout)
    expected_grad = np.array([[0, 0, 1], [1, 0, 1]])
    
    assert np.allclose(dX, expected_grad), "ReLU backward failed"
    print("✓ PASSED")


def test_softmax():
    """Test Softmax activation."""
    print("Testing Softmax... ", end="")
    
    softmax = Softmax()
    X = np.array([[1, 2, 3], [1, 1, 1]])
    
    out = softmax.forward(X)
    
    # Check sum = 1
    assert np.allclose(out.sum(axis=1), [1, 1]), "Softmax doesn't sum to 1"
    # Check all positive
    assert (out > 0).all(), "Softmax has negative values"
    # Check uniform input gives uniform output
    assert np.allclose(out[1], [1/3, 1/3, 1/3]), "Uniform input should give uniform output"
    print("✓ PASSED")


def test_cross_entropy():
    """Test Cross-Entropy loss."""
    print("Testing CrossEntropyLoss... ", end="")
    
    loss_fn = CrossEntropyLoss()
    
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y_true = np.array([[1, 0, 0], [0, 1, 0]])  # One-hot
    
    loss = loss_fn.forward(y_pred, y_true)
    assert loss > 0, "Loss should be positive"
    
    grad = loss_fn.backward(y_pred, y_true)
    assert grad.shape == y_pred.shape, "Gradient shape mismatch"
    print("✓ PASSED")


def test_flatten():
    """Test Flatten layer."""
    print("Testing Flatten... ", end="")
    
    flatten = Flatten()
    X = np.random.randn(4, 8, 8, 64)
    
    out = flatten.forward(X)
    assert out.shape == (4, 8*8*64), f"Expected (4, 4096), got {out.shape}"
    
    dout = np.random.randn(*out.shape)
    dX = flatten.backward(dout)
    assert dX.shape == X.shape, f"Expected {X.shape}, got {dX.shape}"
    print("✓ PASSED")


def test_gradient_numerical():
    """Numerical gradient check for Dense layer."""
    print("Testing numerical gradients (Dense)... ", end="")
    
    dense = Dense(in_features=4, out_features=3)
    X = np.random.randn(2, 4)
    
    # Forward
    out = dense.forward(X)
    dout = np.random.randn(*out.shape)
    
    # Analytical gradient
    dense.backward(dout)
    analytical_dW = dense.dW.copy()
    
    # Numerical gradient
    eps = 1e-5
    numerical_dW = np.zeros_like(dense.W)
    
    for i in range(dense.W.shape[0]):
        for j in range(dense.W.shape[1]):
            W_plus = dense.W.copy()
            W_plus[i, j] += eps
            dense.W = W_plus
            out_plus = dense.forward(X)
            loss_plus = np.sum(out_plus * dout)
            
            W_minus = dense.W.copy()
            W_minus[i, j] -= 2 * eps
            dense.W = W_minus
            out_minus = dense.forward(X)
            loss_minus = np.sum(out_minus * dout)
            
            numerical_dW[i, j] = (loss_plus - loss_minus) / (2 * eps)
            dense.W[i, j] = W_plus[i, j] - eps  # Reset
    
    diff = np.abs(analytical_dW - numerical_dW).max()
    assert diff < 1e-4, f"Gradient check failed, max diff: {diff}"
    print("✓ PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_conv2d_forward,
        test_conv2d_backward,
        test_maxpool_forward,
        test_maxpool_backward,
        test_dense_forward,
        test_dense_backward,
        test_relu_forward_backward,
        test_softmax,
        test_cross_entropy,
        test_flatten,
        test_gradient_numerical,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
