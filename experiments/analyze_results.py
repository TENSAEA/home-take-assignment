#!/usr/bin/env python3
"""
Analyze Results - Generate plots and performance analysis
=========================================================
Creates visualizations comparing serial and parallel implementations.
"""

import os
import sys
import numpy as np
import argparse

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from utils.metrics import TrainingMetrics


def load_metrics(results_dir):
    """Load all metrics files from results directory."""
    metrics = {}
    
    # Serial
    serial_path = os.path.join(results_dir, 'serial_metrics.npz')
    if os.path.exists(serial_path):
        metrics['Serial'] = TrainingMetrics.load(serial_path)
    
    # MPI
    for p in [2, 4]:
        path = os.path.join(results_dir, f'mpi_{p}p_metrics.npz')
        if os.path.exists(path):
            metrics[f'MPI-{p}P'] = TrainingMetrics.load(path)
    
    # Hybrid
    for p, t in [(2, 2), (2, 4)]:
        path = os.path.join(results_dir, f'hybrid_{p}p_{t}t_metrics.npz')
        if os.path.exists(path):
            metrics[f'Hybrid-{p}P×{t}T'] = TrainingMetrics.load(path)
    
    return metrics


def print_performance_table(metrics):
    """Print performance comparison table."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    headers = ['Configuration', 'Total Time(s)', 'Speedup', 'Final Train Acc', 'Final Test Acc']
    print(f"{'Configuration':<20} {'Time(s)':<12} {'Speedup':<10} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 80)
    
    serial_time = metrics['Serial'].total_time if 'Serial' in metrics else None
    
    for name, m in metrics.items():
        summary = m.get_summary()
        time_taken = summary['total_time']
        speedup = serial_time / time_taken if serial_time else 1.0
        train_acc = summary['final_train_acc'] * 100
        test_acc = summary['final_test_acc'] * 100
        
        print(f"{name:<20} {time_taken:<12.2f} {speedup:<10.2f}x {train_acc:<12.2f}% {test_acc:<12.2f}%")
    
    print("="*80)


def print_scalability_analysis(metrics):
    """Print scalability analysis."""
    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS")
    print("="*80)
    
    if 'Serial' not in metrics:
        print("Serial baseline not found, cannot compute scalability.")
        return
    
    serial_time = metrics['Serial'].total_time
    
    print(f"\nSerial Baseline: {serial_time:.2f}s")
    print("\nParallel Configurations:")
    print(f"{'Config':<20} {'Workers':<10} {'Time(s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 70)
    
    configs = [
        ('MPI-2P', 2),
        ('MPI-4P', 4),
        ('Hybrid-2P×2T', 4),
        ('Hybrid-2P×4T', 8),
    ]
    
    for name, workers in configs:
        if name in metrics:
            time_taken = metrics[name].total_time
            speedup = serial_time / time_taken
            efficiency = speedup / workers * 100
            print(f"{name:<20} {workers:<10} {time_taken:<12.2f} {speedup:<10.2f}x {efficiency:<12.2f}%")
    
    print("="*80)


def generate_loss_comparison(metrics, output_dir):
    """Generate loss curve comparison (text-based for simplicity)."""
    print("\n" + "="*80)
    print("LOSS CURVES (First 5 epochs)")
    print("="*80)
    
    print(f"{'Epoch':<8}", end="")
    for name in metrics:
        print(f"{name:<15}", end="")
    print()
    print("-" * (8 + 15 * len(metrics)))
    
    max_epochs = min(5, min(len(m.train_losses) for m in metrics.values()))
    
    for epoch in range(max_epochs):
        print(f"{epoch+1:<8}", end="")
        for name, m in metrics.items():
            print(f"{m.train_losses[epoch]:<15.4f}", end="")
        print()
    
    print("="*80)


def save_results_csv(metrics, output_dir):
    """Save results to CSV file."""
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    
    with open(csv_path, 'w') as f:
        f.write('Configuration,Total_Time,Epochs,Final_Train_Loss,Final_Train_Acc,Final_Test_Loss,Final_Test_Acc\n')
        
        for name, m in metrics.items():
            summary = m.get_summary()
            f.write(f"{name},{summary['total_time']:.2f},{summary['num_epochs']},")
            f.write(f"{summary['final_train_loss']:.4f},{summary['final_train_acc']:.4f},")
            f.write(f"{summary['final_test_loss']:.4f},{summary['final_test_acc']:.4f}\n")
    
    print(f"\nResults saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results-dir', type=str, default='./results', help='Results directory')
    args = parser.parse_args()
    
    print("Loading experiment results...")
    metrics = load_metrics(args.results_dir)
    
    if not metrics:
        print("No results found. Run experiments first.")
        return
    
    print(f"Found {len(metrics)} experiment results: {list(metrics.keys())}")
    
    print_performance_table(metrics)
    print_scalability_analysis(metrics)
    generate_loss_comparison(metrics, args.results_dir)
    save_results_csv(metrics, args.results_dir)


if __name__ == '__main__':
    main()
