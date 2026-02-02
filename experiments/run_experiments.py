#!/usr/bin/env python3
"""
Experiment Runner - Run all experiments and collect results
===========================================================
Runs serial and parallel implementations with various configurations.
"""

import subprocess
import os
import sys
import time
import numpy as np
import argparse


def run_serial_experiment(epochs, batch_size, subset, data_dir, output_dir):
    """Run serial CNN experiment."""
    print("\n" + "="*60)
    print("RUNNING SERIAL EXPERIMENT")
    print("="*60)
    
    output_file = os.path.join(output_dir, 'serial_metrics.npz')
    
    cmd = [
        sys.executable, 'serial_cnn.py',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--data-dir', data_dir,
        '--save-metrics', output_file
    ]
    
    if subset:
        cmd.extend(['--subset', str(subset)])
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + '/..')
    elapsed = time.time() - start_time
    
    print(f"Serial experiment completed in {elapsed:.2f}s")
    return result.returncode == 0


def run_mpi_experiment(epochs, batch_size, num_procs, subset, data_dir, output_dir):
    """Run MPI parallel experiment."""
    print("\n" + "="*60)
    print(f"RUNNING MPI EXPERIMENT ({num_procs} processes)")
    print("="*60)
    
    output_file = os.path.join(output_dir, f'mpi_{num_procs}p_metrics.npz')
    
    cmd = [
        'mpirun', '-np', str(num_procs),
        sys.executable, 'parallel_cnn_mpi.py',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--data-dir', data_dir,
        '--save-metrics', output_file
    ]
    
    if subset:
        cmd.extend(['--subset', str(subset)])
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + '/..')
    elapsed = time.time() - start_time
    
    print(f"MPI {num_procs}P experiment completed in {elapsed:.2f}s")
    return result.returncode == 0


def run_hybrid_experiment(epochs, batch_size, num_procs, num_threads, subset, data_dir, output_dir):
    """Run hybrid MPI+OpenMP experiment."""
    print("\n" + "="*60)
    print(f"RUNNING HYBRID EXPERIMENT ({num_procs} procs × {num_threads} threads)")
    print("="*60)
    
    output_file = os.path.join(output_dir, f'hybrid_{num_procs}p_{num_threads}t_metrics.npz')
    
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(num_threads)
    
    cmd = [
        'mpirun', '-np', str(num_procs),
        sys.executable, 'parallel_cnn_hybrid.py',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--num-threads', str(num_threads),
        '--data-dir', data_dir,
        '--save-metrics', output_file
    ]
    
    if subset:
        cmd.extend(['--subset', str(subset)])
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + '/..', env=env)
    elapsed = time.time() - start_time
    
    print(f"Hybrid {num_procs}P×{num_threads}T experiment completed in {elapsed:.2f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--subset', type=int, default=5000, help='Subset size (0 for full)')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--skip-serial', action='store_true', help='Skip serial experiment')
    parser.add_argument('--skip-mpi', action='store_true', help='Skip MPI experiments')
    parser.add_argument('--skip-hybrid', action='store_true', help='Skip hybrid experiments')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    subset = args.subset if args.subset > 0 else None
    
    print("="*60)
    print("DEEP LEARNING PARALLELIZATION EXPERIMENTS")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Subset: {subset if subset else 'Full dataset'}")
    print(f"Output directory: {args.output_dir}")
    
    results = []
    
    # Serial
    if not args.skip_serial:
        success = run_serial_experiment(
            args.epochs, args.batch_size, subset, args.data_dir, args.output_dir
        )
        results.append(('Serial', success))
    
    # MPI (2 and 4 processes)
    if not args.skip_mpi:
        for num_procs in [2, 4]:
            success = run_mpi_experiment(
                args.epochs, args.batch_size, num_procs, subset, args.data_dir, args.output_dir
            )
            results.append((f'MPI-{num_procs}P', success))
    
    # Hybrid (2 procs × 2 threads, 2 procs × 4 threads)
    if not args.skip_hybrid:
        for num_procs, num_threads in [(2, 2), (2, 4)]:
            success = run_hybrid_experiment(
                args.epochs, args.batch_size, num_procs, num_threads, subset, 
                args.data_dir, args.output_dir
            )
            results.append((f'Hybrid-{num_procs}P×{num_threads}T', success))
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    print("="*60)


if __name__ == '__main__':
    main()
