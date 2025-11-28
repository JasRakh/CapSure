#!/usr/bin/env python3
"""
Utility script to view training results.
Shows training metrics, plots, and model information.
"""

import argparse
import os
from pathlib import Path
import json
from ultralytics import YOLO


def find_latest_training_run(project_dir='runs/detect'):
    """Find the most recent training run."""
    project_path = Path(project_dir)
    if not project_path.exists():
        return None
    
    # Look for directories with training results
    runs = []
    for run_dir in project_path.iterdir():
        if run_dir.is_dir():
            weights_dir = run_dir / 'weights'
            if weights_dir.exists() and (weights_dir / 'best.pt').exists():
                runs.append(run_dir)
    
    if not runs:
        return None
    
    # Return the most recently modified one
    return max(runs, key=lambda x: x.stat().st_mtime)


def view_training_results(run_dir=None, project_dir='runs/detect'):
    """Display training results from a run directory."""
    
    if run_dir is None:
        run_dir = find_latest_training_run(project_dir)
        if run_dir is None:
            print("No training results found!")
            print(f"Looked in: {Path(project_dir).absolute()}")
            print("\nRun training first with: python main.py --train")
            return
    
    run_path = Path(run_dir)
    print("=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Run directory: {run_path.absolute()}\n")
    
    # Check for results files
    results_file = run_path / 'results.csv'
    args_file = run_path / 'args.yaml'
    weights_dir = run_path / 'weights'
    
    # Display model weights
    if weights_dir.exists():
        print("Model Weights:")
        if (weights_dir / 'best.pt').exists():
            print(f"  ✓ Best model: {weights_dir / 'best.pt'}")
            # Get model info
            try:
                model = YOLO(str(weights_dir / 'best.pt'))
                print(f"    Classes: {list(model.names.values())}")
            except:
                pass
        if (weights_dir / 'last.pt').exists():
            print(f"  ✓ Last epoch: {weights_dir / 'last.pt'}")
        print()
    
    # Display training arguments
    if args_file.exists():
        print("Training Configuration:")
        try:
            import yaml
            with open(args_file, 'r') as f:
                args = yaml.safe_load(f)
            for key, value in args.items():
                print(f"  {key}: {value}")
            print()
        except:
            pass
    
    # Display results CSV
    if results_file.exists():
        print("Training Metrics (from results.csv):")
        try:
            import pandas as pd
            df = pd.read_csv(results_file)
            # Show last few epochs
            print("\nLast 5 epochs:")
            print(df.tail(5).to_string(index=False))
            print("\nBest metrics:")
            if 'metrics/mAP50(B)' in df.columns:
                best_map = df['metrics/mAP50(B)'].max()
                best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch']
                print(f"  Best mAP50: {best_map:.4f} at epoch {int(best_epoch)}")
            if 'metrics/mAP50-95(B)' in df.columns:
                best_map95 = df['metrics/mAP50-95(B)'].max()
                best_epoch95 = df.loc[df['metrics/mAP50-95(B)'].idxmax(), 'epoch']
                print(f"  Best mAP50-95: {best_map95:.4f} at epoch {int(best_epoch95)}")
        except Exception as e:
            print(f"  Could not read results.csv: {e}")
        print()
    
    # List available plots
    plots_dir = run_path
    plot_files = []
    for ext in ['.png', '.jpg']:
        plot_files.extend(list(plots_dir.glob(f'*{ext}')))
    
    if plot_files:
        print("Available Plots/Images:")
        for plot_file in sorted(plot_files):
            print(f"  - {plot_file.name}")
        print()
    
    # Show path to results
    print("=" * 60)
    print("To view plots, open the following files:")
    print(f"  {run_path.absolute()}")
    print("\nKey files:")
    print(f"  - results.csv: Training metrics per epoch")
    print(f"  - confusion_matrix.png: Confusion matrix")
    print(f"  - F1_curve.png: F1 score curve")
    print(f"  - PR_curve.png: Precision-Recall curve")
    print(f"  - results.png: Training loss curves")
    print(f"  - weights/best.pt: Best model weights")
    print("=" * 60)


def open_results_directory(run_dir=None, project_dir='runs/detect'):
    """Open the results directory in the default file manager."""
    import subprocess
    import platform
    
    if run_dir is None:
        run_dir = find_latest_training_run(project_dir)
        if run_dir is None:
            print("No training results found!")
            return
    
    run_path = Path(run_dir)
    
    system = platform.system()
    if system == 'Darwin':  # macOS
        subprocess.run(['open', str(run_path)])
    elif system == 'Windows':
        subprocess.run(['explorer', str(run_path)])
    else:  # Linux
        subprocess.run(['xdg-open', str(run_path)])
    
    print(f"Opened: {run_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description='View training results')
    parser.add_argument('--run-dir', type=str, help='Path to specific training run directory')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory')
    parser.add_argument('--open', action='store_true', help='Open results directory in file manager')
    
    args = parser.parse_args()
    
    if args.open:
        open_results_directory(args.run_dir, args.project)
    else:
        view_training_results(args.run_dir, args.project)


if __name__ == '__main__':
    main()

