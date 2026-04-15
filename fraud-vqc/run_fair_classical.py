#!/usr/bin/env python
"""
Run Fair Classical Baseline + Generate Summary Tables
-------------------------------------------------------
Usage: python run_fair_classical.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import load_data, preprocess, split_data, quantum_subsample
from src.fair_classical import run_fair_classical_baseline
from src.generate_tables import generate_all_tables


def main():
    print("\n" + "="*80)
    print("FAIR CLASSICAL BASELINE + SUMMARY TABLES")
    print("="*80)
    
    # Load and preprocess data
    print("\nLoading creditcard dataset...")
    df = load_data("creditcard.csv")
    X, y = preprocess(df, pca_components=8)
    X_train_full, X_test_full, y_train_full, y_test_full = split_data(X, y)
    
    # Get the SAME 160-sample subset that VQC uses
    print("\nSubsampling 160 training samples (80 fraud, 80 legit)...")
    X_train_160, y_train_160 = quantum_subsample(X_train_full, y_train_full,
                                                   n_fraud=80, n_legit=80)
    
    # Create quantum test set (249 samples) to match VQC evaluation
    X_test_small, y_test_small = quantum_subsample(X_test_full, y_test_full,
                                                     n_fraud=49, n_legit=200)
    
    print(f"  Train: {len(X_train_160)} samples (fraud={y_train_160.sum()}, legit={(y_train_160==0).sum()})")
    print(f"  Test:  {len(X_test_small)} samples (fraud={y_test_small.sum()}, legit={(y_test_small==0).sum()})")
    
    # Run fair classical baseline
    print("\nTraining classical models on 160-sample subset...")
    fair_results = run_fair_classical_baseline(X_train_160, y_train_160, 
                                                X_test_small, y_test_small,
                                                save_dir="results")
    
    print("\n✓ Fair classical baseline complete!")
    
    # Generate all comparison tables
    print("\nGenerating summary tables...")
    generate_all_tables(results_dir="results")
    
    print("\n" + "="*80)
    print("SCRIPT COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print("  - results/classical_results_fair_160.json (fair classical on 160 samples)")
    print("\nCheck results/ directory for all plots and JSON results.")
    

if __name__ == "__main__":
    main()
