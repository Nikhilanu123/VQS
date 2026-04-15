"""
Main Experiment Runner
-----------------------
Runs all phases in order:
  Phase 1 → Data preprocessing + classical baselines
  Phase 2 → Standard VQC baseline
  Phase 3 → QCS-VQC (novel method) + ablation studies
  Phase 4 → IBM hardware validation (optional)

Usage:
  python main.py                      # Run all phases (skip IBM)
  python main.py --phase 1            # Classical baselines only
  python main.py --phase 2            # Standard VQC only
  python main.py --phase 3            # QCS-VQC + ablations
  python main.py --ibm YOUR_TOKEN     # Include IBM hardware validation
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import (
    load_data, preprocess, split_data, quantum_subsample, plot_class_distribution
)
from src.classical_baselines import run_all_classical_baselines
from src.vqc_baseline import run_standard_vqc
from src.qcs_vqc import run_qcs_vqc, plot_loss_comparison
from src.ablation_study import run_component_ablations, run_qubit_sweep, run_imbalance_sweep

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "data_path"      : "creditcard.csv",
    "results_dir"    : "results",
    "num_qubits"     : 8,
    "vqc_reps"       : 2,
    "vqc_max_iter"   : 100,
    "qcs_max_iter"   : 80,    # SPSA iterations (2 loss evals each)
    "qcs_n_restarts" : 2,     # multi-restart for main QCS-VQC
    "q_train_fraud"  : 80,    # 80+80=160 total
    "q_train_legit"  : 80,
    # Quantum test set (small! circuit evals are expensive)
    "q_test_fraud"   : 49,    # all fraud in test set (~98 total, take half)
    "q_test_legit"   : 200,   # balanced legit samples
    "q_test_samples" : 20,    # IBM hardware -- absolute minimum
    # Ablation settings
    "ablation_max_iter"      : 40,
    "ablation_qubit_iter"    : 30,
    "ablation_imbalance_iter": 30,
}


# ─────────────────────────────────────────────────────────────────────────────
# FINAL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_final_table(all_results: list):
    print("\n" + "="*75)
    print("FINAL RESULTS SUMMARY")
    print("="*75)
    header = f"{'Model':<40} {'AUC-ROC':>8} {'F1':>8} {'MCC':>8} {'G-Mean':>8}"
    print(header)
    print("-"*75)
    for r in all_results:
        print(f"{r['model']:<40} "
              f"{r.get('auc_roc',0):>8.4f} "
              f"{r.get('f1',0):>8.4f} "
              f"{r.get('mcc',0):>8.4f} "
              f"{r.get('g_mean',0):>8.4f}")
    print("="*75)


def plot_final_comparison(all_results, save_dir="results"):
    import matplotlib.pyplot as plt  # Agg backend already set at top
    os.makedirs(save_dir, exist_ok=True)
    metrics = ['auc_roc', 'auc_prc', 'f1', 'mcc', 'g_mean']
    xlabels = ['AUC-ROC', 'AUC-PRC', 'F1', 'MCC', 'G-Mean']
    colors  = ['steelblue', 'darkorange', 'green', 'crimson', 'purple', 'brown']

    x = np.arange(len(metrics))
    width = 0.8 / len(all_results)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (res, color) in enumerate(zip(all_results, colors)):
        vals = [res.get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=res['model'][:35],
                      color=color, alpha=0.85)

    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 1.05)
    ax.set_title("All Models -- Final Metric Comparison", fontsize=13)
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_path = f"{save_dir}/final_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Final comparison plot saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QCS-VQC Fraud Detection Experiments")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1=classical, 2=VQC, 3=QCS-VQC, 0=all)")
    parser.add_argument("--ibm", type=str, default=None,
                        help="IBM Quantum API token (enables Phase 4)")
    parser.add_argument("--fast", action="store_true",
                        help="Reduce iterations for quick testing")
    args = parser.parse_args()

    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    if args.fast:
        CONFIG["vqc_max_iter"] = 20
        CONFIG["qcs_max_iter"] = 20
        CONFIG["q_train_fraud"] = 50
        CONFIG["q_train_legit"] = 50
        CONFIG["q_test_fraud"]  = 25
        CONFIG["q_test_legit"]  = 75
        print("[FAST MODE] Reduced iterations and samples for quick testing.\n")

    all_results = []

    # ─── LOAD DATA ────────────────────────────────────────────────────────────
    df = load_data(CONFIG["data_path"])
    # Real imbalance ratio -- used by QCS-VQC for cost-sensitive weights
    FRAUD_RATIO = df['Class'].mean()   # ~0.00173 (492/284807)
    print(f"  Dataset fraud ratio: {FRAUD_RATIO:.5f}")
    plot_class_distribution(df['Class'].values, save_path=f"{CONFIG['results_dir']}/class_distribution.png")

    # ─── PHASE 1 -- Classical Baselines ────────────────────────────────────────
    if args.phase in (0, 1):
        print("\n\n>>> PHASE 1: CLASSICAL BASELINES <<<")
        X_full, y = preprocess(df)
        X_train, X_test, y_train, y_test = split_data(X_full, y)
        classical_results, _ = run_all_classical_baselines(X_train, y_train, X_test, y_test,
                                                            save_dir=CONFIG["results_dir"])
        all_results.extend(classical_results)

    # ─── PHASE 2 -- Standard VQC Baseline ─────────────────────────────────────
    if args.phase in (0, 2):
        print("\n\n>>> PHASE 2: STANDARD VQC BASELINE <<<")
        X_q, y_q = preprocess(df, pca_components=CONFIG["num_qubits"])
        X_q_train, X_q_test, y_q_train, y_q_test = split_data(X_q, y_q)
        X_sub, y_sub = quantum_subsample(X_q_train, y_q_train,
                                          n_fraud=CONFIG["q_train_fraud"],
                                          n_legit=CONFIG["q_train_legit"])
        # Small stratified quantum test set (circuit eval is expensive)
        X_q_test_small, y_q_test_small = quantum_subsample(
            X_q_test, y_q_test,
            n_fraud=CONFIG["q_test_fraud"],
            n_legit=CONFIG["q_test_legit"]
        )
        print(f"  Quantum eval set : {len(X_q_test_small)} samples "
              f"(fraud={y_q_test_small.sum()}, legit={(y_q_test_small==0).sum()})")
        vqc_res, _ = run_standard_vqc(
            X_sub, y_sub, X_q_test_small, y_q_test_small,
            num_qubits=CONFIG["num_qubits"],
            reps=CONFIG["vqc_reps"],
            max_iter=CONFIG["vqc_max_iter"],
            save_dir=CONFIG["results_dir"]
        )
        all_results.append(vqc_res)

    # ─── PHASE 3 -- QCS-VQC + Ablations ───────────────────────────────────────
    if args.phase in (0, 3):
        print("\n\n>>> PHASE 3: QCS-VQC (NOVEL METHOD) <<<")
        X_q, y_q = preprocess(df, pca_components=CONFIG["num_qubits"])
        X_q_train, X_q_test, y_q_train, y_q_test = split_data(X_q, y_q)
        X_sub, y_sub = quantum_subsample(X_q_train, y_q_train,
                                          n_fraud=CONFIG["q_train_fraud"],
                                          n_legit=CONFIG["q_train_legit"])
        X_q_test_small, y_q_test_small = quantum_subsample(
            X_q_test, y_q_test,
            n_fraud=CONFIG["q_test_fraud"],
            n_legit=CONFIG["q_test_legit"]
        )
        print(f"  Quantum eval set : {len(X_q_test_small)} samples "
              f"(fraud={y_q_test_small.sum()}, legit={(y_q_test_small==0).sum()})")

        # --- Checkpoint: Main QCS-VQC ---
        qcs_path = f"{CONFIG['results_dir']}/qcs_vqc_results.json"
        if os.path.exists(qcs_path):
            with open(qcs_path) as f:
                qcs_res = json.load(f)
            if qcs_res.get('auc_roc', 0) > 0.4:  # valid result (not a degenerate COBYLA run)
                print(f"  [RESUME] Main QCS-VQC already done (AUC={qcs_res['auc_roc']:.4f}), skipping.")
                all_results.append(qcs_res)
                trained_qcs = None  # no trainer object available on resume
            else:
                print(f"  [RESUME] Stale QCS-VQC result (AUC={qcs_res.get('auc_roc',0):.4f}), re-running.")
                qcs_res = None
        else:
            qcs_res = None

        if qcs_res is None:
            qcs_res, trained_qcs = run_qcs_vqc(
                X_sub, y_sub, X_q_test_small, y_q_test_small,
                num_qubits=CONFIG["num_qubits"],
                reps=CONFIG["vqc_reps"],
                max_iter=CONFIG["qcs_max_iter"],
                use_noise=False, use_zne=False,
                dataset_fraud_ratio=FRAUD_RATIO,
                n_restarts=CONFIG["qcs_n_restarts"],
                save_dir=CONFIG["results_dir"]
            )
            all_results.append(qcs_res)

        # --- Checkpoint: Component Ablations ---
        print("\n\n>>> PHASE 3b: ABLATION STUDIES <<<")
        abl_path = f"{CONFIG['results_dir']}/ablation_component_results.json"
        if os.path.exists(abl_path):
            with open(abl_path) as f:
                abl_data = json.load(f)
            if isinstance(abl_data, list) and len(abl_data) >= 5:
                print(f"  [RESUME] Component ablations done ({len(abl_data)} configs), skipping.")
            else:
                print(f"  [RESUME] Partial component ablations ({len(abl_data) if isinstance(abl_data, list) else 0}/5), resuming...")
                run_component_ablations(X_sub, y_sub, X_q_test_small, y_q_test_small,
                                         num_qubits=CONFIG["num_qubits"],
                                         max_iter=CONFIG["ablation_max_iter"],
                                         dataset_fraud_ratio=FRAUD_RATIO,
                                         save_dir=CONFIG["results_dir"])
        else:
            run_component_ablations(X_sub, y_sub, X_q_test_small, y_q_test_small,
                                     num_qubits=CONFIG["num_qubits"],
                                     max_iter=CONFIG["ablation_max_iter"],
                                     dataset_fraud_ratio=FRAUD_RATIO,
                                     save_dir=CONFIG["results_dir"])

        # --- Checkpoint: Qubit Sweep ---
        qs_path = f"{CONFIG['results_dir']}/ablation_qubit_sweep.json"
        if os.path.exists(qs_path):
            with open(qs_path) as f:
                qs_data = json.load(f)
            if isinstance(qs_data, list) and len(qs_data) >= 3:
                print(f"  [RESUME] Qubit sweep done ({len(qs_data)} configs), skipping.")
            else:
                run_qubit_sweep(X_q_train, y_q_train, X_q_test, y_q_test,
                                 qubit_counts=(4, 6, 8), max_iter=CONFIG["ablation_qubit_iter"],
                                 save_dir=CONFIG["results_dir"], raw_df=df,
                                 dataset_fraud_ratio=FRAUD_RATIO)
        else:
            run_qubit_sweep(X_q_train, y_q_train, X_q_test, y_q_test,
                             qubit_counts=(4, 6, 8), max_iter=CONFIG["ablation_qubit_iter"],
                             save_dir=CONFIG["results_dir"], raw_df=df,
                             dataset_fraud_ratio=FRAUD_RATIO)

        # --- Checkpoint: Imbalance Sweep ---
        imb_path = f"{CONFIG['results_dir']}/ablation_imbalance_sweep.json"
        if os.path.exists(imb_path):
            with open(imb_path) as f:
                imb_data = json.load(f)
            if isinstance(imb_data, dict) and len(imb_data.get('qcs', [])) >= 4:
                print(f"  [RESUME] Imbalance sweep done, skipping.")
            else:
                run_imbalance_sweep(X_sub, y_sub, X_q_test_small, y_q_test_small,
                                     ratios=(0.1, 0.2, 0.3, 0.5),
                                     num_qubits=CONFIG["num_qubits"],
                                     max_iter=CONFIG["ablation_imbalance_iter"],
                                     save_dir=CONFIG["results_dir"])
        else:
            run_imbalance_sweep(X_sub, y_sub, X_q_test_small, y_q_test_small,
                                 ratios=(0.1, 0.2, 0.3, 0.5),
                                 num_qubits=CONFIG["num_qubits"],
                                 max_iter=CONFIG["ablation_imbalance_iter"],
                                 save_dir=CONFIG["results_dir"])

        # Compare loss curves
        vqc_loss_path  = f"{CONFIG['results_dir']}/vqc_standard_results.json"
        if os.path.exists(vqc_loss_path):
            with open(vqc_loss_path) as f:
                vqc_std_data = json.load(f)
            plot_loss_comparison(vqc_std_data.get('loss_history', []),
                                  qcs_res.get('loss_history', []),
                                  save_path=f"{CONFIG['results_dir']}/loss_comparison.png")

    # ─── PHASE 4 -- IBM Hardware Validation ───────────────────────────────────
    if args.ibm:
        print("\n\n>>> PHASE 4: IBM QUANTUM HARDWARE VALIDATION <<<")
        print("  WARNING: This uses your free IBM quantum minutes!")
        print("  Keep X_q_test_small to <= 20 samples.")
        from src.ibm_validation import run_ibm_validation, plot_simulator_vs_hardware

        # Use smallest possible test set
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_q_test), size=CONFIG["q_test_samples"], replace=False)
        X_hw_test = X_q_test[idx]
        y_hw_test = y_q_test[idx]

        hw_metrics = run_ibm_validation(
            trained_trainer=trained_qcs,
            X_test_small=X_hw_test,
            y_test_small=y_hw_test,
            ibm_token=args.ibm,
            save_dir=CONFIG["results_dir"]
        )
        if hw_metrics:
            all_results.append(hw_metrics)
            plot_simulator_vs_hardware(qcs_res, hw_metrics,
                                         save_dir=CONFIG["results_dir"])

    # ─── FINAL SUMMARY ────────────────────────────────────────────────────────
    if all_results:
        print_final_table(all_results)
        plot_final_comparison(all_results, save_dir=CONFIG["results_dir"])

        out_path = f"{CONFIG['results_dir']}/all_results_summary.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\n  All results saved to: {out_path}")

    print("\n\nExperiment complete!")


if __name__ == "__main__":
    main()
