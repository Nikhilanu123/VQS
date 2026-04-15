"""
Phase 3 - Ablation Studies
----------------------------
Systematically removes each component of QCS-VQC to prove
each contribution is necessary.

Ablation 1: No cost-sensitivity         → Standard VQC + ZNE
Ablation 2: No ZNE                      → QCS-VQC without noise mitigation
Ablation 3: No layerwise training       → QCS-VQC full depth at once
Ablation 4: Vary qubit count            → 4, 6, 8, 10 qubits
Ablation 5: Vary imbalance ratio        → Synthetic imbalance experiment
"""

import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.qcs_vqc import QCSVQCTrainer, compute_metrics
from src.data_preprocessing import quantum_subsample


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION 1 & 2 & 3 -- Component removal
# ─────────────────────────────────────────────────────────────────────────────

def run_component_ablations(X_train, y_train, X_test, y_test,
                             num_qubits=8, max_iter=60, save_dir="results",
                             dataset_fraud_ratio=None):
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/ablation_component_results.json"

    # Resume: load any previously completed configs
    completed = []
    completed_names = set()
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                completed = json.load(f)
            completed_names = {r['model'] for r in completed}
            print(f"  [RESUME] Found {len(completed)} completed ablation configs.")
        except (json.JSONDecodeError, KeyError):
            completed = []

    results = list(completed)

    configs = [
        {
            "name"      : "QCS-VQC (Full -- Proposed)",
            "use_noise" : False,
            "use_zne"   : False,
            "layerwise" : True,
        },
        {
            "name"      : "Ablation: No Cost-Sensitivity",
            "use_noise" : False,
            "use_zne"   : False,
            "layerwise" : True,
            "equal_weights": True,   # force fraud_w = legit_w = 1
        },
        {
            "name"      : "Ablation: No Layerwise Training",
            "use_noise" : False,
            "use_zne"   : False,
            "layerwise" : False,
        },
        {
            "name"      : "Ablation: Under Noise (no ZNE)",
            "use_noise" : True,
            "use_zne"   : False,
            "layerwise" : True,
        },
        {
            "name"      : "Ablation: Under Noise + ZNE",
            "use_noise" : True,
            "use_zne"   : True,
            "layerwise" : True,
        },
    ]

    print("\n" + "="*60)
    print("ABLATION STUDY -- COMPONENT REMOVAL")
    print("="*60)

    for cfg in configs:
        if cfg['name'] in completed_names:
            print(f"\n--- {cfg['name']} --- [SKIP - already done]")
            continue

        print(f"\n--- {cfg['name']} ---")
        trainer = QCSVQCTrainer(
            num_qubits=num_qubits,
            reps=2,
            max_iter=max_iter,
            use_noise=cfg['use_noise'],
            use_zne=cfg['use_zne'],
            dataset_fraud_ratio=None if cfg.get('equal_weights') else dataset_fraud_ratio,
            n_restarts=1   # single restart for ablations (speed)
        )

        trainer.fit(X_train, y_train, layerwise=cfg['layerwise'])
        y_pred = trainer.predict(X_test)
        y_prob = trainer.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_prob, cfg['name'])
        m['loss_history'] = trainer.loss_history
        results.append(m)

        # Incremental save after each config (crash-safe)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"  [CHECKPOINT] Saved {len(results)}/{len(configs)} configs")

    print(f"\n  Ablation results saved to: {out_path}")

    _plot_ablation_bar(results, save_path=f"{save_dir}/ablation_component_bar.png")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION 4 -- Qubit count sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_qubit_sweep(X_train_full, y_train_full, X_test_full, y_test_full,
                    qubit_counts=(4, 6, 8), max_iter=50, save_dir="results",
                    raw_df=None, dataset_fraud_ratio=None):
    """
    Train QCS-VQC for different qubit counts (= different PCA dimensionalities).
    Must reprocess data for each qubit count.
    """
    from src.data_preprocessing import preprocess, split_data
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/ablation_qubit_sweep.json"

    # Resume: load previously completed qubit counts
    completed = []
    completed_qubits = set()
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                completed = json.load(f)
            completed_qubits = {r['num_qubits'] for r in completed}
            print(f"  [RESUME] Found {len(completed)} completed qubit configs.")
        except (json.JSONDecodeError, KeyError):
            completed = []

    results = list(completed)

    print("\n" + "="*60)
    print("ABLATION -- QUBIT COUNT SWEEP")
    print("="*60)

    for n_q in qubit_counts:
        if n_q in completed_qubits:
            print(f"\n--- {n_q} Qubits --- [SKIP - already done]")
            continue

        print(f"\n--- {n_q} Qubits ---")
        if raw_df is not None:
            X, y = preprocess(raw_df, pca_components=n_q)
            X_tr, X_te, y_tr, y_te = split_data(X, y)
        else:
            X_tr = X_train_full[:, :n_q]
            X_te = X_test_full[:, :n_q]
            y_tr = y_train_full
            y_te = y_test_full

        X_q, y_q = quantum_subsample(X_tr, y_tr, n_fraud=80, n_legit=80)
        # Small test set for quantum eval
        X_q_te, y_q_te = quantum_subsample(X_te, y_te, n_fraud=min(49, (y_te==1).sum()),
                                            n_legit=150)

        trainer = QCSVQCTrainer(
            num_qubits=n_q, reps=2, max_iter=max_iter,
            use_noise=False, use_zne=False,
            dataset_fraud_ratio=dataset_fraud_ratio,
            n_restarts=1
        )
        trainer.fit(X_q, y_q, layerwise=True)
        y_pred = trainer.predict(X_q_te)
        y_prob = trainer.predict_proba(X_q_te)
        m = compute_metrics(y_q_te, y_pred, y_prob, f"QCS-VQC ({n_q} qubits)")
        m['num_qubits'] = n_q
        results.append(m)

        # Incremental save after each qubit config
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"  [CHECKPOINT] Saved {len(results)}/{len(qubit_counts)} qubit configs")

    _plot_qubit_sweep(results, save_path=f"{save_dir}/ablation_qubit_sweep.png")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION 5 -- Imbalance ratio sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_imbalance_sweep(X_q_train, y_q_train, X_test, y_test,
                        ratios=(0.1, 0.2, 0.3, 0.5),
                        num_qubits=8, max_iter=50, save_dir="results"):
    """
    Synthetically vary the class imbalance ratio in training data
    to show that QCS-VQC degrades gracefully vs Standard VQC.
    """
    os.makedirs(save_dir, exist_ok=True)
    from src.qcs_vqc import QCSVQCTrainer
    out_path = f"{save_dir}/ablation_imbalance_sweep.json"

    # Resume: load previously completed ratios
    results_qcs = []
    results_std = []
    completed_ratios = set()
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                saved = json.load(f)
            results_qcs = saved.get("qcs", [])
            results_std = saved.get("std", [])
            completed_ratios = {r['ratio'] for r in results_qcs}
            print(f"  [RESUME] Found {len(completed_ratios)} completed imbalance ratios.")
        except (json.JSONDecodeError, KeyError):
            results_qcs, results_std = [], []

    n_fraud_avail = (y_q_train == 1).sum()

    print("\n" + "="*60)
    print("ABLATION -- IMBALANCE RATIO SWEEP")
    print("="*60)

    for ratio in ratios:
        if ratio in completed_ratios:
            print(f"\n  Ratio 1:{int(1/ratio)} -- [SKIP - already done]")
            continue

        n_legit = int(n_fraud_avail / ratio)
        print(f"\n  Ratio 1:{int(1/ratio)} -- fraud={n_fraud_avail}, legit={n_legit}")
        X_s, y_s = quantum_subsample(X_q_train, y_q_train,
                                      n_fraud=n_fraud_avail,
                                      n_legit=min(n_legit, (y_q_train==0).sum()))

        # QCS-VQC
        t = QCSVQCTrainer(num_qubits=num_qubits, max_iter=max_iter,
                           use_noise=False, use_zne=False,
                           dataset_fraud_ratio=0.00173, n_restarts=1)
        t.fit(X_s, y_s, layerwise=True)
        m_qcs = compute_metrics(y_test, t.predict(X_test), t.predict_proba(X_test),
                                  f"QCS-VQC ratio={ratio:.1f}")
        m_qcs['ratio'] = ratio
        results_qcs.append(m_qcs)

        # Standard VQC (no cost-sensitivity, no ZNE)
        t_std = QCSVQCTrainer(num_qubits=num_qubits, max_iter=max_iter,
                               use_noise=False, use_zne=False, n_restarts=1)
        t_std.fit(X_s, y_s, layerwise=False)
        m_std = compute_metrics(y_test, t_std.predict(X_test), t_std.predict_proba(X_test),
                                  f"Std VQC ratio={ratio:.1f}")
        m_std['ratio'] = ratio
        results_std.append(m_std)

        # Incremental save after each ratio
        with open(out_path, "w") as f:
            json.dump({"qcs": results_qcs, "std": results_std}, f, indent=2, default=float)
        print(f"  [CHECKPOINT] Saved {len(results_qcs)}/{len(ratios)} imbalance ratios")

    _plot_imbalance_sweep(results_qcs, results_std,
                           save_path=f"{save_dir}/ablation_imbalance_sweep.png")
    return results_qcs, results_std


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_ablation_bar(results, save_path):
    metrics = ['auc_roc', 'f1', 'mcc', 'g_mean']
    labels  = [r['model'] for r in results]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ['crimson', 'steelblue', 'darkorange', 'green', 'purple']
    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [res[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=res['model'][:40], color=color, alpha=0.8)

    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(['AUC-ROC', 'F1', 'MCC', 'G-Mean'])
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_title("Ablation Study -- Component Contribution")
    ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Ablation bar chart saved to: {save_path}")


def _plot_qubit_sweep(results, save_path):
    qubits  = [r['num_qubits'] for r in results]
    auc_roc = [r['auc_roc'] for r in results]
    f1      = [r['f1'] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(qubits, auc_roc, 'o-', color='steelblue', label='AUC-ROC')
    ax.plot(qubits, f1,      's--', color='crimson',  label='F1-Score')
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Score")
    ax.set_title("QCS-VQC Performance vs Qubit Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Qubit sweep plot saved to: {save_path}")


def _plot_imbalance_sweep(results_qcs, results_std, save_path):
    ratios  = [r['ratio'] for r in results_qcs]
    qcs_mcc = [r['mcc'] for r in results_qcs]
    std_mcc = [r['mcc'] for r in results_std]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ratios, qcs_mcc, 'o-', color='crimson',   label='QCS-VQC (Proposed)')
    ax.plot(ratios, std_mcc, 's--', color='steelblue', label='Standard VQC')
    ax.set_xlabel("Fraud Ratio in Training Data")
    ax.set_ylabel("MCC")
    ax.set_title("MCC vs Class Imbalance Ratio")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Imbalance sweep plot saved to: {save_path}")
