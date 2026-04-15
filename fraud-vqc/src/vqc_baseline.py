"""
Phase 2 - Standard VQC Baseline
---------------------------------
Implements a standard Variational Quantum Classifier using Qiskit Machine Learning.
Uses ZZFeatureMap + RealAmplitudes ansatz on 8-qubit PCA-compressed features.
This is the baseline to beat with QCS-VQC.
"""

import numpy as np
import os
import json
import warnings
import time
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore")

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import SPSA
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, confusion_matrix
)
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# METRICS HELPER
# -----------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob, model_name="Model"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "model"     : model_name,
        "auc_roc"   : roc_auc_score(y_true, y_prob),
        "auc_prc"   : average_precision_score(y_true, y_prob),
        "f1"        : f1_score(y_true, y_pred, zero_division=0),
        "mcc"       : matthews_corrcoef(y_true, y_pred),
        "g_mean"    : geometric_mean_score(y_true, y_pred),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }
    print(f"\n{'-'*50}")
    print(f"  {model_name}")
    print(f"{'-'*50}")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"  AUC-PRC  : {metrics['auc_prc']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  MCC      : {metrics['mcc']:.4f}")
    print(f"  G-Mean   : {metrics['g_mean']:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return metrics


# -----------------------------------------------------------------------------
# STANDARD VQC
# -----------------------------------------------------------------------------

# Training progress tracker
_loss_history = []


def build_vqc(num_qubits: int = 8, reps: int = 2, max_iter: int = 100):
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
    ansatz      = RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement='linear')
    optimizer   = SPSA(maxiter=max_iter)
    sampler     = StatevectorSampler()
    _loss_history.clear()
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=_training_callback
    )
    return vqc


def _training_callback(weights, obj_func_eval):
    _loss_history.append(float(obj_func_eval))
    step = len(_loss_history)
    if step % 10 == 0:
        print(f"    Step {step:4d} | Loss: {obj_func_eval:.6f}")


def run_standard_vqc(X_train, y_train, X_test, y_test,
                     num_qubits=8, reps=2, max_iter=100,
                     save_dir="results"):
    global _loss_history
    _loss_history = []

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("STANDARD VQC BASELINE")
    print("=" * 60)
    print(f"  Qubits      : {num_qubits}")
    print(f"  Ansatz reps : {reps}")
    print(f"  Max iter    : {max_iter}")
    print(f"  Train size  : {len(X_train)}  (fraud={y_train.sum()})")
    print(f"  Test size   : {len(X_test)}   (fraud={y_test.sum()})")
    print(f"\n  Training...")

    vqc = build_vqc(num_qubits=num_qubits, reps=reps, max_iter=max_iter)

    t0 = time.time()
    vqc.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"\n  Training time : {train_time:.1f}s")

    print(f"  Running inference on {len(X_test)} test samples...")
    t1 = time.time()
    y_pred = vqc.predict(X_test)
    try:
        y_prob = vqc.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = y_pred.astype(float)
    infer_time = time.time() - t1
    print(f"  Inference time : {infer_time:.1f}s")

    metrics = compute_metrics(y_test, y_pred, y_prob, "Standard VQC")
    metrics['train_time_s']  = round(train_time, 2)
    metrics['infer_time_s']  = round(infer_time, 2)
    metrics['loss_history'] = _loss_history

    # Save results
    out_path = f"{save_dir}/vqc_standard_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved to : {out_path}")

    _plot_loss_curve(_loss_history, "Standard VQC",
                     f"{save_dir}/loss_curve_standard_vqc.png")

    return metrics, vqc


def _plot_loss_curve(loss_history, title, save_path):
    if not loss_history:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, color='steelblue')
    plt.title(f"Training Loss — {title}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Loss curve saved to : {save_path}")


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.data_preprocessing import (
        load_data, preprocess, split_data, quantum_subsample
    )

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
    df = load_data(DATA_PATH)
    X, y = preprocess(df, pca_components=8)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_q_train, y_q_train = quantum_subsample(X_train, y_train, n_fraud=100, n_legit=100)
    X_q_test,  y_q_test  = quantum_subsample(X_test,  y_test,  n_fraud=49,  n_legit=150)

    run_standard_vqc(X_q_train, y_q_train, X_q_test, y_q_test,
                     num_qubits=8, reps=2, max_iter=50)
