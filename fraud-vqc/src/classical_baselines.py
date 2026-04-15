"""
Phase 1 - Classical Baselines
-------------------------------
Trains Logistic Regression, Random Forest, and XGBoost on the full feature set.
Reports AUC-ROC, AUC-PRC, MCC, F1, G-Mean for comparison with quantum models.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report
)
from xgboost import XGBClassifier
from imblearn.metrics import geometric_mean_score


# -----------------------------------------------------------------------------
# METRICS HELPER
# -----------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob, model_name="Model"):
    """Compute and return all relevant metrics as a dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "model"      : model_name,
        "auc_roc"    : roc_auc_score(y_true, y_prob),
        "auc_prc"    : average_precision_score(y_true, y_prob),
        "f1"         : f1_score(y_true, y_pred),
        "mcc"        : matthews_corrcoef(y_true, y_pred),
        "g_mean"     : geometric_mean_score(y_true, y_pred),
        "precision"  : tp / (tp + fp + 1e-9),
        "recall"     : tp / (tp + fn + 1e-9),
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
# CLASSIFIERS
# -----------------------------------------------------------------------------

def run_logistic_regression(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_pred, y_prob, "Logistic Regression"), clf


def run_random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_pred, y_prob, "Random Forest"), clf


def run_xgboost(X_train, y_train, X_test, y_test):
    # scale_pos_weight handles class imbalance natively in XGBoost
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    clf = XGBClassifier(
        n_estimators=300, scale_pos_weight=scale_pos,
        max_depth=5, learning_rate=0.05,
        use_label_encoder=False, eval_metric='aucpr',
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_pred, y_prob, "XGBoost"), clf


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

def plot_roc_curves(results_list, X_test, y_test, models_list,
                   save_path="results/roc_curves_classical.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['steelblue', 'darkorange', 'green']
    for (res, model), color in zip(zip(results_list, models_list), colors):
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0].plot(fpr, tpr, color=color,
                     label=f"{res['model']} (AUC={res['auc_roc']:.3f})")

        # PRC
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        axes[1].plot(rec, prec, color=color,
                     label=f"{res['model']} (AP={res['auc_prc']:.3f})")

    axes[0].plot([0,1],[0,1],'k--', alpha=0.4)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.suptitle("Classical Baselines — Credit Card Fraud Detection", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  ROC/PRC plot saved to: {save_path}")


def plot_metrics_bar(all_results, save_path="results/metrics_comparison.png"):
    """Bar chart comparing key metrics across all models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_to_plot = ['auc_roc', 'auc_prc', 'f1', 'mcc', 'g_mean']
    labels = [r['model'] for r in all_results]

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(all_results)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, res in enumerate(all_results):
        vals = [res[m] for m in metrics_to_plot]
        ax.bar(x + i * width, vals, width, label=res['model'])

    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(['AUC-ROC', 'AUC-PRC', 'F1', 'MCC', 'G-Mean'])
    ax.set_ylim(0, 1.05)
    ax.set_title("Classical Baselines — Metric Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Metrics bar chart saved to: {save_path}")


# -----------------------------------------------------------------------------
# RUNNER
# -----------------------------------------------------------------------------

def run_all_classical_baselines(X_train, y_train, X_test, y_test,
                                 save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("CLASSICAL BASELINES")
    print("=" * 60)

    lr_res, lr_clf   = run_logistic_regression(X_train, y_train, X_test, y_test)
    rf_res, rf_clf   = run_random_forest(X_train, y_train, X_test, y_test)
    xgb_res, xgb_clf = run_xgboost(X_train, y_train, X_test, y_test)

    all_results = [lr_res, rf_res, xgb_res]
    all_models  = [lr_clf, rf_clf, xgb_clf]

    plot_roc_curves(all_results, X_test, y_test, all_models,
                    save_path=f"{save_dir}/roc_curves_classical.png")
    plot_metrics_bar(all_results,
                     save_path=f"{save_dir}/metrics_comparison_classical.png")

    # Save results to JSON
    out_path = f"{save_dir}/classical_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to : {out_path}")

    return all_results, all_models


# -----------------------------------------------------------------------------
# QUICK TEST
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.data_preprocessing import load_data, preprocess, split_data

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    run_all_classical_baselines(X_train, y_train, X_test, y_test)
