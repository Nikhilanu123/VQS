"""
Fair Classical Baseline on 160-sample subset
----------------------------------------------
Trains classical models on the SAME 160 samples (80 fraud, 80 legit)
that VQC models use, enabling fair comparison.
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, confusion_matrix
from xgboost import XGBClassifier
from imblearn.metrics import geometric_mean_score


def compute_metrics(y_true, y_pred, y_prob, model_name="Model"):
    """Compute all relevant metrics as a dict."""
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


def run_fair_classical_baseline(X_train_small, y_train_small, X_test, y_test,
                                 save_dir="results"):
    """
    Train classical models on the SAME 160-sample subset (80 fraud, 80 legit)
    that VQC uses, for a fair comparison.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("FAIR CLASSICAL BASELINES (160 training samples)")
    print("=" * 60)
    print(f"  Train set: {len(X_train_small)} samples (fraud={y_train_small.sum()}, legit={(y_train_small==0).sum()})")
    print(f"  Test set:  {len(X_test)} samples")

    # Logistic Regression
    print("\n  Training Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_small, y_train_small)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    lr_res = compute_metrics(y_test, y_pred_lr, y_prob_lr, "LR (160 samples)")

    # Random Forest
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train_small, y_train_small)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    rf_res = compute_metrics(y_test, y_pred_rf, y_prob_rf, "RF (160 samples)")

    # XGBoost
    print("\n  Training XGBoost...")
    scale_pos = (y_train_small == 0).sum() / (y_train_small == 1).sum()
    xgb = XGBClassifier(n_estimators=300, scale_pos_weight=scale_pos,
                        max_depth=5, learning_rate=0.05,
                        use_label_encoder=False, eval_metric='aucpr',
                        random_state=42, n_jobs=-1)
    xgb.fit(X_train_small, y_train_small)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    xgb_res = compute_metrics(y_test, y_pred_xgb, y_prob_xgb, "XGB (160 samples)")

    all_results = [lr_res, rf_res, xgb_res]

    # Save results to JSON
    out_path = f"{save_dir}/classical_results_fair_160.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return all_results
