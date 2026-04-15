"""
Generate Comparison Tables and Summary Report
-----------------------------------------------
Combines all results (classical, VQC, ablations, etc.) into publication-ready tables.
"""

import json
import os
import pandas as pd
from tabulate import tabulate


def load_json(filepath):
    """Safely load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def generate_all_tables(results_dir="results"):
    """Generate and print all comparison tables."""
    
    print("\n" + "="*90)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*90)
    
    # Load all results
    classical_full = load_json(f"{results_dir}/classical_results.json")
    classical_fair = load_json(f"{results_dir}/classical_results_fair_160.json")
    vqc_std = load_json(f"{results_dir}/vqc_standard_results.json")
    qcs_vqc = load_json(f"{results_dir}/qcs_vqc_results.json")
    ablation = load_json(f"{results_dir}/ablation_component_results.json")
    qubit_sweep = load_json(f"{results_dir}/ablation_qubit_sweep.json")
    imbalance = load_json(f"{results_dir}/ablation_imbalance_sweep.json")
    
    # ─── TABLE 1: Classical Baselines Comparison ───────────────────────────────────
    print("\n" + "-"*90)
    print("TABLE 1: Classical Baselines Comparison")
    print("-"*90)
    
    if classical_full and classical_fair:
        classical_data = []
        for model in classical_full:
            classical_data.append({
                "Model": model['model'] + " (227k samples)",
                "AUC-ROC": f"{model['auc_roc']:.4f}",
                "AUC-PRC": f"{model['auc_prc']:.4f}",
                "F1": f"{model['f1']:.4f}",
                "MCC": f"{model['mcc']:.4f}",
                "G-Mean": f"{model['g_mean']:.4f}"
            })
        for model in classical_fair:
            classical_data.append({
                "Model": model['model'].split('(')[0].strip() + " (160 samples)",
                "AUC-ROC": f"{model['auc_roc']:.4f}",
                "AUC-PRC": f"{model['auc_prc']:.4f}",
                "F1": f"{model['f1']:.4f}",
                "MCC": f"{model['mcc']:.4f}",
                "G-Mean": f"{model['g_mean']:.4f}"
            })
        print(tabulate(classical_data, headers="keys", tablefmt="grid"))
    
    # ─── TABLE 2: Quantum Methods Comparison ────────────────────────────────────────
    print("\n" + "-"*90)
    print("TABLE 2: Quantum Methods (160 training samples)")
    print("-"*90)
    
    quantum_data = []
    if vqc_std:
        vqc_std_dict = vqc_std if isinstance(vqc_std, dict) else vqc_std[0]
        quantum_data.append({
            "Model": "Standard VQC (COBYLA)",
            "AUC-ROC": f"{vqc_std_dict['auc_roc']:.4f}",
            "AUC-PRC": f"{vqc_std_dict['auc_prc']:.4f}",
            "F1": f"{vqc_std_dict['f1']:.4f}",
            "MCC": f"{vqc_std_dict['mcc']:.4f}",
            "G-Mean": f"{vqc_std_dict['g_mean']:.4f}",
            "Train Time (s)": f"{vqc_std_dict.get('train_time_s', 'N/A'):.0f}" if isinstance(vqc_std_dict.get('train_time_s'), (int, float)) else "N/A"
        })
    if qcs_vqc:
        qcs_dict = qcs_vqc if isinstance(qcs_vqc, dict) else qcs_vqc[0]
        quantum_data.append({
            "Model": "QCS-VQC (SPSA, layerwise)",
            "AUC-ROC": f"{qcs_dict['auc_roc']:.4f}",
            "AUC-PRC": f"{qcs_dict['auc_prc']:.4f}",
            "F1": f"{qcs_dict['f1']:.4f}",
            "MCC": f"{qcs_dict['mcc']:.4f}",
            "G-Mean": f"{qcs_dict['g_mean']:.4f}",
            "Train Time (s)": f"{qcs_dict.get('train_time_s', 'N/A'):.0f}" if isinstance(qcs_dict.get('train_time_s'), (int, float)) else "N/A"
        })
    
    if quantum_data:
        print(tabulate(quantum_data, headers="keys", tablefmt="grid"))
    
    # ─── TABLE 3: Component Ablation Study ──────────────────────────────────────────
    print("\n" + "-"*90)
    print("TABLE 3: Component Ablation Study (160 training samples)")
    print("-"*90)
    
    if ablation:
        ablation_data = []
        for model in ablation:
            ablation_data.append({
                "Configuration": model['model'],
                "AUC-ROC": f"{model['auc_roc']:.4f}",
                "F1": f"{model['f1']:.4f}",
                "MCC": f"{model['mcc']:.4f}",
                "G-Mean": f"{model['g_mean']:.4f}",
                "TP": model['tp'],
                "TN": model['tn'],
                "FP": model['fp'],
                "FN": model['fn']
            })
        print(tabulate(ablation_data, headers="keys", tablefmt="grid"))
    
    # ─── TABLE 4: Qubit Scalability ────────────────────────────────────────────────
    print("\n" + "-"*90)
    print("TABLE 4: Qubit Scalability Study")
    print("-"*90)
    
    if qubit_sweep:
        qubit_data = []
        for model in qubit_sweep:
            qubit_data.append({
                "Qubits": model['num_qubits'],
                "AUC-ROC": f"{model['auc_roc']:.4f}",
                "AUC-PRC": f"{model['auc_prc']:.4f}",
                "F1": f"{model['f1']:.4f}",
                "MCC": f"{model['mcc']:.4f}",
                "G-Mean": f"{model['g_mean']:.4f}"
            })
        print(tabulate(qubit_data, headers="keys", tablefmt="grid"))
    
    # ─── TABLE 5: Imbalance Ratio Study ────────────────────────────────────────────
    print("\n" + "-"*90)
    print("TABLE 5: Class Imbalance Robustness")
    print("-"*90)
    
    if imbalance and isinstance(imbalance, dict):
        imb_data = []
        qcs_list = imbalance.get('qcs', [])
        std_list = imbalance.get('std', [])
        
        for qcs_model, std_model in zip(qcs_list, std_list):
            ratio = qcs_model['ratio']
            imb_data.append({
                "Ratio": f"1:{int(1/ratio)}",
                "QCS-VQC AUC": f"{qcs_model['auc_roc']:.4f}",
                "QCS-VQC F1": f"{qcs_model['f1']:.4f}",
                "Std VQC AUC": f"{std_model['auc_roc']:.4f}",
                "Std VQC F1": f"{std_model['f1']:.4f}"
            })
        
        if imb_data:
            print(tabulate(imb_data, headers="keys", tablefmt="grid"))
    
    # ─── Summary Metrics ───────────────────────────────────────────────────────────
    print("\n" + "-"*90)
    print("KEY FINDINGS")
    print("-"*90)
    
    findings = {
        "Metric": [
            "Classical XGB (227k samples) top AUC",
            "Classical XGB (160 samples) AUC",
            "Standard VQC (COBYLA) AUC",
            "QCS-VQC (SPSA) AUC",
            "Best Ablation AUC (No Cost-Sens)",
            "Layerwise Improvement",
            "Qubit Count (qubits)",
            "Imbalance Ratio (best)"
        ],
        "Value": [
            f"{classical_full[2]['auc_roc']:.4f}" if classical_full else "N/A",
            f"{classical_fair[2]['auc_roc']:.4f}" if classical_fair else "N/A",
            f"{vqc_std_dict['auc_roc']:.4f}" if vqc_std else "N/A",
            f"{qcs_dict['auc_roc']:.4f}" if qcs_vqc else "N/A",
            f"{ablation[1]['auc_roc']:.4f}" if ablation else "N/A",
            f"+{(ablation[1]['auc_roc'] - ablation[2]['auc_roc']):.4f}" if ablation else "N/A",
            "8 (best: 8→6→4)",
            "1:2 (most robust)"
        ]
    }
    
    print(tabulate(findings, headers="keys", tablefmt="grid"))
    
    print("\n" + "="*90)


if __name__ == "__main__":
    generate_all_tables()
