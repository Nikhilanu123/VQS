"""
Phase 4 - IBM Quantum Hardware Validation
------------------------------------------
Runs a small final validation on real IBM Quantum hardware.
Strategy: train on simulator, submit inference-only job to IBM.
Uses your 10 free minutes wisely -- small circuit, few shots.
"""

import numpy as np
import os
import json
import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def run_ibm_validation(trained_trainer, X_test_small, y_test_small,
                        ibm_token: str, backend_name: str = "least_busy",
                        shots: int = 1024, save_dir: str = "results"):
    """
    Submits the trained QCS-VQC circuit to IBM Quantum for evaluation.
    
    Args:
        trained_trainer  : Fitted QCSVQCTrainer instance (trained on simulator)
        X_test_small     : Small test set (keep to 20-50 samples to save time)
        y_test_small     : True labels for X_test_small
        ibm_token        : Your IBM Quantum API token
        backend_name     : 'least_busy' auto-selects least busy device
        shots            : Number of measurement shots (1024 recommended)
        save_dir         : Where to save results
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        from qiskit_ibm_runtime import Session
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except ImportError:
        print("[ERROR] qiskit-ibm-runtime not installed. Run: pip install qiskit-ibm-runtime")
        return None

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("IBM QUANTUM HARDWARE VALIDATION")
    print("="*60)
    print("  Saving IBM token and connecting...")

    # Save and load token
    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token=ibm_token,
        overwrite=True
    )
    service = QiskitRuntimeService(channel="ibm_quantum")

    # Select backend
    if backend_name == "least_busy":
        backend = service.least_busy(
            min_num_qubits=trained_trainer.num_qubits,
            operational=True,
            simulator=False
        )
        print(f"  Selected backend : {backend.name}")
    else:
        backend = service.backend(backend_name)

    print(f"  Qubit count      : {backend.num_qubits}")
    print(f"  Test samples     : {len(X_test_small)}")
    print(f"  Shots per circuit: {shots}")
    print(f"\n  Transpiling circuits...")

    # Build and bind circuits for each test sample
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    circuits = []
    for x in X_test_small:
        fp = trained_trainer._feature_map.parameters
        vp = trained_trainer._ansatz.parameters
        param_dict = dict(zip(fp, x))
        param_dict.update(dict(zip(vp, trained_trainer.opt_params)))
        bound = trained_trainer._circuit.assign_parameters(param_dict)
        bound.measure_all()
        transpiled = pm.run(bound)
        circuits.append(transpiled)

    print(f"  Submitting {len(circuits)} circuits to {backend.name}...")
    print("  (This uses your IBM free minutes -- expect 2-8 mins queue time)")

    hardware_probs = []
    with Session(backend=backend) as session:
        sampler = Sampler(session=session)
        job = sampler.run(circuits, shots=shots)
        print(f"  Job ID: {job.job_id()} -- waiting for results...")
        result = job.result()

        for i, pub_result in enumerate(result):
            counts = pub_result.data.meas.get_counts()
            n_total = sum(counts.values())
            # Count |1> outcomes on qubit 0 as "fraud probability"
            n_fraud = sum(v for k, v in counts.items() if k[-1] == '1')
            prob = n_fraud / n_total
            hardware_probs.append(prob)

    hardware_probs = np.array(hardware_probs)
    y_pred_hw = (hardware_probs >= 0.5).astype(int)

    # Compute metrics
    from src.qcs_vqc import compute_metrics
    metrics = compute_metrics(y_test_small, y_pred_hw, hardware_probs,
                               f"QCS-VQC on {backend.name} (IBM Hardware)")
    metrics['backend']  = backend.name
    metrics['shots']    = shots
    metrics['job_id']   = job.job_id()

    out_path = f"{save_dir}/ibm_hardware_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\n  IBM results saved to: {out_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR vs HARDWARE COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_simulator_vs_hardware(sim_metrics, hw_metrics, save_dir="results"):
    """Side-by-side bar chart comparing simulator and hardware performance."""
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    metric_keys = ['auc_roc', 'f1', 'mcc', 'g_mean']
    labels = ['AUC-ROC', 'F1', 'MCC', 'G-Mean']

    sim_vals = [sim_metrics.get(k, 0) for k in metric_keys]
    hw_vals  = [hw_metrics.get(k, 0)  for k in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, sim_vals, width, label='Simulator (Aer)', color='steelblue')
    ax.bar(x + width/2, hw_vals,  width, label=f'IBM Hardware ({hw_metrics.get("backend","IBM")})',
           color='crimson', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_title("QCS-VQC: Simulator vs IBM Quantum Hardware")
    ax.legend()
    plt.tight_layout()
    save_path = f"{save_dir}/simulator_vs_hardware.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Comparison plot saved to: {save_path}")
