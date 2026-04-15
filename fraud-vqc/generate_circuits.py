"""
Generate publication-quality quantum circuit diagrams:
  1. ZZFeatureMap (8 qubits, reps=2)
  2. RealAmplitudes ansatz (8 qubits, reps=2, linear entanglement)
  3. Full QCS-VQC circuit (feature map + ansatz combined)
  4. Small illustrative circuits (4 qubits) for paper clarity
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit import QuantumCircuit


def save_circuit(circuit, filename, title=None, fold=-1, scale=0.7):
    """Draw circuit using matplotlib and save as high-DPI PNG."""
    os.makedirs("results", exist_ok=True)
    fig = circuit.draw(
        output='mpl',
        fold=fold,
        scale=scale,
        style={
            'backgroundcolor': '#FFFFFF',
            'textcolor': '#000000',
            'gatefacecolor': '#E8F4FD',
            'gatetextcolor': '#1A1A2E',
            'barrierfacecolor': '#CCCCCC',
            'fontsize': 12,
            'subfontsize': 9,
        }
    )
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(f"results/{filename}", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: results/{filename}")


def generate_all_diagrams():
    print("=" * 60)
    print("GENERATING QUANTUM CIRCUIT DIAGRAMS")
    print("=" * 60)

    # ─── 1. ZZFeatureMap (small, 4-qubit for clarity) ──────────────────
    print("\n[1/6] ZZFeatureMap (4 qubits, reps=1) — illustrative")
    zz_small = ZZFeatureMap(feature_dimension=4, reps=1, entanglement='linear')
    save_circuit(zz_small, "zzfeaturemap_4q_reps1.png",
                 title="ZZFeatureMap (4 qubits, reps=1, linear)", fold=40)

    # ─── 2. ZZFeatureMap (small, 4-qubit, reps=2) ─────────────────────
    print("[2/6] ZZFeatureMap (4 qubits, reps=2) — illustrative")
    zz_small2 = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
    save_circuit(zz_small2, "zzfeaturemap_4q_reps2.png",
                 title="ZZFeatureMap (4 qubits, reps=2, linear)", fold=40)

    # ─── 3. Full ZZFeatureMap (8-qubit, reps=2 — actual experiment) ───
    print("[3/6] ZZFeatureMap (8 qubits, reps=2) — actual experiment")
    zz_full = ZZFeatureMap(feature_dimension=8, reps=2, entanglement='linear')
    save_circuit(zz_full, "zzfeaturemap_8q_reps2.png",
                 title="ZZFeatureMap (8 qubits, reps=2, linear)",
                 fold=30, scale=0.6)

    # ─── 4. RealAmplitudes ansatz (4-qubit for clarity) ────────────────
    print("[4/6] RealAmplitudes ansatz (4 qubits, reps=2) — illustrative")
    ansatz_small = RealAmplitudes(num_qubits=4, reps=2, entanglement='linear')
    save_circuit(ansatz_small, "realamplitudes_4q_reps2.png",
                 title="RealAmplitudes Ansatz (4 qubits, reps=2, linear)", fold=40)

    # ─── 5. Full ansatz (8-qubit, reps=2 — actual experiment) ─────────
    print("[5/6] RealAmplitudes ansatz (8 qubits, reps=2) — actual experiment")
    ansatz_full = RealAmplitudes(num_qubits=8, reps=2, entanglement='linear')
    save_circuit(ansatz_full, "realamplitudes_8q_reps2.png",
                 title="RealAmplitudes Ansatz (8 qubits, reps=2, linear)",
                 fold=30, scale=0.6)

    # ─── 6. Complete QCS-VQC circuit (4-qubit for paper figure) ───────
    print("[6/6] Complete QCS-VQC circuit (4 qubits) — paper figure")
    fm = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
    ans = RealAmplitudes(num_qubits=4, reps=2, entanglement='linear')
    full_circuit = QuantumCircuit(4)
    full_circuit.compose(fm, inplace=True)
    full_circuit.barrier()
    full_circuit.compose(ans, inplace=True)
    save_circuit(full_circuit, "qcsvqc_full_circuit_4q.png",
                 title="QCS-VQC Circuit: ZZFeatureMap + RealAmplitudes (4 qubits)",
                 fold=30, scale=0.65)

    # ─── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ALL CIRCUIT DIAGRAMS GENERATED")
    print(f"{'=' * 60}")
    print("\nFiles created:")
    print("  results/zzfeaturemap_4q_reps1.png    — ZZFeatureMap (4q, r=1)")
    print("  results/zzfeaturemap_4q_reps2.png    — ZZFeatureMap (4q, r=2)")
    print("  results/zzfeaturemap_8q_reps2.png    — ZZFeatureMap (8q, r=2)")
    print("  results/realamplitudes_4q_reps2.png  — RealAmplitudes (4q, r=2)")
    print("  results/realamplitudes_8q_reps2.png  — RealAmplitudes (8q, r=2)")
    print("  results/qcsvqc_full_circuit_4q.png   — Full QCS-VQC (4q)")
    print("\nRecommended for paper:")
    print("  Figure 2a: zzfeaturemap_4q_reps2.png")
    print("  Figure 2b: realamplitudes_4q_reps2.png")
    print("  Figure 2c: qcsvqc_full_circuit_4q.png")
    print("  Supplementary: 8-qubit versions for completeness")


if __name__ == "__main__":
    generate_all_diagrams()
