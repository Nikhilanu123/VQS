# Codebase Overview

## Directory Structure

### Root Files
- **`main.py`** — Master experiment runner orchestrating all 4 phases
- **`run_fair_classical.py`** — Runner for fair classical baseline on 160 samples + summary table generation
- **`creditcard.csv`** — Kaggle credit card fraud dataset (284,807 samples, 492 fraud)
- **`requirements.txt`** — Python dependencies (qiskit, sklearn, xgboost, etc.)

### `/src/` — Core Implementation

#### Data Processing
- **`data_preprocessing.py`** (176 lines)
  - `load_data()` — Loads creditcard.csv, handles missing values
  - `preprocess()` — PCA dimensionality reduction (default 8 components)
  - `split_data()` — 80/20 train-test split
  - `quantum_subsample()` — Balanced subsampling (e.g., 80 fraud + 80 legit)
  - `plot_class_distribution()` — Visualizes imbalance ratio

#### Classical Methods
- **`classical_baselines.py`** (212 lines)
  - `run_logistic_regression()` — LR with class_weight='balanced'
  - `run_random_forest()` — RF with 200 estimators
  - `run_xgboost()` — XGB with scale_pos_weight
  - `compute_metrics()` — Unified metric computation (AUC, F1, MCC, G-Mean)
  - `plot_roc_curves()` — ROC + PRC curves
  - `run_all_classical_baselines()` — Orchestrator (full 227k samples)
  - `plot_metrics_bar()` — Bar chart comparison

#### Fair Classical Baseline
- **`fair_classical.py`** (NEW, 89 lines)
  - `run_fair_classical_baseline()` — Trains classical on same 160-sample subset as VQC
  - Ensures fair AUC comparison

#### Standard VQC
- **`vqc_baseline.py`** (173 lines)
  - Uses Qiskit's built-in `VQC` class with COBYLA optimizer
  - 8 qubits, ZZFeatureMap + RealAmplitudes ansatz
  - Tests on 249-sample quantum evaluation set
  - Reports baseline performance: AUC 0.9187

#### Novel QCS-VQC Method
- **`qcs_vqc.py`** (631 lines) — Core quantum method
  - `QCSVQCTrainer` class with custom SPSA optimizer
  - **Key components:**
    - `_build_train_estimator()` — Statevector (exact) for training
    - `_build_infer_estimator()` — Noisy (shots=1024) for evaluation
    - `_batch_expectations()` — Single-job circuit evaluation (no chunking, prevents thread exhaustion)
    - `_loss()` — Cost-sensitive loss: `weights * (exp - target)²`
    - `_layerwise_train()` — Grow circuit complexity (reps 1→2)
    - `_spsa()` — Custom SPSA with auto-calibration, multi-restart support
    - `_zne_predict_proba()` — Zero-noise extrapolation at inference
    - `predict_proba()` — Probability predictions (clipped [0,1])
  - **Hyperparameters:**
    - fraud_w=2.0, legit_w=1.0 (cost-sensitive weighting)
    - SPSA: a=0.15, c=0.2, α=0.602, γ=0.101, A_frac=0.1
    - Layerwise: reps grown from 1 to 2
    - Multi-restart: 2 on layer 1, 1 on layer 2

#### Ablation Studies
- **`ablation_study.py`** (314 lines)
  - `run_component_ablations()` — 5 ablation configs:
    1. Full QCS-VQC (fraud_w=2.0, layerwise, no noise/ZNE)
    2. No Cost-Sensitivity (fraud_w=1.0, equal weights)
    3. No Layerwise (single-layer training)
    4. Under Noise, no ZNE (noisy inference, no extrapolation)
    5. Under Noise + ZNE (noisy + zero-noise extrapolation)
  - `run_qubit_sweep()` — Tests 4, 6, 8 qubits (PCA components)
  - `run_imbalance_sweep()` — Tests imbalance ratios 1:10, 1:5, 1:3, 1:2
  - Checkpoint/resume logic: incremental saving after each config
  - Plotting functions for ablation bar chart, qubit sweep, imbalance sweep

#### IBM Hardware Validation (Optional)
- **`ibm_validation.py`** — Runs trained circuit on real IBM quantum hardware
  - Requires IBM Quantum API token
  - Compares simulator vs hardware results

#### Summary & Reporting
- **`generate_tables.py`** (NEW, 189 lines)
  - `generate_all_tables()` — Loads all JSON results and prints publication-ready tables:
    1. Classical baselines (full vs 160 samples)
    2. Quantum methods comparison
    3. Component ablations
    4. Qubit scalability study
    5. Class imbalance robustness
    6. Key findings summary

### `/results/` — Experimental Results

#### JSON Results (Saved Metrics & Loss Curves)
- **`classical_results.json`** — 3 classifiers on full 227k samples (LR, RF, XGB)
- **`classical_results_fair_160.json`** — Same 3 on 160-sample subset
- **`vqc_standard_results.json`** — Standard Qiskit VQC (COBYLA)
- **`qcs_vqc_results.json`** — Main QCS-VQC (SPSA, layerwise)
- **`ablation_component_results.json`** — 5 ablation configs with metrics & loss history
- **`ablation_qubit_sweep.json`** — 3 qubit variants (4, 6, 8)
- **`ablation_imbalance_sweep.json`** — 4 imbalance ratios × 2 methods (QCS-VQC + Std VQC)
- **`all_results_summary.json`** — Main QCS-VQC result with full loss history

#### Plots (PNG)
- **`class_distribution.png`** — Dataset imbalance visualization
- **`roc_curves_classical.png`** — ROC + PRC curves for classical baselines
- **`metrics_comparison_classical.png`** — Bar chart comparing classical metrics
- **`ablation_component_bar.png`** — Ablation study bar chart (5 configs)
- **`ablation_qubit_sweep.png`** — Qubit count vs AUC line plot
- **`ablation_imbalance_sweep.png`** — Imbalance ratio robustness
- **`loss_comparison.png`** — Standard VQC vs QCS-VQC loss curves
- **`final_comparison.png`** — Overall results summary plot

### Root Documentation
- **`RESULTS_SUMMARY.md`** — Publication-ready tables + key findings
- **`CODEBASE_OVERVIEW.md`** — This file; explains all files and their purpose
- **`DIAGRAMS_DESCRIPTION.md`** — Detailed description of all plots and what they show
- **`TABLES_DESCRIPTION.md`** — Detailed description of all result tables
- **`PAPER_OUTLINE.md`** — Full paper content outline and structure

---

## Experimental Pipeline

### Phase 1: Classical Baselines
- **Input:** creditcard.csv
- **Process:** Train LR, RF, XGB on full 227k samples
- **Output:** `classical_results.json`, classical ROC/metrics plots

### Phase 2: Standard VQC Baseline
- **Input:** 160 quantum training samples (80 fraud, 80 legit)
- **Process:** Train standard Qiskit VQC with COBYLA
- **Output:** `vqc_standard_results.json`

### Phase 3: QCS-VQC + Ablations
- **Input:** Same 160 training samples
- **Process:**
  1. Main QCS-VQC with SPSA + layerwise training
  2. Component ablations (5 variants)
  3. Qubit scalability (4, 6, 8 qubits)
  4. Imbalance robustness (4 ratios)
- **Output:** 4 JSON files + 4 plots

### Fair Classical Baseline (New)
- **Input:** Same 160 training samples as VQC
- **Process:** Train LR, RF, XGB on 160 samples only
- **Output:** `classical_results_fair_160.json`

---

## Key Features

### Thread Safety
- Single batch job for statevector (no chunking → prevents thread accumulation)
- Fresh `AerEstimator` created between layers
- Periodic `gc.collect()` every 20 SPSA iterations

### Crash Resilience
- Checkpoint/resume logic in `main.py`
- Incremental saving in ablation functions
- Automatically detects stale results and re-runs

### Hyperparameter Configuration
- All experiments use CONFIG dict in `main.py`
- Easily tunable: num_qubits, max_iter, n_restarts, etc.

### Multi-Seed Ready
- Random seed = 42 in all classifiers
- SPSA has built-in multi-restart (easily extended to seeds)

---

## Usage

```bash
# Full pipeline (all phases)
python main.py

# Individual phases
python main.py --phase 1        # Classical baselines only
python main.py --phase 2        # Standard VQC only
python main.py --phase 3        # QCS-VQC + ablations

# Fair classical baseline + tables
python run_fair_classical.py

# IBM hardware validation (requires token)
python main.py --ibm YOUR_TOKEN
```

---

## Dependencies

See `requirements.txt`:
- **Quantum:** qiskit 1.2.4, qiskit-aer 0.15.0, qiskit-machine-learning 0.8.2
- **ML:** scikit-learn, xgboost, imbalanced-learn
- **Plotting:** matplotlib
- **Other:** pandas, numpy, tabulate (for pretty tables)
- **Python:** 3.11.2 (tested on this version)

---

## Test Files

- **`test_spsa.py`** — Standalone SPSA sanity test (can be deleted)

---

Generated: March 1, 2026
