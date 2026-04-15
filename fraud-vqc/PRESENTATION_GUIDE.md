# How to Present This Project to a University Reviewer

A practical guide for defending/presenting the QCS-VQC fraud detection project in a viva, thesis review, or project evaluation.

---

## 1. Opening Statement (2 minutes)

Start with the problem, not the solution.

> "Credit card fraud causes over $32 billion in annual losses. The standard dataset for benchmarking fraud detection — the UCI credit card dataset — has only 0.17% fraud among 284,807 transactions. Classical machine learning handles this well, achieving up to 97% AUC-ROC. But can quantum computing offer anything here? And more importantly, do standard ML techniques like cost-sensitive learning still work when transplanted into quantum circuits? That's what this project investigates."

**Key points to hit:**
- Real-world relevance (fraud losses)
- Extreme class imbalance (492 fraud out of 284,807)
- Research question: Do classical ML best-practices transfer to quantum?

---

## 2. What We Built (3 minutes)

### Architecture Overview

Explain the pipeline in plain language:

1. **Data** → PCA reduces 30 features to 8 components (57% variance) → scale to [0, π]
2. **Quantum encoding** → ZZFeatureMap maps each feature to one qubit via Hadamard + Z-rotation + ZZ-entanglement gates
3. **Trainable circuit** → RealAmplitudes ansatz with R_Y rotations + CNOT ladder
4. **Training** → Custom SPSA optimizer with auto-calibration, layerwise circuit growth (reps 1→2)
5. **Loss function** → Cost-sensitive MSE: fraud samples weighted 2× more than legitimate
6. **Prediction** → Measure Pauli-Z on qubit 0; convert expectation to fraud probability

**Show these diagrams:**
- `results/zzfeaturemap_4q_reps2.png` — "This is how data gets encoded into quantum states"
- `results/realamplitudes_4q_reps2.png` — "These are the trainable parameters the optimizer adjusts"
- `results/qcsvqc_full_circuit_4q.png` — "This is the complete circuit: data goes in, classification comes out"

**Anticipated question:** "Why only 8 qubits?"
> "PCA to 8 components captures 57% of variance. Each qubit encodes one feature. Going below 8 causes performance collapse — our qubit sweep study shows this. Going above 8 would need more qubits but current NISQ hardware is noisy; we'd need error correction."

---

## 3. The Novel Contribution (3 minutes)

**Be very clear about what is new:**

> "The novelty is NOT that we built a quantum fraud detector — others have done that. The novelty is threefold:"

### Contribution 1: Cost-Sensitive Quantum Loss
> "We integrated asymmetric cost-weighting into the VQC loss function. In classical ML, this always helps with imbalanced data. We found it HURTS in quantum circuits — a surprising negative result."

### Contribution 2: Systematic Ablation
> "We didn't just try one configuration. We ran five ablation experiments to isolate which component helps and which hurts. This is unusual in quantum ML papers, which typically report one result."

### Contribution 3: Practical Findings
> "We provide actionable recommendations:
> - Use layerwise training (biggest improvement: +5.3% AUC)
> - Don't use cost-sensitive loss on balanced subsamples
> - Don't use zero-noise extrapolation without calibrated noise models
> - 8 qubits minimum for 8-dimensional data"

**Anticipated question:** "But your method underperforms the standard VQC. Isn't that a failure?"
> "No — negative results are scientifically valuable. We showed WHY it fails: cost-sensitive loss causes all-fraud predictions on balanced data. The ablation without cost-sensitivity actually improves performance. The contribution is understanding what works and what doesn't, not claiming quantum superiority."

---

## 4. Results Walkthrough (5 minutes)

Present results in this order — it tells a story:

### Step 1: Classical Ceiling
> "First, let's establish what classical ML achieves."

| Method | Training Data | AUC-ROC |
|--------|-------------|---------|
| XGBoost | 227,845 | 0.9746 |
| Random Forest | 160 | 0.9188 |

> "Even with only 160 training samples, Random Forest gets 91.88% AUC. This is our target."

### Step 2: Quantum Results
> "Now our quantum methods, trained on the same 160 samples:"

| Method | AUC-ROC | Key Insight |
|--------|---------|-------------|
| Standard VQC (COBYLA) | **0.9187** | Matches classical RF! |
| QCS-VQC (Cost-Sensitive) | 0.8369 | Cost-weighting hurts |

> "Standard VQC essentially matches Random Forest at 91.87% vs 91.88%. But our cost-sensitive version underperforms. Why?"

### Step 3: Ablation Answers "Why"
Show `results/ablation_component_bar.png` and explain:

> "The ablation study reveals three findings:"

1. **Cost-sensitivity is harmful** (-3.0% AUC): It pushes the classifier to predict everything as fraud (TP=49, FP=200, TN=0). On already-balanced data, extra fraud weighting overcorrects.

2. **Layerwise training is essential** (+5.3% AUC): Training shallow circuit first, then growing deeper avoids barren plateaus — regions where gradients vanish and the optimizer gets stuck.

3. **ZNE is catastrophic** (-31% AUC): Zero-noise extrapolation with an uncalibrated noise model amplifies errors instead of correcting them.

### Step 4: Scalability
Show `results/ablation_qubit_sweep.png`:

> "Below 8 qubits, performance collapses. 6 qubits gives 23.6% AUC — worse than random. This tells us one qubit per feature is the minimum encoding requirement."

### Step 5: Imbalance Sensitivity
Show `results/ablation_imbalance_sweep.png`:

> "Interestingly, performance peaks at 1:3 fraud-to-legitimate ratio, not at balanced (1:2) or extreme (1:10). This non-monotonic behaviour suggests quantum classifiers need some asymmetry in training data to learn effectively."

---

## 5. Technical Depth — Be Ready for These Questions

### Q: "Why SPSA instead of COBYLA?"
> "SPSA needs only 2 loss evaluations per step regardless of parameter count — O(2) instead of O(n). For 24 parameters, this is 12× cheaper. SPSA is the standard for variational quantum algorithms. However, our results show COBYLA actually works better for Standard VQC — which is itself an interesting finding about optimizer-architecture interaction."

### Q: "Why not use a real quantum computer?"
> "We used Qiskit Aer simulator with statevector for exact results, plus a depolarising noise model for realistic inference. Real hardware validation on IBM Quantum is planned as future work but requires API access and credits. The simulator gives us controlled experimental conditions — we can isolate cost-sensitivity effects without hardware noise confounding the results."

### Q: "How is this different from just running sklearn on the same data?"
> "The classical models are baselines, not the contribution. The contribution is understanding how quantum-specific techniques (cost-sensitive quantum loss, layerwise circuit training, zero-noise extrapolation) interact. These are quantum phenomena — barren plateaus, unitary folding, parameterised circuits — that don't exist in classical ML."

### Q: "What about overfitting with 160 samples?"
> "We use a separate test set of 249 samples and report threshold-independent metrics (AUC-ROC, AUC-PRC). The quantum circuit has only 24 parameters for 160 training samples — a much lower parameter-to-sample ratio than typical neural networks. Cross-validation was not performed due to computational cost (~4000 seconds per training run), but this is flagged as a limitation."

### Q: "Why does the cost-sensitive loss hurt?"
> "Three hypotheses: (1) The training set is already balanced (80:80), so adding fraud weight = 2.0 creates an artificial majority class in the loss landscape. (2) The quantum loss landscape has many degenerate local minima; cost-weighting steers the optimizer into one that predicts all-fraud. (3) SPSA's stochastic gradient estimates may be biased by asymmetric loss scaling. The confusion matrix confirms hypothesis 2: TP=49, FP=200, TN=0 — a trivial all-fraud classifier."

### Q: "Is there quantum advantage here?"
> "No. Standard VQC matches classical RF but doesn't surpass it, and is 10-100× slower. We don't claim quantum advantage. Our contribution is understanding component interactions in quantum fraud classifiers, which informs future algorithm design when hardware improves."

### Q: "What would you do differently with more time?"
> "Four things: (1) Multi-seed runs (5 seeds per config) for error bars and statistical significance. (2) Cost-weight hyperparameter sweep over α ∈ {0.5, 1.0, 1.5, 2.0, 3.0}. (3) IBM hardware validation. (4) Test on naturally imbalanced training sets instead of artificially balanced ones — our main finding suggests cost-sensitivity might work better there."

---

## 6. Presentation Structure Summary

| Segment | Duration | What to Show |
|---------|----------|-------------|
| Problem & Motivation | 2 min | Class imbalance stats, fraud impact |
| Architecture | 3 min | Circuit diagrams (ZZFeatureMap, ansatz, full circuit) |
| Novel Contribution | 3 min | Cost-sensitive loss + ablation study + findings |
| Results | 5 min | Tables 1-4, ablation bar chart, qubit/imbalance plots |
| Discussion | 3 min | Why cost-sensitivity hurts, layerwise helps |
| Future Work | 2 min | Multi-seed, hardware validation, cost-weight sweep |
| Q&A | 5-10 min | See prepared answers above |
| **Total** | **~25 min** | |

---

## 7. Slides Outline (if using PowerPoint/Beamer)

**Slide 1:** Title — "Cost-Sensitive Loss Harms Variational Quantum Classifiers on Balanced Training Data"

**Slide 2:** Problem — Fraud statistics, class imbalance bar chart (`class_distribution.png`)

**Slide 3:** Research Question — "Do classical ML techniques transfer to quantum circuits?"

**Slide 4:** Background — What is a VQC? (simple diagram: data → quantum circuit → measurement → classification)

**Slide 5:** Circuit Design — ZZFeatureMap diagram (`zzfeaturemap_4q_reps2.png`)

**Slide 6:** Circuit Design — RealAmplitudes ansatz (`realamplitudes_4q_reps2.png`)

**Slide 7:** Full Circuit — Combined QCS-VQC (`qcsvqc_full_circuit_4q.png`)

**Slide 8:** Training — SPSA optimizer formula, layerwise growth diagram (reps 1→2)

**Slide 9:** Cost-Sensitive Loss — Formula with weights, hypothesis for improvement

**Slide 10:** Classical Baselines — Table with LR/RF/XGB results (227k and 160 samples)

**Slide 11:** Quantum Results — Standard VQC vs QCS-VQC comparison table

**Slide 12:** KEY FINDING — "Cost-sensitive loss HURTS" — ablation bar chart (`ablation_component_bar.png`)

**Slide 13:** Ablation Details — Confusion matrices for Full vs No-Cost-Sensitivity

**Slide 14:** Layerwise Training — +5.3% improvement, barren plateau explanation

**Slide 15:** ZNE Failure — -31% with uncalibrated noise

**Slide 16:** Qubit Scalability — Line chart (`ablation_qubit_sweep.png`)

**Slide 17:** Imbalance Robustness — Dual line chart (`ablation_imbalance_sweep.png`)

**Slide 18:** Summary of Findings — 5 bullet points (key takeaways)

**Slide 19:** Limitations & Future Work — What we'd do next

**Slide 20:** Thank You / Q&A

---

## 8. Common Pitfalls to Avoid

1. **Don't oversell quantum.** You didn't beat classical. Own it — the ablation findings are the real contribution.

2. **Don't skip the negative results.** The cost-sensitivity failure IS the main finding. Present it confidently as a discovery, not an apology.

3. **Don't get bogged down in Qiskit API details.** Reviewers care about the methodology and findings, not which Python function you called.

4. **Don't show 8-qubit circuit diagrams in slides.** They're too wide to read. Use the 4-qubit versions — they show the same structure clearly.

5. **Don't compare unfairly.** Always mention that classical models on 227k samples are NOT a fair comparison. Emphasise the 160-sample matched baseline.

6. **Don't claim novelty you don't have.** The algorithm itself (VQC + cost-sensitive loss) is a straightforward extension. The novelty is the systematic evaluation and unexpected findings.

---

## 9. One-Paragraph Summary (for abstract/elevator pitch)

> "We investigated whether cost-sensitive learning — a standard technique for handling class imbalance in classical machine learning — transfers effectively to variational quantum classifiers for credit card fraud detection. Using the UCI credit card dataset with 0.17% fraud rate, we built a Quantum Cost-Sensitive VQC combining asymmetric loss weighting, SPSA optimization, and layerwise circuit training. Through systematic ablation of five configurations, we discovered that cost-sensitive loss paradoxically reduces quantum classifier performance by 3% AUC on balanced training data, producing degenerate all-fraud predictions. In contrast, layerwise training improved performance by 5.3%, and zero-noise extrapolation caused a catastrophic 31% AUC degradation under uncalibrated noise. A standard VQC achieved 91.87% AUC, matching classical Random Forest at 91.88% on identical 160-sample training sets. These findings demonstrate that classical ML best-practices do not straightforwardly transfer to quantum circuits, providing practical guidance for NISQ-era algorithm design."

---

Generated: March 2, 2026
