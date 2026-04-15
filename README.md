# Hybrid Classical–Quantum Fraud Detection using QCS-VQC

## Overview

This project implements a **Hybrid Classical–Quantum Machine Learning framework** for fraud detection using a **Quantum-Classical Synergistic Variational Quantum Classifier (QCS-VQC)**.

The system combines:

* Classical preprocessing techniques
* Quantum feature encoding (ZZFeatureMap)
* Variational quantum circuits (RealAmplitudes)
* Hybrid optimization for classification

The goal is to evaluate whether **quantum-enhanced models** can improve performance on **imbalanced fraud detection datasets**.

---

## Key Features

* Hybrid classical–quantum architecture
* Variational Quantum Classifier (VQC)
* Custom QCS-VQC model
* Class imbalance analysis
* Ablation studies (qubits, reps, components)
* Comparison with classical baselines
* Evaluation using robust metrics (AUC, MCC, F1, G-Mean)

---

## Project Structure

```
src/
├── data_preprocessing.py       # Data cleaning and scaling
├── classical_baselines.py     # Traditional ML models
├── vqc_baseline.py            # Standard VQC implementation
├── qcs_vqc.py                 # Proposed QCS-VQC model
├── fair_classical.py          # Fairness-aware models
├── ablation_study.py          # Experimental analysis
├── ibm_validation.py          # Quantum backend validation
├── generate_tables.py         # Result summarization

main.py                        # Main execution script
run_fair_classical.py          # Fair model execution
generate_circuits.py           # Circuit generation
creditcard.csv                 # Dataset (optional)

results/                       # Generated outputs (ignored in Git)
```

---

## System Architecture

The system follows a hybrid pipeline:

1. **Data Preprocessing**

   * Normalization and scaling
   * Train-test split

2. **Feature Encoding**

   * Classical data mapped to quantum states using ZZFeatureMap

3. **Variational Quantum Circuit**

   * Parameterized gates (RealAmplitudes)
   * Entanglement using CNOT gates

4. **Measurement**

   * Quantum states converted to probabilities

5. **Classification**

   * Binary output: Legitimate (0) / Fraudulent (1)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Main Pipeline

```bash
python main.py
```

### Run Ablation Study

```bash
python src/ablation_study.py
```

### Run Classical Baselines

```bash
python src/classical_baselines.py
```

---

## Results

The model is evaluated using:

* AUC-ROC
* AUC-PR
* F1 Score
* MCC (Matthews Correlation Coefficient)
* G-Mean

Key findings:

* Strong ranking performance (high AUC)
* Sensitivity to class imbalance
* QCS-VQC improves performance at moderate imbalance levels

---

## Technologies Used

* Python
* Qiskit
* NumPy / Pandas
* Scikit-learn
* Matplotlib / Seaborn

---

## Limitations

* Quantum simulations are computationally expensive
* Performance depends heavily on feature encoding
* Limited scalability with increasing qubits

---

## Future Work

* Integration with real quantum hardware (IBM Quantum)
* Improved encoding strategies
* Advanced hybrid optimization techniques
* Scaling to larger datasets

---

## Author

Molathoti Nikhil

---

## License

This project is for academic and research purposes.

## Output

<img width="567" height="327" alt="Screenshot 2026-03-04 095907" src="https://github.com/user-attachments/assets/f6a6edbb-ac05-4d25-b80f-3188336de119" />
<img width="997" height="454" alt="Screenshot 2026-03-04 095702" src="https://github.com/user-attachments/assets/9a58922a-e8b2-4378-a257-4b59d72309ae" />
<img width="701" height="347" alt="Screenshot 2026-03-03 141331" src="https://github.com/user-attachments/assets/e0d9fb13-7043-4fed-97a0-0f7158fbb0aa" />
<img width="356" height="207" alt="Screenshot 2026-03-03 141240" src="https://github.com/user-attachments/assets/c2052f13-b0e2-493b-8070-efffca299d26" />
<img width="699" height="268" alt="Screenshot 2026-03-03 141157" src="https://github.com/user-attachments/assets/c79cc770-477a-478a-b4b3-6206bd479ebb" />
<img width="699" height="466" alt="Screenshot 2026-03-03 141100" src="https://github.com/user-attachments/assets/34633e73-7a6a-4818-9a7d-b78dfab136a2" />
<img width="534" height="185" alt="Screenshot 2026-03-03 083531" src="https://github.com/user-attachments/assets/6e945cca-b77d-4cca-abaa-3ab9d7812eec" />
<img width="700" height="251" alt="Screenshot 2026-03-03 141826" src="https://github.com/user-attachments/assets/d964929f-9370-48b3-a5d5-bd0f91e51a19" />
