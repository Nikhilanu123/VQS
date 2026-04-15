"""
Phase 1 - Data Preprocessing
-----------------------------
Loads the creditcard.csv dataset, handles class imbalance analysis,
scales features, and prepares train/test splits for all experiments.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import os


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load creditcard.csv and print basic statistics."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {filepath}\n"
            "Please download from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place creditcard.csv in the project root folder.\n"
        )
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("DATASET LOADED")
    print("=" * 60)
    print(f"  Shape            : {df.shape}")
    print(f"  Total samples    : {len(df):,}")
    print(f"  Fraud samples    : {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"  Legit samples    : {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.3f}%)")
    print(f"  Missing values   : {df.isnull().sum().sum()}")
    print("=" * 60)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, pca_components: int = None):
    """
    - Scales 'Amount' and 'Time'
    - Optionally reduces to top-N PCA components for quantum experiments
    - Returns (X, y)
    """
    df = df.copy()

    # Scale Amount and Time (V1-V28 already PCA-transformed by dataset)
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time']   = scaler.fit_transform(df[['Time']])
    df.drop(['Amount', 'Time'], axis=1, inplace=True)

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # Optional: reduce features for quantum circuit (needs small qubit count)
    if pca_components is not None:
        pca = PCA(n_components=pca_components, random_state=42)
        X = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA: {pca_components} components explain {explained*100:.2f}% variance")

        # Re-scale to [0, π] for quantum angle encoding
        mms = MinMaxScaler(feature_range=(0, np.pi))
        X = mms.fit_transform(X)

    print(f"  Feature shape    : {X.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified split to preserve class ratio."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"  Train size       : {len(X_train):,} (fraud: {y_train.sum()})")
    print(f"  Test size        : {len(X_test):,}  (fraud: {y_test.sum()})")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUBSAMPLE FOR QUANTUM EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

def quantum_subsample(X_train, y_train, n_fraud=200, n_legit=200, random_state=42):
    """
    Quantum circuits are slow — create a small balanced subsample for VQC training.
    n_fraud: number of fraud samples to keep
    n_legit: number of legitimate samples to keep
    """
    rng = np.random.default_rng(random_state)

    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]

    sel_fraud = rng.choice(fraud_idx, size=min(n_fraud, len(fraud_idx)), replace=False)
    sel_legit = rng.choice(legit_idx, size=min(n_legit, len(legit_idx)), replace=False)

    sel_idx = np.concatenate([sel_fraud, sel_legit])
    rng.shuffle(sel_idx)

    print(f"\n  Quantum subsample: {len(sel_idx)} samples "
          f"(fraud: {len(sel_fraud)}, legit: {len(sel_legit)})")
    return X_train[sel_idx], y_train[sel_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 5. SMOTE OVERSAMPLING (for classical + standard VQC comparison)
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to training set only."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(X_res):,} samples (fraud: {y_res.sum():,})")
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(y, save_path="results/class_distribution.png"):
    """Bar plot of class imbalance."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = ['Legitimate', 'Fraud']
    counts = [(y == 0).sum(), (y == 1).sum()]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=['steelblue', 'crimson'], width=0.4)
    ax.set_title("Class Distribution — Credit Card Fraud Dataset")
    ax.set_ylabel("Sample Count")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{count:,}", ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")

    df = load_data(DATA_PATH)
    plot_class_distribution(df['Class'].values)

    print("\n--- Full feature set (for classical models) ---")
    X_full, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X_full, y)

    print("\n--- Reduced feature set (for quantum models: 8 qubits) ---")
    X_q, y_q = preprocess(df, pca_components=8)
    X_q_train, X_q_test, y_q_train, y_q_test = split_data(X_q, y_q)
    X_q_sub, y_q_sub = quantum_subsample(X_q_train, y_q_train, n_fraud=200, n_legit=200)
