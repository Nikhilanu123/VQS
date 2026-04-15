"""Quick SPSA sanity test on real creditcard data."""
import numpy as np, sys, time
sys.path.insert(0, '.')
from src.data_preprocessing import load_data, preprocess, split_data, quantum_subsample
from src.qcs_vqc import QCSVQCTrainer
from sklearn.metrics import roc_auc_score

np.random.seed(42)
df = load_data('creditcard.csv')
X, y = preprocess(df, pca_components=8)
X_tr, X_te, y_tr, y_te = split_data(X, y)
X_sub, y_sub = quantum_subsample(X_tr, y_tr, n_fraud=30, n_legit=30)
X_test, y_test = quantum_subsample(X_te, y_te, n_fraud=15, n_legit=45)

t0 = time.time()
t = QCSVQCTrainer(num_qubits=8, reps=2, max_iter=60,
                  use_noise=False, use_zne=False,
                  dataset_fraud_ratio=df['Class'].mean())
t.fit(X_sub, y_sub, layerwise=True)
elapsed = time.time() - t0

p = t.predict_proba(X_test)
pred = (p >= 0.5).astype(int)

tp = ((pred == 1) & (y_test == 1)).sum()
tn = ((pred == 0) & (y_test == 0)).sum()
fp = ((pred == 1) & (y_test == 0)).sum()
fn = ((pred == 0) & (y_test == 1)).sum()

print(f'\n=== RESULTS ({elapsed:.0f}s) ===')
print(f'Fraud probas: {np.round(p[y_test==1], 3)}')
print(f'Legit probas (first 15): {np.round(p[y_test==0][:15], 3)}')
print(f'AUC={roc_auc_score(y_test, p):.4f}')
print(f'TP={tp} TN={tn} FP={fp} FN={fn}')
print(f'Recall={tp/(tp+fn+1e-9):.3f} Precision={tp/(tp+fp+1e-9):.3f}')
