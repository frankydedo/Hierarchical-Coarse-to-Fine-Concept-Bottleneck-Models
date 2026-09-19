"""
Phase 4 — Flat LF-CBM Baseline
Trains a single logistic regression (L1) on ALL 769 concepts concatenated,
predicting 100 fine classes directly. No hierarchy, no gating.
Serves as the interpretable ablation baseline.
"""
import os
import sys
import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

def run_flat_baseline():
    print("=" * 60)
    print("PHASE 4 — Flat LF-CBM Baseline")
    print("=" * 60)

    # ── Load projected features ──────────────────────────────────
    print("\nLoading train/test projections...")
    train = torch.load(os.path.join(DATA_DIR, "train_projected.pt"), weights_only=False)
    test  = torch.load(os.path.join(DATA_DIR, "test_projected.pt"),  weights_only=False)

    # Concatenate coarse + fine concept scores into one flat vector
    X_train = np.concatenate([train["coarse_scores"].numpy(),
                              train["fine_scores"].numpy()], axis=1)
    X_test  = np.concatenate([test["coarse_scores"].numpy(),
                              test["fine_scores"].numpy()],  axis=1)

    y_train = train["fine_labels"].numpy()
    y_test  = test["fine_labels"].numpy()

    print(f"  Feature dim (coarse + fine): {X_train.shape[1]}")
    print(f"  Train samples: {len(y_train)} | Test samples: {len(y_test)}")

    # ── Scale + Train ─────────────────────────────────────────────
    print("\nTraining Flat LF-CBM (L1 Logistic Regression)...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=0.1,
        max_iter=3000,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────
    fine_pred  = clf.predict(X_test)
    fine_probs = clf.predict_proba(X_test)
    top1_acc   = accuracy_score(y_test, fine_pred)
    top3_acc   = np.mean([y_test[i] in np.argsort(fine_probs[i])[-3:]
                          for i in range(len(y_test))])

    # Sparsity: average number of non-zero concepts active per class
    n_nonzero_per_class = (clf.coef_ != 0).sum(axis=1)
    avg_active   = n_nonzero_per_class.mean()
    total_concepts = clf.coef_.shape[1]

    print(f"\n[Flat LF-CBM] Top-1 Accuracy:      {top1_acc:.4f}")
    print(f"[Flat LF-CBM] Top-3 Accuracy:      {top3_acc:.4f}")
    print(f"[Flat LF-CBM] Avg active concepts: {avg_active:.1f} / {total_concepts}  "
          f"({100*avg_active/total_concepts:.1f}% sparsity used)")

    # ── Save model + results ──────────────────────────────────────
    results = {
        "model": "Flat LF-CBM",
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "avg_active_concepts": float(avg_active),
        "total_concepts": total_concepts,
        "clf": clf,
        "scaler": scaler,
        "y_pred": fine_pred,
        "y_true": y_test,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "flat_baseline.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"\n✅ Results saved to models/flat_baseline.pkl")
    return results

if __name__ == "__main__":
    run_flat_baseline()
