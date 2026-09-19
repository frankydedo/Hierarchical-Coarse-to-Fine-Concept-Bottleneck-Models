"""
Step 4.4 — Train Hierarchical Classifiers (GLM-SAGA)
Trains two L1-regularized logistic regression classifiers:
  1. Coarse: predicts superclass from coarse concept activations.
  2. Fine:   predicts subclass from fine concept activations, using a
             logical mask that zeros out concepts not belonging to the
             ground-truth superclass (teacher forcing during training).
"""
import os
import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def build_fine_mask(concept_bank, n_superclasses):
    """
    Returns a binary mask of shape (N_fine_concepts, N_superclasses).
    mask[i, j] = 1 if fine concept i belongs to superclass j.
    Used to zero out irrelevant fine concepts per predicted/true superclass.
    """
    fine_sc_ids = concept_bank["fine_concept_sc_ids"]  # (N_fine,)
    n_fine = fine_sc_ids.shape[0]
    mask = torch.zeros(n_fine, n_superclasses, dtype=torch.float32)
    for i, sc_id in enumerate(fine_sc_ids):
        mask[i, sc_id.item()] = 1.0
    return mask  # (N_fine, N_superclasses)

def apply_mask(fine_scores, true_sc_labels, mask):
    """
    Applies the logical mask: for each sample, zero out all fine concept
    activations that don't belong to the sample's superclass.
    fine_scores: numpy (N, N_fine)
    true_sc_labels: numpy (N,) integer superclass indices
    mask: numpy (N_fine, N_superclasses)
    Returns: (N, N_fine) masked scores
    """
    # For each sample i, keep only concepts whose SC matches true_sc_labels[i]
    sc_weights = mask[:, true_sc_labels].T  # (N, N_fine)
    return fine_scores * sc_weights

def train_coarse(train_proj, val_proj=None):
    print("\n=== Training Coarse Classifier ===")
    X_train = train_proj["coarse_scores"].numpy()
    y_train = train_proj["coarse_labels"].numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=0.5,          # inverse regularization strength; smaller = sparser
        max_iter=2000,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    train_acc = clf.score(X_train_scaled, y_train)
    print(f"  Train accuracy: {train_acc:.4f}")

    if val_proj is not None:
        X_val = scaler.transform(val_proj["coarse_scores"].numpy())
        y_val = val_proj["coarse_labels"].numpy()
        val_acc = clf.score(X_val, y_val)
        print(f"  Test  accuracy: {val_acc:.4f}")
        n_nonzero = np.sum(clf.coef_ != 0)
        print(f"  Non-zero coef:  {n_nonzero} / {clf.coef_.size}")

    return clf, scaler

def train_fine(train_proj, concept_bank, val_proj=None):
    print("\n=== Training Fine Classifier (with Logical Mask) ===")
    superclasses = train_proj["superclasses"]
    n_superclasses = len(superclasses)
    mask = build_fine_mask(concept_bank, n_superclasses).numpy()

    X_train_raw = train_proj["fine_scores"].numpy()
    y_train_coarse = train_proj["coarse_labels"].numpy()
    y_train_fine = train_proj["fine_labels"].numpy()

    # Teacher forcing: mask using ground-truth superclass during training
    X_train_masked = apply_mask(X_train_raw, y_train_coarse, mask)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_masked)

    clf = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=0.5,
        max_iter=2000,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train_fine)

    train_acc = clf.score(X_train_scaled, y_train_fine)
    print(f"  Train accuracy: {train_acc:.4f}")

    if val_proj is not None:
        X_val_raw = val_proj["fine_scores"].numpy()
        y_val_coarse = val_proj["coarse_labels"].numpy()
        y_val_fine = val_proj["fine_labels"].numpy()
        # At test time: mask with ground-truth SC (oracle upper bound)
        X_val_masked = apply_mask(X_val_raw, y_val_coarse, mask)
        X_val_scaled = scaler.transform(X_val_masked)
        val_acc = clf.score(X_val_scaled, y_val_fine)
        print(f"  Test  accuracy (oracle mask): {val_acc:.4f}")
        n_nonzero = np.sum(clf.coef_ != 0)
        print(f"  Non-zero coef: {n_nonzero} / {clf.coef_.size}")

    return clf, scaler, mask

def main():
    print("Loading projected features and concept bank...")
    train_proj = torch.load(os.path.join(DATA_DIR, "train_projected.pt"), weights_only=False)
    test_proj  = torch.load(os.path.join(DATA_DIR, "test_projected.pt"),  weights_only=False)
    concept_bank = torch.load(os.path.join(DATA_DIR, "concept_bank.pt"),  weights_only=False)

    # Train coarse classifier
    coarse_clf, coarse_scaler = train_coarse(train_proj, val_proj=test_proj)

    # Train fine classifier (with masking)
    fine_clf, fine_scaler, mask = train_fine(train_proj, concept_bank, val_proj=test_proj)

    # Save everything
    checkpoint = {
        "coarse_clf": coarse_clf,
        "coarse_scaler": coarse_scaler,
        "fine_clf": fine_clf,
        "fine_scaler": fine_scaler,
        "fine_mask": mask,  # (N_fine_concepts, N_superclasses)
    }
    save_path = os.path.join(MODELS_DIR, "classifiers.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"\n✅ Classifiers saved to {save_path}")

if __name__ == "__main__":
    main()
