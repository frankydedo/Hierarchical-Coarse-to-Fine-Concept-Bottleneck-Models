"""
Step 4.5 — Evaluate & Explain
Runs the full hierarchical inference pipeline on the test set:
  1. Coarse prediction from coarse concept scores.
  2. Fine prediction from masked fine concept scores (using PREDICTED coarse).
Generates: accuracy metrics, concept attribution charts.
"""
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def apply_mask(fine_scores, sc_pred, mask):
    """Zero-out fine concepts that don't belong to the predicted superclass."""
    sc_weights = mask[:, sc_pred].T  # (N, N_fine)
    return fine_scores * sc_weights

def evaluate():
    print("Loading model and data...")
    test_proj = torch.load(os.path.join(DATA_DIR, "test_projected.pt"), weights_only=False)
    concept_bank = torch.load(os.path.join(DATA_DIR, "concept_bank.pt"), weights_only=False)
    with open(os.path.join(MODELS_DIR, "classifiers.pkl"), "rb") as f:
        ckpt = pickle.load(f)

    coarse_clf    = ckpt["coarse_clf"]
    coarse_scaler = ckpt["coarse_scaler"]
    fine_clf      = ckpt["fine_clf"]
    fine_scaler   = ckpt["fine_scaler"]
    mask          = ckpt["fine_mask"]   # (N_fine, N_superclasses)

    X_coarse_raw = test_proj["coarse_scores"].numpy()
    X_fine_raw   = test_proj["fine_scores"].numpy()
    y_coarse_true = test_proj["coarse_labels"].numpy()
    y_fine_true   = test_proj["fine_labels"].numpy()

    superclasses  = concept_bank["superclasses"]
    fine_classes  = concept_bank["fine_classes"]
    coarse_names  = concept_bank["coarse_concept_names"]
    fine_names    = concept_bank["fine_concept_names"]

    # ── Stage 1: Coarse prediction ──
    X_coarse = coarse_scaler.transform(X_coarse_raw)
    coarse_pred = coarse_clf.predict(X_coarse)
    coarse_acc  = np.mean(coarse_pred == y_coarse_true)
    print(f"\n[Coarse] Accuracy: {coarse_acc:.4f} ({int(coarse_acc*len(y_coarse_true))}/{len(y_coarse_true)})")

    # ── Stage 2: Fine prediction (using PREDICTED coarse — true pipeline) ──
    X_fine_masked = apply_mask(X_fine_raw, coarse_pred, mask)
    X_fine = fine_scaler.transform(X_fine_masked)
    fine_pred = fine_clf.predict(X_fine)
    fine_acc  = np.mean(fine_pred == y_fine_true)

    # Top-3 accuracy
    fine_probs = fine_clf.predict_proba(X_fine)
    top3_acc   = np.mean([y_fine_true[i] in np.argsort(fine_probs[i])[-3:] for i in range(len(y_fine_true))])

    print(f"[Fine]   Accuracy (predicted mask): {fine_acc:.4f} ({int(fine_acc*len(y_fine_true))}/{len(y_fine_true)})")
    print(f"[Fine]   Top-3 Accuracy:            {top3_acc:.4f}")

    # Oracle: fine with ground-truth mask (upper bound)
    X_fine_oracle = apply_mask(X_fine_raw, y_coarse_true, mask)
    X_fine_oracle = fine_scaler.transform(X_fine_oracle)
    fine_oracle_acc = np.mean(fine_clf.predict(X_fine_oracle) == y_fine_true)
    print(f"[Fine]   Accuracy (oracle mask):    {fine_oracle_acc:.4f}")

    # ── Per-class accuracy breakdown ──
    print("\n--- Per-class fine accuracy ---")
    for i, fc in enumerate(fine_classes):
        idx = np.where(y_fine_true == i)[0]
        if len(idx) == 0: continue
        acc = np.mean(fine_pred[idx] == y_fine_true[idx])
        print(f"  {fc:<20s}: {acc:.2f}  ({int(acc*len(idx))}/{len(idx)})")

    # ── Concept attribution plot ──
    _plot_coarse_attributions(coarse_clf, coarse_names, superclasses)
    _plot_fine_attributions(fine_clf, fine_names, fine_classes)

    print(f"\n✅ Evaluation complete. Figures saved to {FIGURES_DIR}")

def _plot_coarse_attributions(clf, concept_names, superclasses):
    """Bar chart of top coarse concept weights per superclass."""
    coef = clf.coef_   # (N_superclasses, N_coarse_concepts)
    n_sc = len(superclasses)
    top_k = 8

    import math
    n_cols = 4
    n_rows = math.ceil(n_sc / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 4))
    axes = axes.flatten()
    fig.suptitle("Coarse Classifier: Top Concept Weights per Superclass", fontsize=14, y=1.01)

    for i, sc in enumerate(superclasses):
        ax = axes[i]
        weights = coef[i]
        top_idx = np.argsort(np.abs(weights))[-top_k:][::-1]
        top_w   = weights[top_idx]
        top_names = [concept_names[j] for j in top_idx]

        colors = ["#2ecc71" if w > 0 else "#e74c3c" for w in top_w]
        ax.barh(range(top_k), top_w[::-1], color=colors[::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([n[:30] for n in top_names[::-1]], fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(sc.replace("_", " ").title(), fontsize=10, fontweight='bold')
        ax.set_xlabel("Weight")

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "coarse_attributions.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: coarse_attributions.png")

def _plot_fine_attributions(clf, concept_names, fine_classes):
    """Bar chart of top fine concept weights for selected fine classes."""
    import math
    coef = clf.coef_  # (N_fine_classes, N_fine_concepts)
    top_k = 8
    n_classes = len(fine_classes)

    n_cols = 6
    n_rows = math.ceil(n_classes / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("Fine Classifier: Top Concept Weights per Subclass", fontsize=14, y=1.01)

    for i, fc in enumerate(fine_classes):
        ax = axes[i]
        weights = coef[i]
        nonzero = np.where(weights != 0)[0]
        if len(nonzero) == 0:
            ax.set_title(fc, fontsize=8)
            ax.axis('off')
            continue
        top_idx = nonzero[np.argsort(np.abs(weights[nonzero]))[-top_k:][::-1]]
        top_w   = weights[top_idx]
        top_names = [concept_names[j] for j in top_idx]

        colors = ["#2ecc71" if w > 0 else "#e74c3c" for w in top_w]
        ax.barh(range(len(top_idx)), top_w[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([n[:25] for n in top_names[::-1]], fontsize=6)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(fc.replace("_", " ").title(), fontsize=8, fontweight='bold')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fine_attributions.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: fine_attributions.png")

if __name__ == "__main__":
    evaluate()
