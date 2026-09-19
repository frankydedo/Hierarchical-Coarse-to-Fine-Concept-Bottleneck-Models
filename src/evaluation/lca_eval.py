"""
Phase 4 — LCA Evaluation & Results Aggregation
Computes the Lowest Common Ancestor (LCA) Error Distance for all three models
and prints the final comparison table.

LCA distance:
  0 → correct prediction
  1 → sibling mistake (same superclass)
  2 → catastrophic mistake (different superclass)
"""
import os
import sys
import pickle
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.taxonomy import cifar100_taxonomy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

# ── Build taxonomic helpers ──────────────────────────────────────
SUPERCLASSES     = list(cifar100_taxonomy.keys())
FINE_CLASSES_ORD = [fc for sc in SUPERCLASSES for fc in cifar100_taxonomy[sc]]

# Map each fine class index → superclass index
def build_fine_to_super():
    mapping = {}
    for sc_idx, (sc, fine_classes) in enumerate(cifar100_taxonomy.items()):
        for fc in fine_classes:
            fc_idx = FINE_CLASSES_ORD.index(fc)
            mapping[fc_idx] = sc_idx
    return mapping

FINE_TO_SUPER = build_fine_to_super()

def lca_distance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Returns an array of LCA distances for each sample:
      0 → correct
      1 → same superclass (sibling)
      2 → different superclass (catastrophic)
    """
    dists = []
    for t, p in zip(y_true, y_pred):
        if t == p:
            dists.append(0)
        elif FINE_TO_SUPER[t] == FINE_TO_SUPER[p]:
            dists.append(1)
        else:
            dists.append(2)
    return np.array(dists)

def load_hierarchical_results():
    """Load the predictions from our Phase 3 Hierarchical CBM."""
    test = torch.load(os.path.join(DATA_DIR, "test_projected.pt"), weights_only=False)
    with open(os.path.join(MODELS_DIR, "classifiers.pkl"), "rb") as f:
        ckpt = pickle.load(f)

    from sklearn.preprocessing import StandardScaler
    coarse_clf    = ckpt["coarse_clf"]
    coarse_scaler = ckpt["coarse_scaler"]
    fine_clf      = ckpt["fine_clf"]
    fine_scaler   = ckpt["fine_scaler"]
    mask          = ckpt["fine_mask"]

    X_coarse = coarse_scaler.transform(test["coarse_scores"].numpy())
    coarse_pred = coarse_clf.predict(X_coarse)

    X_fine_raw = test["fine_scores"].numpy()
    sc_weights = mask[:, coarse_pred].T
    X_fine_masked = X_fine_raw * sc_weights
    X_fine = fine_scaler.transform(X_fine_masked)
    fine_pred = fine_clf.predict(X_fine)

    fine_probs = fine_clf.predict_proba(X_fine)
    y_true = test["fine_labels"].numpy()

    # Sparsity from coarse + fine classifiers combined
    n_nonzero_coarse = (coarse_clf.coef_ != 0).sum(axis=1).mean()
    n_nonzero_fine   = (fine_clf.coef_ != 0).sum(axis=1).mean()
    avg_active_str   = f"({n_nonzero_coarse:.1f}+{n_nonzero_fine:.1f})/2"
    total_concepts = coarse_clf.coef_.shape[1] + fine_clf.coef_.shape[1]


    # Oracle accuracy
    y_coarse_true = test["coarse_labels"].numpy()
    sc_weights_oracle = mask[:, y_coarse_true].T
    X_fine_oracle = fine_scaler.transform(X_fine_raw * sc_weights_oracle)
    oracle_pred = fine_clf.predict(X_fine_oracle)
    oracle_acc = np.mean(oracle_pred == y_true)

    top1 = np.mean(fine_pred == y_true)
    top3 = np.mean([y_true[i] in np.argsort(fine_probs[i])[-3:]
                    for i in range(len(y_true))])

    hier_std = {
        "model": "Hier. CBM (Ours)",
        "top1_acc": top1,
        "top3_acc": top3,
        "avg_active_concepts": avg_active_str,
        "total_concepts": total_concepts,
        "y_pred": fine_pred,
        "y_true": y_true,
    }

    hier_oracle = {
        "model": "Hier. CBM (Oracle)",
        "top1_acc": oracle_acc,
        "top3_acc": None,
        "avg_active_concepts": f"{n_nonzero_fine:.1f}",
        "total_concepts": fine_clf.coef_.shape[1],
        "y_pred": oracle_pred,
        "y_true": y_true,
    }

    return hier_std, hier_oracle

def run_lca_eval():
    print("=" * 60)
    print("PHASE 4 — LCA Evaluation & Final Comparison Table")
    print("=" * 60)

    # Load all model results
    hier, hier_oracle = load_hierarchical_results()
    print(f"  [OK] Loaded Hierarchical CBM results (Standard & Oracle).")

    flat_path = os.path.join(MODELS_DIR, "flat_baseline.pkl")
    zs_path   = os.path.join(MODELS_DIR, "zeroshot_baseline.pkl")

    if not os.path.exists(flat_path):
        print("  [!!] flat_baseline.pkl not found. Run flat_baseline.py first.")
        flat = None
    else:
        with open(flat_path, "rb") as f:
            flat = pickle.load(f)
        print(f"  [OK] Loaded Flat LF-CBM results.")

    if not os.path.exists(zs_path):
        print("  [!!] zeroshot_baseline.pkl not found. Run zeroshot_baseline.py first.")
        zs = None
    else:
        with open(zs_path, "rb") as f:
            zs = pickle.load(f)
        zs["avg_active_concepts"] = "512(dim)"
        print(f"  [OK] Loaded CLIP Linear Probe results.")

    # ── LCA distances ────────────────────────────────────────────
    print("\nComputing LCA distances...")
    models = [m for m in [hier, hier_oracle, flat, zs] if m is not None]
    for m in models:
        dists = lca_distance(m["y_true"], m["y_pred"])
        m["lca_dist_mean"]         = float(dists.mean())
        m["lca_pct_correct"]       = float((dists == 0).mean())
        m["lca_pct_sibling"]       = float((dists == 1).mean())
        m["lca_pct_catastrophic"]  = float((dists == 2).mean())


    # ── Print and save comparison table ──────────────────────────────
    table_lines = []
    table_lines.append("\n" + "=" * 100)
    table_lines.append(f"{'Metric':<30} {'Hier. CBM (Ours)':>22} {'Hier. CBM (Oracle)':>24} {'Flat LF-CBM':>18} {'CLIP Linear Probe':>20}")
    table_lines.append("=" * 100)

    def row(label, key, fmt=".4f", na="N/A"):
        vals = []
        for m in [hier, hier_oracle, flat, zs]:
            if m is None:
                vals.append(na)
            else:
                v = m.get(key)
                if isinstance(v, str):
                    vals.append(v)
                elif v is not None:
                    vals.append(f"{v:{fmt}}")
                else:
                    vals.append(na)
        table_lines.append(f"  {label:<28} {vals[0]:>22} {vals[1]:>24} {vals[2]:>18} {vals[3]:>20}")


    row("Top-1 Accuracy",             "top1_acc")
    row("Top-3 Accuracy",             "top3_acc")
    row("Avg Active Concepts",        "avg_active_concepts",    fmt=".1f", na="—")
    row("Avg LCA Error Distance",     "lca_dist_mean")
    row("% Sibling Mistake (LCA=1)",  "lca_pct_sibling",        fmt=".3f")
    row("% Catastrophic (LCA=2)",     "lca_pct_catastrophic",   fmt=".3f")
    table_lines.append("=" * 100)
    
    table_str = "\n".join(table_lines)
    print(table_str)

    with open(os.path.join(MODELS_DIR, "phase4_table.txt"), "w") as f:
        f.write(table_str)


    # Save results summary
    summary = {k: v for m in models if m for k, v in m.items()
               if k not in ("clf", "scaler", "y_pred", "y_true")}
    out_path = os.path.join(MODELS_DIR, "phase4_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\n✅ Full results saved to models/phase4_results.pkl")

if __name__ == "__main__":
    run_lca_eval()
