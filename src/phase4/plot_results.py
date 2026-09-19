"""
Phase 4 — Research Visualizations
Generates publication-quality figures highlighting the key findings:
  Fig 1 — Accuracy vs Interpretability trade-off (4 models)
  Fig 2 — LCA Error Distribution: stacked bars (fails gracefully)
  Fig 3 — Oracle Intervention Effect: before/after human correction
  Fig 4 — Concept Sparsity: active concepts per model heatmap breakdown
  Fig 5 — Per-superclass Error Profile: catastrophic rate per category
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.taxonomy import cifar100_taxonomy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIG_DIR      = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Colour palette (accessible, academic) ────────────────────────
C_OURS   = "#2563EB"   # blue  — Hier. CBM
C_ORACLE = "#0EA5E9"   # light blue — Oracle
C_FLAT   = "#F59E0B"   # amber — Flat LF-CBM
C_CLIP   = "#6B7280"   # grey  — CLIP (black-box)
C_ERR1   = "#FCA5A5"   # rose  — sibling mistake
C_ERR2   = "#DC2626"   # red   — catastrophic
C_OK     = "#86EFAC"   # green — correct

STYLE = {"font.family": "DejaVu Sans", "axes.spines.top": False,
          "axes.spines.right": False, "axes.grid": True,
          "grid.alpha": 0.3, "grid.linestyle": "--",
          "figure.facecolor": "white", "axes.facecolor": "white"}
plt.rcParams.update(STYLE)

SUPERCLASSES     = list(cifar100_taxonomy.keys())
FINE_CLASSES_ORD = [fc for sc in SUPERCLASSES for fc in cifar100_taxonomy[sc]]
FINE_TO_SUPER    = {}
for sc_idx, (sc, fcs) in enumerate(cifar100_taxonomy.items()):
    for fc in fcs:
        FINE_TO_SUPER[FINE_CLASSES_ORD.index(fc)] = sc_idx

def lca_distance(y_true, y_pred):
    dists = []
    for t, p in zip(y_true, y_pred):
        if t == p:                                   dists.append(0)
        elif FINE_TO_SUPER[t] == FINE_TO_SUPER[p]:  dists.append(1)
        else:                                         dists.append(2)
    return np.array(dists)

# ── Load all predictions ─────────────────────────────────────────
def load_all():
    test = torch.load(os.path.join(DATA_DIR, "test_projected.pt"), weights_only=False)
    with open(os.path.join(MODELS_DIR, "classifiers.pkl"), "rb") as f:
        ckpt = pickle.load(f)

    coarse_clf, coarse_scaler = ckpt["coarse_clf"], ckpt["coarse_scaler"]
    fine_clf,   fine_scaler   = ckpt["fine_clf"],   ckpt["fine_scaler"]
    mask = ckpt["fine_mask"]

    y_true        = test["fine_labels"].numpy()
    y_coarse_true = test["coarse_labels"].numpy()
    X_coarse_raw  = test["coarse_scores"].numpy()
    X_fine_raw    = test["fine_scores"].numpy()

    # Standard inference
    coarse_pred    = coarse_clf.predict(coarse_scaler.transform(X_coarse_raw))
    X_fine_masked  = X_fine_raw * mask[:, coarse_pred].T
    fine_pred      = fine_clf.predict(fine_scaler.transform(X_fine_masked))
    fine_probs     = fine_clf.predict_proba(fine_scaler.transform(X_fine_masked))

    # Oracle inference
    X_fine_oracle  = X_fine_raw * mask[:, y_coarse_true].T
    oracle_pred    = fine_clf.predict(fine_scaler.transform(X_fine_oracle))
    oracle_probs   = fine_clf.predict_proba(fine_scaler.transform(X_fine_oracle))

    # Flat baseline
    with open(os.path.join(MODELS_DIR, "flat_baseline.pkl"), "rb") as f:
        flat = pickle.load(f)

    # CLIP baseline
    with open(os.path.join(MODELS_DIR, "zeroshot_baseline.pkl"), "rb") as f:
        zs = pickle.load(f)

    # Sparsity
    n_coarse_act = (coarse_clf.coef_ != 0).sum(axis=1).mean()
    n_fine_act   = (fine_clf.coef_   != 0).sum(axis=1).mean()
    n_flat_act   = (flat["clf"].coef_ != 0).sum(axis=1).mean()

    return {
        "y_true":       y_true,
        "hier_pred":    fine_pred,
        "hier_probs":   fine_probs,
        "oracle_pred":  oracle_pred,
        "flat_pred":    flat["y_pred"],
        "flat_ytrue":   flat["y_true"],
        "zs_pred":      zs["y_pred"],
        "zs_ytrue":     zs["y_true"],
        "n_coarse_act": n_coarse_act,
        "n_fine_act":   n_fine_act,
        "n_flat_act":   n_flat_act,
        "coarse_pred":  coarse_pred,
        "y_coarse_true":y_coarse_true,
    }


# ════════════════════════════════════════════════════════════════════
# FIG 1 — Accuracy vs Interpretability Trade-off
# ════════════════════════════════════════════════════════════════════
def fig1_accuracy_tradeoff(d):
    fig, ax = plt.subplots(figsize=(9, 5))

    models  = ["Hier. CBM\n(Ours)", "Hier. CBM\n(Oracle)", "Flat LF-CBM", "CLIP\nLinear Probe"]
    top1    = [
        np.mean(d["hier_pred"] == d["y_true"]),
        np.mean(d["oracle_pred"] == d["y_true"]),
        np.mean(d["flat_pred"] == d["flat_ytrue"]),
        np.mean(d["zs_pred"] == d["zs_ytrue"]),
    ]
    colors  = [C_OURS, C_ORACLE, C_FLAT, C_CLIP]
    interp  = ["✓ Interpretable\n✓ Hierarchical",
                "✓ Interpretable\n✓ Hierarchical\n+ Human Correction",
                "✓ Interpretable\n✗ Hierarchical",
                "✗ Black-Box"]

    x = np.arange(len(models))
    bars = ax.bar(x, top1, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=1.5)

    for bar, val, note in zip(bars, top1, interp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, 0.03,
                note, ha="center", va="bottom", fontsize=7.5, color="white",
                multialignment="center", linespacing=1.4)

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Top-1 Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy vs Interpretability Trade-off", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_accuracy_tradeoff.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Fig 1 saved → {os.path.relpath(path, PROJECT_ROOT)}")


# ════════════════════════════════════════════════════════════════════
# FIG 2 — LCA Error Distribution (Stacked Bars)
# ════════════════════════════════════════════════════════════════════
def fig2_lca_stacked(d):
    fig, ax = plt.subplots(figsize=(9, 5))

    models = ["Hier. CBM\n(Ours)", "Hier. CBM\n(Oracle)", "Flat LF-CBM", "CLIP\nLinear Probe"]
    preds  = [d["hier_pred"], d["oracle_pred"], d["flat_pred"], d["zs_pred"]]
    trues  = [d["y_true"],    d["y_true"],      d["flat_ytrue"], d["zs_ytrue"]]

    pct_ok, pct_s1, pct_s2 = [], [], []
    for yp, yt in zip(preds, trues):
        lca = lca_distance(yt, yp)
        pct_ok.append((lca == 0).mean())
        pct_s1.append((lca == 1).mean())
        pct_s2.append((lca == 2).mean())

    x = np.arange(len(models))
    w = 0.5
    b1 = ax.bar(x, pct_ok, w, label="Correct (LCA=0)",            color=C_OK,   zorder=3)
    b2 = ax.bar(x, pct_s1, w, bottom=pct_ok,                       label="Sibling (LCA=1)",     color=C_ERR1, zorder=3)
    b3 = ax.bar(x, pct_s2, w, bottom=np.array(pct_ok)+np.array(pct_s1), label="Catastrophic (LCA=2)", color=C_ERR2, zorder=3)

    # Labels on catastrophic bars
    for i, (p_ok, p_s1, p_s2) in enumerate(zip(pct_ok, pct_s1, pct_s2)):
        if p_s2 > 0:
            ax.text(x[i], p_ok + p_s1 + p_s2/2, f"{p_s2:.1%}",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        ax.text(x[i], p_ok/2, f"{p_ok:.1%}",
                ha="center", va="center", fontsize=9, color="#166534")

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Proportion of test samples", fontsize=12)
    ax.set_ylim(0, 1.04)
    ax.set_title("Hierarchy-Aware Error Distribution (LCA)", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_lca_distribution.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Fig 2 saved → {os.path.relpath(path, PROJECT_ROOT)}")


# ════════════════════════════════════════════════════════════════════
# FIG 3 — Oracle Intervention Effect (Before / After)
# ════════════════════════════════════════════════════════════════════
def fig3_oracle_intervention(d):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, title, y_pred, color in [
        (axes[0], "Standard Inference\n(No Intervention)",  d["hier_pred"],   C_OURS),
        (axes[1], "Oracle Intervention\n(Human corrects Coarse)", d["oracle_pred"], C_ORACLE),
    ]:
        lca = lca_distance(d["y_true"], y_pred)
        sizes  = [(lca==0).mean(), (lca==1).mean(), (lca==2).mean()]
        labels = [f"Correct\n{sizes[0]:.1%}", f"Sibling\nMistake\n{sizes[1]:.1%}",
                  f"Catastrophic\n{sizes[2]:.1%}"]
        explode = [0, 0.05, 0.12]
        wedge_colors = [C_OK, C_ERR1, C_ERR2]

        wedges, _ = ax.pie(sizes, labels=None, colors=wedge_colors,
                           explode=explode, startangle=90,
                           wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.18),
                  fontsize=9, framealpha=0.9, ncol=1)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    fig.suptitle("Effect of Human Intervention at Coarse Level (Oracle Test)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_oracle_intervention.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Fig 3 saved → {os.path.relpath(path, PROJECT_ROOT)}")


# ════════════════════════════════════════════════════════════════════
# FIG 4 — Concept Sparsity: bubble chart (accuracy vs concepts)
# ════════════════════════════════════════════════════════════════════
def fig4_sparsity_bubble(d):
    fig, ax = plt.subplots(figsize=(9, 6))

    accs = [
        np.mean(d["hier_pred"]  == d["y_true"]),
        np.mean(d["oracle_pred"] == d["y_true"]),
        np.mean(d["flat_pred"]  == d["flat_ytrue"]),
        np.mean(d["zs_pred"]    == d["zs_ytrue"]),
    ]
    active = [
        (d["n_coarse_act"] + d["n_fine_act"]) / 2,
        d["n_fine_act"],
        d["n_flat_act"],
        512,
    ]
    vocabs  = [769, 469, 769, 512]
    labels  = ["Hier. CBM\n(Ours)", "Hier. CBM\n(Oracle)", "Flat LF-CBM", "CLIP\nLinear Probe"]
    colors  = [C_OURS, C_ORACLE, C_FLAT, C_CLIP]

    for acc, act, vocab, color in zip(accs, active, vocabs, colors):
        ax.scatter(act, acc, s=vocab * 0.8, color=color, alpha=0.85,
                   edgecolors="white", linewidth=2, zorder=4)

    # Manual, non-overlapping annotation positions (xytext in data coords)
    annot_params = [
        # (point_x, point_y, text_x, text_y, ha)
        (active[0], accs[0], 140, 0.845, "left"),    # Hier. CBM Ours — right+down
        (active[1], accs[1], 140, 0.900, "left"),    # Oracle — right
        (active[2], accs[2], 140, 0.955, "left"),    # Flat — right+up
        (active[3], accs[3], 360, 0.975, "left"),    # CLIP — above-left of bubble
    ]

    for (px, py, tx, ty, ha), label, color in zip(annot_params, labels, colors):
        ax.annotate(label,
                    xy=(px, py), xytext=(tx, ty),
                    ha=ha, va="center", fontsize=9.5,
                    fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="-", color=color,
                                   lw=1.0, alpha=0.6,
                                   connectionstyle="arc3,rad=0.0"),
                    zorder=5)

    ax.set_xlabel("Avg Active Concepts at Inference", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy", fontsize=12)
    ax.set_title("Sparsity vs Accuracy Trade-off\n(bubble size ∝ concept vocabulary size)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(-30, 620)
    ax.set_ylim(0.82, 1.00)

    # Interpretability zone
    ax.axvspan(-30, 55, alpha=0.06, color=C_OURS, zorder=0)
    ax.text(-15, 0.998, "← High\nInterpretability\nZone", fontsize=7.5,
            color=C_OURS, ha="center", va="top", style="italic", multialignment="center")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_sparsity_bubble.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Fig 4 saved → {os.path.relpath(path, PROJECT_ROOT)}")


# ════════════════════════════════════════════════════════════════════
# FIG 5 — Per-superclass Catastrophic Error Rate
# ════════════════════════════════════════════════════════════════════
def fig5_per_superclass(d):
    fig, ax = plt.subplots(figsize=(13, 5))

    # Compute per-superclass catastrophic rate for Hier. CBM vs Flat
    cats_hier, cats_flat = [], []
    for sc_idx in range(len(SUPERCLASSES)):
        mask_sc = np.array([FINE_TO_SUPER[t] == sc_idx for t in d["y_true"]])
        if mask_sc.sum() == 0:
            cats_hier.append(0); cats_flat.append(0); continue

        # Hier
        lca_h = lca_distance(d["y_true"][mask_sc], d["hier_pred"][mask_sc])
        cats_hier.append((lca_h == 2).mean())

        # Flat — need to match y_true to flat's y_true
        mask_flat = np.array([FINE_TO_SUPER[t] == sc_idx for t in d["flat_ytrue"]])
        lca_f = lca_distance(d["flat_ytrue"][mask_flat], d["flat_pred"][mask_flat])
        cats_flat.append((lca_f == 2).mean())

    sc_labels = [sc.replace("_", "\n") for sc in SUPERCLASSES]
    x = np.arange(len(SUPERCLASSES))
    w = 0.38

    ax.bar(x - w/2, cats_flat, w, label="Flat LF-CBM", color=C_FLAT, alpha=0.85, zorder=3)
    ax.bar(x + w/2, cats_hier, w, label="Hier. CBM (Ours)", color=C_OURS, alpha=0.85, zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(sc_labels, fontsize=6.5, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("% Catastrophic Mistakes (LCA=2)", fontsize=11)
    ax.set_title("Catastrophic Error Rate per Superclass: Flat LF-CBM vs Hier. CBM",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_per_superclass_catastrophic.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Fig 5 saved → {os.path.relpath(path, PROJECT_ROOT)}")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4 — Generating Research Visualizations")
    print("=" * 60)
    print("\nLoading model predictions...")
    data = load_all()
    print(f"  [OK] {len(data['y_true'])} test samples loaded.\n")

    print("Generating figures...")
    fig1_accuracy_tradeoff(data)
    fig2_lca_stacked(data)
    fig3_oracle_intervention(data)
    fig4_sparsity_bubble(data)
    fig5_per_superclass(data)

    print(f"\n✅ All figures saved to docs/figures/")
