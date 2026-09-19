"""
Phase 4 — Zero-Shot CLIP Black-Box Baseline
Uses CLIP's raw visual features (512-dim) with a Linear Probe 
to predict the 100 fine classes. No concepts, no interpretability.
Serves as the non-interpretable performance upper-bound.
"""
import os
import sys
import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageFile
import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.taxonomy import cifar100_taxonomy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
IMAGES_DIR   = os.path.join(PROJECT_ROOT, "data", "custom_images")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Build ordered class list (same order as our label encoding) ──
SUPERCLASSES     = list(cifar100_taxonomy.keys())
FINE_CLASSES_ORD = [fc for sc in SUPERCLASSES for fc in cifar100_taxonomy[sc]]

def extract_clip_visual_features(model, processor, split="train"):
    """Extract CLIP visual embeddings for all images in the given split."""
    feats, labels = [], []
    split_dir = os.path.join(IMAGES_DIR, split)
    for sc_dir in sorted(os.listdir(split_dir)):
        sc_path = os.path.join(split_dir, sc_dir)
        if not os.path.isdir(sc_path):
            continue
        for fc_dir in sorted(os.listdir(sc_path)):
            fc_path = os.path.join(sc_path, fc_dir)
            if not os.path.isdir(fc_path):
                continue
            # Match fc_dir to FINE_CLASSES_ORD (strip underscores vs spaces)
            fc_norm = fc_dir.replace("_", " ").lower()
            matches = [i for i, fc in enumerate(FINE_CLASSES_ORD)
                       if fc.lower() == fc_norm or fc.lower() == fc_dir.lower()]
            if not matches:
                continue
            fc_global = matches[0]
            paths = sorted(glob.glob(os.path.join(fc_path, "*.jpg")) +
                           glob.glob(os.path.join(fc_path, "*.png")) +
                           glob.glob(os.path.join(fc_path, "*.jpeg")))
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    img.load()
                except Exception:
                    continue
                arr = np.array(img, dtype=np.uint8)
                if arr.ndim != 3 or arr.shape[2] != 3:
                    continue
                try:
                    inputs = processor(images=arr, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        emb = model.get_image_features(**inputs)
                except Exception:
                    continue
                emb = emb / emb.norm(dim=-1, keepdim=True)
                feats.append(emb.cpu().squeeze().numpy())
                labels.append(fc_global)
    if len(feats) == 0:
        return np.array([]), np.array([])
    return np.vstack(feats), np.array(labels)


def run_zeroshot_baseline():
    print("=" * 60)
    print("PHASE 4 — Zero-Shot / Linear Probe CLIP Baseline")
    print("=" * 60)

    print("\nLoading CLIP model...")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    print("Extracting train visual features from disk...")
    X_train, y_train = extract_clip_visual_features(model, processor, split="train")
    print(f"  Train: {X_train.shape}")
    print("Extracting test visual features from disk...")
    X_test,  y_test  = extract_clip_visual_features(model, processor, split="test")
    print(f"  Test:  {X_test.shape}")

    # ── Linear Probe ──────────────────────────────────────────────
    print("\nTraining Linear Probe on CLIP visual features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(
        penalty='l2',        # L2 for black-box; no interpretability goal
        solver='lbfgs',
        C=1.0,
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train_s, y_train)

    fine_pred  = clf.predict(X_test_s)
    fine_probs = clf.predict_proba(X_test_s)
    top1_acc   = accuracy_score(y_test, fine_pred)
    top3_acc   = np.mean([y_test[i] in np.argsort(fine_probs[i])[-3:]
                          for i in range(len(y_test))])

    print(f"\n[CLIP Linear Probe] Top-1 Accuracy: {top1_acc:.4f}")
    print(f"[CLIP Linear Probe] Top-3 Accuracy: {top3_acc:.4f}")
    print("  (Non-interpretable: uses raw 512-dim pixel embeddings, not human concepts)")

    results = {
        "model": "CLIP Linear Probe",
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "avg_active_concepts": None,   # Not applicable
        "total_concepts": 512,         # Raw embedding dim
        "y_pred": fine_pred,
        "y_true": y_test,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "zeroshot_baseline.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"\n✅ Results saved to models/zeroshot_baseline.pkl")
    return results

if __name__ == "__main__":
    run_zeroshot_baseline()
