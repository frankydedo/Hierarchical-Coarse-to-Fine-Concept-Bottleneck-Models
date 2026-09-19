"""
Step 4.2 — Feature Extraction
Passes all images through CLIP-ViT-B/32 and saves unit-normalized image
embeddings alongside their coarse and fine class labels.
"""
import os
import glob
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import CLIPProcessor, CLIPModel
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.taxonomy import cifar100_taxonomy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def extract_split(split, clip_model, processor, device):
    print(f"\n--- Extracting {split} features ---")
    images_dir = os.path.join(DATA_DIR, "custom_images", split)

    superclasses = list(cifar100_taxonomy.keys())
    fine_classes_ordered = [fc for sc in superclasses for fc in cifar100_taxonomy[sc]]
    sc_to_idx = {sc: i for i, sc in enumerate(superclasses)}
    fc_to_idx = {fc: i for i, fc in enumerate(fine_classes_ordered)}

    all_embeddings = []
    all_coarse_labels = []
    all_fine_labels = []
    all_fine_class_names = []

    for sc in superclasses:
        for fc in cifar100_taxonomy[sc]:
            fc_path = os.path.join(images_dir, sc, fc)
            images = glob.glob(os.path.join(fc_path, "*.jpg"))

            if not images:
                print(f"  [WARNING] No images found for {fc} in {split} split!")
                continue

            print(f"  {sc}/{fc}: {len(images)} images")

            # Process in batches of 16 to avoid OOM
            batch_size = 16
            for i in range(0, len(images), batch_size):
                batch_paths = images[i:i+batch_size]
                loaded = []
                import numpy as np
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        img.load()
                        img_arr = np.array(img)
                        if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3 and img_arr.shape[0] > 10 and img_arr.shape[1] > 10:
                            loaded.append(img)
                    except Exception as e:
                        pass
                
                if not loaded:
                    continue

                inputs = processor(images=loaded, return_tensors="pt").to(device)
                with torch.no_grad():
                    emb = clip_model.get_image_features(**inputs)
                    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                    all_embeddings.append(emb.cpu())

                n = len(loaded)
                all_coarse_labels.extend([sc_to_idx[sc]] * n)
                all_fine_labels.extend([fc_to_idx[fc]] * n)
                all_fine_class_names.extend([fc] * n)

    features = {
        "embeddings": torch.cat(all_embeddings, dim=0),        # (N, 512)
        "coarse_labels": torch.tensor(all_coarse_labels, dtype=torch.long),  # (N,)
        "fine_labels": torch.tensor(all_fine_labels, dtype=torch.long),      # (N,)
        "fine_class_names": all_fine_class_names,
        "superclasses": superclasses,
        "fine_classes": fine_classes_ordered,
    }

    save_path = os.path.join(DATA_DIR, f"{split}_features.pt")
    torch.save(features, save_path)
    print(f"  ✅ Saved {features['embeddings'].shape[0]} embeddings → {save_path}")
    return features

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    extract_split("train", clip_model, processor, device)
    extract_split("test", clip_model, processor, device)
    print("\n✅ Feature extraction complete.")

if __name__ == "__main__":
    main()
