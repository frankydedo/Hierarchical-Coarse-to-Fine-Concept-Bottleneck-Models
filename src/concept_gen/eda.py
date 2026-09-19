import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.manifold import TSNE
from transformers import CLIPProcessor, CLIPModel
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.taxonomy import cifar100_taxonomy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def check_data_distribution():
    print("Checking Custom High-Res dataset distribution...")
    data_dir = os.path.join(DATA_DIR, "custom_images")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"ERROR: Image directories not found in {data_dir}")
        return False
        
    for sc, fine_classes in cifar100_taxonomy.items():
        for fc in fine_classes:
            train_fc = os.path.join(train_dir, sc, fc)
            test_fc = os.path.join(test_dir, sc, fc)
            
            n_train = len(glob.glob(os.path.join(train_fc, "*.jpg"))) if os.path.exists(train_fc) else 0
            n_test = len(glob.glob(os.path.join(test_fc, "*.jpg"))) if os.path.exists(test_fc) else 0
            
            if n_train == 0 or n_test == 0:
                print(f"WARNING: Missing images for {fc}: train={n_train}, test={n_test}")
                
    total_classes = sum(len(fcs) for fcs in cifar100_taxonomy.values())
    print(f"Data distribution check complete for {total_classes} classes.")
    return True

def concept_activation_validation():
    print("Performing Concept Activation Validation (Heatmap) on the full custom taxonomy...")
    sub_sc = list(cifar100_taxonomy.keys())
    
    fine_classes = []
    for sc in sub_sc:
        fine_classes.extend(cifar100_taxonomy[sc])
        
    with open(os.path.join(DATA_DIR, "filtered_coarse_concepts.json"), "r") as f:
        coarse_dict = json.load(f)
    with open(os.path.join(DATA_DIR, "filtered_fine_concepts.json"), "r") as f:
        fine_dict = json.load(f)
        
    relevant_coarse_concepts = []
    coarse_labels = []
    for sc in sub_sc:
        c = coarse_dict.get(sc, [])
        relevant_coarse_concepts.extend(c)
        coarse_labels.extend([f"C: {sc}"] * len(c))
        
    relevant_fine_concepts = []
    fine_labels = []
    for fc in fine_classes:
        c = fine_dict.get(fc, [])
        relevant_fine_concepts.extend(c)
        fine_labels.extend([f"F: {fc}"] * len(c))
        
    all_concepts = relevant_coarse_concepts + relevant_fine_concepts
    if not all_concepts:
        print("No concepts found to plot!")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print(f"Extracting text embeddings for {len(all_concepts)} concepts...")
    text_embeddings = []
    batch_size = 32
    for i in range(0, len(all_concepts), batch_size):
        batch = all_concepts[i:i+batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb = clip_model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            text_embeddings.append(emb)
    text_embeddings = torch.cat(text_embeddings)
    
    print("Extracting image embeddings...")
    test_dir = os.path.join(DATA_DIR, "custom_images", "test")
    
    activation_matrix = np.zeros((len(fine_classes), len(all_concepts)))
    
    for i, fc in enumerate(fine_classes):
        sc = next(k for k, v in cifar100_taxonomy.items() if fc in v)
        fc_path = os.path.join(test_dir, sc, fc)
        images = glob.glob(os.path.join(fc_path, "*.jpg"))[:10]
        if not images: continue
        
        loaded_images = []
        for p in images:
            try:
                img = Image.open(p).convert("RGB")
                img.load()
                img_arr = np.array(img)
                if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3 and img_arr.shape[0] > 10 and img_arr.shape[1] > 10:
                    loaded_images.append(img)
            except Exception:
                pass
        
        if not loaded_images: continue
        inputs = processor(images=loaded_images, return_tensors="pt").to(device)
        with torch.no_grad():
            img_emb = clip_model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            sims = (img_emb @ text_embeddings.T).mean(dim=0).cpu().numpy()
            activation_matrix[i, :] = sims
            
    print("Plotting heatmap...")
    plt.figure(figsize=(16, 8))
    sns.heatmap(activation_matrix, 
                cmap="viridis", 
                xticklabels=False, 
                yticklabels=fine_classes,
                cbar_kws={'label': 'Mean CLIP Similarity'})
                
    # Add vertical lines to separate coarse and fine concept blocks
    current_idx = len(relevant_coarse_concepts)
    plt.axvline(x=current_idx, color='white', linestyle='--', linewidth=2)
    
    plt.title("Concept Activation Validation (Custom High-Res Dataset)")
    plt.xlabel("Concepts (Left: Coarse, Right: Fine)")
    plt.ylabel("Test Images (By Subclass)")
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, "activation_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved activation heatmap to {save_path}")

def baseline_visual_features():
    print("Extracting baseline visual features for custom dataset...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    test_dir = os.path.join(DATA_DIR, "custom_images", "test")
    all_features = []
    labels = []
    
    for sc, subclasses in cifar100_taxonomy.items():
        for fc in subclasses:
            fc_path = os.path.join(test_dir, sc, fc)
            images = glob.glob(os.path.join(fc_path, "*.jpg"))[:10]
            if not images: continue
            
            loaded_images = []
            for p in images:
                try:
                    img = Image.open(p).convert("RGB")
                    img.load()
                    img_arr = np.array(img)
                    if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3 and img_arr.shape[0] > 10 and img_arr.shape[1] > 10:
                        loaded_images.append(img)
                except Exception:
                    pass
            
            if not loaded_images: continue
            inputs = processor(images=loaded_images, return_tensors="pt").to(device)
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                all_features.append(features.cpu().numpy())
                labels.extend([sc] * len(loaded_images))
                
    if not all_features: return
    features_np = np.vstack(all_features)
    
    print("Running t-SNE on visual features...")
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    features_2d = tsne.fit_transform(features_np)
    
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=label, color=color, alpha=0.8)
        
    plt.title("Visual Feature Space (Custom High-Res Dataset) grouped by Superclass")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, "visual_tsne.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visual features t-SNE to {save_path}")

if __name__ == "__main__":
    if check_data_distribution():
        concept_activation_validation()
        baseline_visual_features()
        print("Phase 2 EDA complete.")
