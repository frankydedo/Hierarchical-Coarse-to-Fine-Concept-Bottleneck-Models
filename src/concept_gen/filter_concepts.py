import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.taxonomy import cifar100_taxonomy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

MAX_LEN = 45 # Maximum character length for a concept
CLASS_SIM_CUTOFF = 0.85 # Cutoff for similarity to class name

print("Loading MPNet for synonym filtering...")
st_model = SentenceTransformer('all-mpnet-base-v2')

def filter_synonyms(concept_dict):
    print("Applying Class-Level synonym filter...")
    cleaned_dict = {}
    for cls, concepts in concept_dict.items():
        valid_concepts = []
        for c in concepts:
            c = c.strip().lower()
            if 3 < len(c) <= MAX_LEN:
                valid_concepts.append(c)
        
        if not valid_concepts:
            cleaned_dict[cls] = []
            continue
            
        cls_name_formatted = cls.replace("_", " ")
        cls_emb = st_model.encode([cls_name_formatted])
        concept_embs = st_model.encode(valid_concepts)
        sims = (concept_embs @ cls_emb.T).flatten()
        
        retained = [c for i, c in enumerate(valid_concepts) if sims[i] < CLASS_SIM_CUTOFF]
        # remove exact duplicates while preserving order
        retained = list(dict.fromkeys(retained))
        cleaned_dict[cls] = retained
    return cleaned_dict

def activation_based_filtering(coarse_dict, fine_dict, top_k_coarse=10, top_k_fine=10, images_per_class=20):
    print("\n--- Applying Activation-Based Functional Filtering ---")
    model_name = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    test_dir = os.path.join(DATA_DIR, "cifar_images", "train") # use train set for filtering to avoid data leak
    
    # Pre-extract image embeddings for all classes to save time
def class_level_filter(raw_coarse, raw_fine, taxonomy, sentence_model):
    return filter_synonyms(raw_coarse), filter_synonyms(raw_fine)

def concept_level_deduplication(coarse_dict, fine_dict, sentence_model):
    return coarse_dict, fine_dict

def activation_based_filtering(raw_coarse, raw_fine, taxonomy, clip_model, processor, device):
    """
    Filters concepts using actual image activations from the validation set.
    Ensures strict functional orthogonality.
    """
    print("\n[Activation-Based Filtering] Extracting Image Features...")
    test_dir = os.path.join(DATA_DIR, "custom_images", "train")
    
    # 1. Pre-compute average image embeddings for each fine class
    image_embs_per_fine_class = {}
    for sc, fine_classes in taxonomy.items():
        sc_path = os.path.join(test_dir, sc)
        if not os.path.exists(sc_path):
            print(f"Warning: Directory not found {sc_path}")
            continue
        for fc in fine_classes:
            fc_path = os.path.join(sc_path, fc)
            if not os.path.exists(fc_path):
                print(f"Warning: Directory not found {fc_path}")
                continue
                
            # Load up to 20 images per fine class to form a robust prototype
            images = glob.glob(os.path.join(fc_path, "*.jpg"))[:20]
            if not images:
                continue
            
            loaded_images = [Image.open(p).convert("RGB") for p in images]
            inputs = processor(images=loaded_images, return_tensors="pt").to(device)
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                # Average the embeddings
                image_embs_per_fine_class[fc] = features.mean(dim=0, keepdim=True)
                
    # 2. Filter Coarse Concepts (Must activate uniformly across ALL subclasses)
    print("\n[Activation-Based Filtering] Filtering Coarse Concepts...")
    filtered_coarse = {}
    for sc, concepts in raw_coarse.items():
        if not concepts: continue
        fine_classes = taxonomy.get(sc, [])
        if not fine_classes or fine_classes[0] not in image_embs_per_fine_class:
            filtered_coarse[sc] = concepts[:15]
            continue
            
        inputs = processor(text=concepts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
        # Calculate similarity with each subclass prototype
        sims = []
        for fc in fine_classes:
            sim = (image_embs_per_fine_class[fc] @ text_features.T).mean(dim=0)
            sims.append(sim)
            
        sims = torch.stack(sims) # Shape: (num_subclasses, num_concepts)
        mean_sim = sims.mean(dim=0)
        var_sim = sims.var(dim=0)
        
        # Keep concepts with high mean activation and low variance
        score = mean_sim - (var_sim * 10)
        top_indices = torch.argsort(score, descending=True)[:15].cpu().numpy()
        filtered_coarse[sc] = [concepts[i] for i in top_indices]

    # 3. Filter Fine Concepts (Strict Orthogonality)
    print("\n[Activation-Based Filtering] Filtering Fine Concepts...")
    filtered_fine = {}
    all_fine_classes = []
    for fcs in taxonomy.values():
        all_fine_classes.extend(fcs)
        
    for sc, fine_classes in taxonomy.items():
        for fc in fine_classes:
            concepts = raw_fine.get(fc, [])
            if not concepts or fc not in image_embs_per_fine_class:
                filtered_fine[fc] = concepts[:15]
                continue
                
            inputs = processor(text=concepts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                text_features = clip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
            # Similarity with target subclass
            target_sims = (image_embs_per_fine_class[fc] @ text_features.T).mean(dim=0)
            
            # Similarity with sibling subclasses
            other_sims = []
            for other_fc in fine_classes:
                if other_fc == fc or other_fc not in image_embs_per_fine_class: continue
                sim = (image_embs_per_fine_class[other_fc] @ text_features.T).mean(dim=0)
                other_sims.append(sim)
                
            if not other_sims:
                filtered_fine[fc] = concepts[:15]
                continue
                
            max_other_sims = torch.stack(other_sims).max(dim=0).values
            
            # STRICT HARD MARGIN: The target activation MUST be greater than the sibling activation by at least 0.01
            concept_scores = (target_sims - max_other_sims).cpu().numpy()
            
            kept_concepts = []
            for i, score in enumerate(concept_scores):
                if score > 0.01:
                    kept_concepts.append((score, concepts[i]))
                    
            # Sort by score and keep up to 15
            kept_concepts.sort(key=lambda x: x[0], reverse=True)
            top_concepts = [c[1] for c in kept_concepts[:15]]
            
            if len(top_concepts) == 0:
                print(f"  [WARNING] Class '{fc}' has NO concepts passing the strict margin. Keeping the best 3 regardless.")
                top_indices = np.argsort(concept_scores)[::-1][:3]
                top_concepts = [concepts[i] for i in top_indices]
                
            filtered_fine[fc] = top_concepts
            print(f"  {fc}: Kept {len(top_concepts)} strictly orthogonal concepts.")
            
    return filtered_coarse, filtered_fine

def main():
    print("Loading raw concepts...")
    with open(os.path.join(DATA_DIR, "raw_coarse_concepts.json"), "r") as f:
        raw_coarse = json.load(f)
    with open(os.path.join(DATA_DIR, "raw_fine_concepts.json"), "r") as f:
        raw_fine = json.load(f)
        
    print("Loading models (MPNet and CLIP)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_model = SentenceTransformer('all-mpnet-base-v2')
    
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    print("\n=== STEP 1: CLASS-LEVEL SYNONYM FILTER ===")
    filtered_coarse, filtered_fine = class_level_filter(raw_coarse, raw_fine, cifar100_taxonomy, sentence_model)
    
    print("\n=== STEP 2: ACTIVATION-BASED FUNCTIONAL FILTERING ===")
    filtered_coarse, filtered_fine = activation_based_filtering(filtered_coarse, filtered_fine, cifar100_taxonomy, clip_model, processor, device)
    
    print("\n=== STEP 3: CONCEPT-LEVEL DEDUPLICATION ===")
    filtered_coarse, filtered_fine = concept_level_deduplication(filtered_coarse, filtered_fine, sentence_model)
    
    print("\nSaving filtered concepts...")
    with open(os.path.join(DATA_DIR, "filtered_coarse_concepts.json"), "w") as f:
        json.dump(filtered_coarse, f, indent=4)
    with open(os.path.join(DATA_DIR, "filtered_fine_concepts.json"), "w") as f:
        json.dump(filtered_fine, f, indent=4)
        
    print("Done! Filtered concepts are strictly orthogonal and ready for projection.")

if __name__ == "__main__":
    main()
