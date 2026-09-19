"""
Step 4.1 — Build Concept Bank
Encodes all filtered concepts as unit-normalized CLIP text embeddings and
saves them to data/concept_bank.pt for downstream projection and training.
"""
import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.taxonomy import cifar100_taxonomy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def build_concept_bank():
    print("Loading filtered concepts...")
    with open(os.path.join(DATA_DIR, "filtered_coarse_concepts.json")) as f:
        coarse_dict = json.load(f)
    with open(os.path.join(DATA_DIR, "filtered_fine_concepts.json")) as f:
        fine_dict = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    superclasses = list(cifar100_taxonomy.keys())
    sc_to_idx = {sc: i for i, sc in enumerate(superclasses)}

    # --- Build coarse concept bank ---
    coarse_concepts_all = []
    coarse_concept_names = []
    coarse_concept_sc_ids = []  # which superclass each coarse concept belongs to

    for sc in superclasses:
        concepts = coarse_dict.get(sc, [])
        coarse_concept_names.extend(concepts)
        coarse_concept_sc_ids.extend([sc_to_idx[sc]] * len(concepts))
        coarse_concepts_all.extend(concepts)

    # --- Build fine concept bank ---
    fine_concepts_all = []
    fine_concept_names = []
    fine_concept_sc_ids = []   # which superclass each fine concept belongs to
    fine_concept_fc_names = [] # which fine class each concept belongs to

    for sc, fine_classes in cifar100_taxonomy.items():
        for fc in fine_classes:
            concepts = fine_dict.get(fc, [])
            fine_concept_names.extend(concepts)
            fine_concept_sc_ids.extend([sc_to_idx[sc]] * len(concepts))
            fine_concept_fc_names.extend([fc] * len(concepts))
            fine_concepts_all.extend(concepts)

    def encode_concepts(concepts):
        """Encode a list of concept strings into unit-normalized embeddings."""
        embeddings = []
        batch_size = 64
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i+batch_size]
            inputs = processor(
                text=batch, return_tensors="pt",
                padding=True, truncation=True
            ).to(device)
            with torch.no_grad():
                emb = clip_model.get_text_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    print(f"Encoding {len(coarse_concepts_all)} coarse concepts...")
    coarse_matrix = encode_concepts(coarse_concepts_all)

    print(f"Encoding {len(fine_concepts_all)} fine concepts...")
    fine_matrix = encode_concepts(fine_concepts_all)

    # Save the concept bank
    concept_bank = {
        # Embedding matrices
        "coarse_matrix": coarse_matrix,           # (N_coarse, 512)
        "fine_matrix": fine_matrix,               # (N_fine, 512)
        # Concept names (for interpretability)
        "coarse_concept_names": coarse_concept_names,
        "fine_concept_names": fine_concept_names,
        "fine_concept_fc_names": fine_concept_fc_names,
        # Mapping from concept index to superclass index (for masking)
        "coarse_concept_sc_ids": torch.tensor(coarse_concept_sc_ids, dtype=torch.long),
        "fine_concept_sc_ids": torch.tensor(fine_concept_sc_ids, dtype=torch.long),
        # Label mappings
        "superclasses": superclasses,
        "fine_classes": [fc for sc in superclasses for fc in cifar100_taxonomy[sc]],
        "sc_to_idx": sc_to_idx,
    }

    save_path = os.path.join(DATA_DIR, "concept_bank.pt")
    torch.save(concept_bank, save_path)

    print(f"\n✅ Concept Bank saved to {save_path}")
    print(f"   Coarse concept matrix shape: {coarse_matrix.shape}")
    print(f"   Fine concept matrix shape:   {fine_matrix.shape}")
    print(f"   Superclasses ({len(superclasses)}): {superclasses}")

if __name__ == "__main__":
    build_concept_bank()
