"""
Step 4.3 — Project onto Concept Space
Computes dot-product similarity between image embeddings and concept embeddings.
Output: coarse_scores and fine_scores for every image in train/test.
"""
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def project_split(split, concept_bank):
    print(f"\n--- Projecting {split} features ---")
    features = torch.load(os.path.join(DATA_DIR, f"{split}_features.pt"), weights_only=False)

    embeddings = features["embeddings"]               # (N, 512)
    coarse_matrix = concept_bank["coarse_matrix"]    # (N_coarse, 512)
    fine_matrix = concept_bank["fine_matrix"]        # (N_fine, 512)

    # Dot product: (N, 512) @ (512, N_coarse) → (N, N_coarse)
    coarse_scores = embeddings @ coarse_matrix.T
    fine_scores   = embeddings @ fine_matrix.T

    print(f"  Embeddings:     {embeddings.shape}")
    print(f"  Coarse scores:  {coarse_scores.shape}")
    print(f"  Fine scores:    {fine_scores.shape}")

    projected = {
        "coarse_scores":      coarse_scores,
        "fine_scores":        fine_scores,
        "coarse_labels":      features["coarse_labels"],
        "fine_labels":        features["fine_labels"],
        "fine_class_names":   features["fine_class_names"],
        "superclasses":       features["superclasses"],
        "fine_classes":       features["fine_classes"],
    }

    save_path = os.path.join(DATA_DIR, f"{split}_projected.pt")
    torch.save(projected, save_path)
    print(f"  ✅ Saved → {save_path}")

def main():
    print("Loading concept bank...")
    concept_bank = torch.load(os.path.join(DATA_DIR, "concept_bank.pt"), weights_only=False)
    print(f"  Coarse bank: {concept_bank['coarse_matrix'].shape}")
    print(f"  Fine bank:   {concept_bank['fine_matrix'].shape}")

    project_split("train", concept_bank)
    project_split("test", concept_bank)
    print("\n✅ Projection complete.")

if __name__ == "__main__":
    main()
