# Project Implementation Plan: Constrained Coarse-to-Fine Prediction Mechanics

This document outlines an extensive, end-to-end implementation plan for developing a hierarchical, coarse-to-fine Concept Bottleneck Model (CBM) as described in **Path B**.

## 1. Project Objective
To build a CBM where the prediction of high-level (coarse) categories acts as a strict prerequisite or "gate" for the evaluation of fine-grained concepts and labels. The goal is to prove this structure improves interpretability, enforces logical consistency, and reduces the severity of classification mistakes.

---

## 2. Phase 1: Data Harvesting & Preparation
To evaluate a hierarchical model, we need a dataset with a clear, multi-level taxonomy. 

### Recommended Dataset: CIFAR-100 or a Custom Subset of ImageNet
- **Why CIFAR-100?** It natively possesses a 2-level hierarchy: 20 superclasses (e.g., *vehicles 1*, *fish*) and 100 fine-grained subclasses (e.g., *bicycle*, *trout*). 

### Concept Generation (Label-Free Approach)
Instead of relying on expensive human annotations, we will use the Label-Free CBM (LF-CBM) methodology:
1. **Coarse Concept Generation:** Query an LLM (e.g., GPT-4) to generate 10-15 distinguishing visual features for each of the 20 superclasses.
2. **Fine Concept Generation:** Query the LLM to generate 10-15 distinguishing visual features for each of the 100 subclasses, specifically asking it to differentiate subclasses within the *same* superclass.
3. **Concept Filtering:** Remove overly generic or overlapping concepts across branches to ensure the concept banks are distinct.

---

## 3. Phase 2: Exploratory Data Analysis (EDA)
Before building the model, we must validate the data and our generated concepts.
- **Concept Semantic Overlap:** Calculate the cosine similarity between the CLIP text embeddings of the coarse and fine concepts. We want to ensure that coarse concepts (e.g., "has wheels") form distinct clusters from fine concepts (e.g., "spoked bicycle wheels").
- **Baseline Feature Analysis:** Extract image embeddings using a pretrained backbone (e.g., CLIP ViT-B/16 or ResNet50) and perform t-SNE/PCA to visualize whether the superclasses naturally separate in the latent space.
- **Data Distribution:** Check for class imbalances at both the coarse and fine levels.

---

## 4. Phase 3: Methodology & Architecture
We will implement a dual-stage, gated Concept Bottleneck Model.

### Step A: The Coarse-Level Predictor (Level 1)
- **Concept Activation:** Compute the similarity between the input image embedding and the text embeddings of all **coarse concepts**.
- **Coarse Classification:** Pass the coarse concept activations through a sparse linear layer (trained with ElasticNet/L1 regularization) to predict the **superclass** (e.g., *Vehicle*).

### Step B: The Dynamic Gating Mechanism
- **The Mask:** Based on the predicted superclass from Level 1, generate a binary mask vector. 
- **Application:** This mask zeroes out all fine-grained concepts belonging to unpredicted superclasses. If the model predicts *Vehicle*, concepts like "has fur" or "gills" are strictly deactivated (set to 0).

### Step C: The Fine-Level Predictor (Level 2)
- **Filtered Activation:** The image is now evaluated only against the surviving (unmasked) **fine concepts**.
- **Fine Classification:** A second sparse linear layer predicts the final **subclass** (e.g., *Bicycle*) using only the filtered fine concepts.

---

## 5. Phase 4: Experiments & Evaluation Strategy

### Experiment 1: Predictive Performance
- **Metrics:** Top-1 Accuracy at both the coarse and fine levels. 
- **Baselines:** Compare against a standard "Flat" LF-CBM that uses all concepts simultaneously, and a standard ResNet/CLIP linear probe.

### Experiment 2: Interpretability and Sparsity
- **Metrics:** Average number of active concepts per image.
- **Hypothesis:** The gated architecture should drastically reduce the number of active concepts per image without sacrificing accuracy, leading to a much more readable explanation for the user.

### Experiment 3: Hierarchy-Aware Mistake Evaluation (Crucial for P1)
- **Metrics:** Hierarchical Cross-Entropy (HXE) or Lowest Common Ancestor (LCA) distance of errors.
- **Hypothesis:** Even if the overall accuracy is similar to a flat CBM, the mistakes made by the Coarse-to-Fine model should be "better" (e.g., confusing a bicycle with a motorcycle, rather than confusing a bicycle with a fish).

### Experiment 4: Intervenability
- **Simulation:** Simulate a human-in-the-loop expert who corrects the Coarse Level prediction when the model gets it wrong.
- **Hypothesis:** Correcting the coarse prediction will flip the gating mask, which should cause a massive, cascading improvement in the fine-level accuracy.

## 6. Phase 1 Technical Execution Plan

We will now proceed with Phase 1: generating the concept dictionaries using an LLM.

### Script Details: `generate_concepts.py`
We will create a Python script that automates the generation of concepts for CIFAR-100:
1. **Taxonomy Loading:** The script will define or load the CIFAR-100 taxonomy (20 superclasses, 100 subclasses).
2. **LLM Querying:** It will iterate through the superclasses and subclasses and send prompt templates to an LLM to extract distinguishing visual features.
   - *Coarse Prompt Example:* "List 15 distinct, visual features that are most important for recognizing the general category of '{superclass}'."
   - *Fine Prompt Example:* "List 15 distinct, visual features that are most important for recognizing a '{subclass}', specifically to differentiate it from other types of '{superclass}'."
3. **JSON Output:** The script will save the generated concepts into structured JSON files (`coarse_concepts.json` and `fine_concepts.json`).

### Post-Processing: `filter_concepts.py`
A subsequent script will clean the generated concepts by:
- Removing overly long phrases.
- Deduplicating exact matches.
- Using a lightweight embedding model (e.g., `all-mpnet-base-v2`) to remove concepts that are semantically identical to the class names or highly redundant.
