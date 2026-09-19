---
title: "Interpretable Hierarchical Concept Reasoning through Attention-Guided Graph Learning"
authors: ["David Debot", "Pietro Barbiero", "Gabriele Dominici", "Giuseppe Marra"]
year: 2025
tags: [paper]
---

# Interpretable Hierarchical Concept Reasoning through Attention-Guided Graph Learning

## Summary
Concept-Based Models (CBMs) typically explain task predictions using human-interpretable concepts, but the prediction of the concepts themselves is usually a black-box process. This paper introduces the [[Hierarchical Concept Memory Reasoner]] (H-CMR), a novel CBM that provides interpretability for *both* concept and task predictions. H-CMR learns a directed acyclic graph (DAG) where nodes are concepts (and tasks) and edges represent symbolic logic rules stored in a learned memory. During inference, a neural attention mechanism selects the most relevant logic rules, which are then hierarchically executed to predict concepts. H-CMR matches the expressivity and accuracy of black-box models while enabling powerful human interventions at both inference and training times.

## Key Contributions
- **[[Hierarchical Concept Memory Reasoner]] (H-CMR):** The first universal CBM that provides interpretability at both the concept and task levels.
- **Rule-based Concept Prediction:** Concepts are not predicted directly by a neural network (except for source concepts), but rather inferred through symbolic execution of learned logic rules (e.g., $C_3 \leftarrow C_1 \land C_2$).
- **Expressivity Guarantee:** Formally proves that H-CMR is a universal binary classifier, capable of achieving high task accuracy regardless of the chosen concept set.
- **Enhanced Intervenability:** 
  - *Concept Interventions (Inference):* Because concepts are not conditionally independent, correcting a parent concept logically propagates to child concepts, making interventions highly effective.
  - *Model Interventions (Training/Setup):* Humans can inject prior knowledge by manually adding rules, enforcing graph structures (like forcing a concept to be a sink/source), or editing learned rules.

## Insights
- Treating concepts as conditionally independent (like most standard CBMs do) severely limits the downstream impact of human interventions at test time. Learning and utilizing the hierarchical dependencies between concepts solves this.
- Providing interpretability doesn't have to come at the cost of accuracy. By combining neural rule selection with symbolic logic execution, a model can retain the expressivity of deep learning while remaining transparent.
- Allowing experts to intervene on the *model structure* itself (e.g., defining logic rules) significantly improves data efficiency, especially in low-data regimes.
