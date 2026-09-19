---
title: "Hierarchical concept Bottleneck models for vision and their application to explainable fine classification and tracking"
authors: ["Federico Pittino", "Vesna Dimitrievska", "Rudolf Heer"]
year: 2023
tags: [paper]
---

# Hierarchical concept Bottleneck models for vision and their application to explainable fine classification and tracking

## Summary
This paper applies [[Hierarchical Concept Bottleneck Models]] to the practical use-case of fine-grained image classification and [[Object Tracking]] for objects inside a smart fridge. The authors replace the final layer of a Mask R-CNN with a two-stage concept bottleneck that predicts hierarchical concepts (e.g., 3D shape, Visual category, Logical group) before the final classification. This approach gracefully handles ambiguity and provides understandable tracking by comparing concept features between consecutive frames.

## Key Contributions
- Adapts Concept Bottleneck Models for [[Fine Classification]] and [[Object Tracking]].
- Proposes a novel method for handling multi-label classification to avoid incompatible class combinations.
- Uses Mask R-CNN to extract bounding boxes and contours, feeding regions into the concept model.
- Demonstrates competitive accuracy but significantly higher explainability compared to standard Mask R-CNN.
