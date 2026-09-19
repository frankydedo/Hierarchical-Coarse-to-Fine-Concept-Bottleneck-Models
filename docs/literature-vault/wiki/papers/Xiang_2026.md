---
title: "HiProto: Hierarchical Prototype Learning for Interpretable Object Detection Under Low-quality Conditions"
authors: ["Jianlin Xiang", "Linhui Dai", "Xue Yang", "Chaolei Yang", "Yanshan Li"]
year: 2026
tags: [paper]
---

# HiProto: Hierarchical Prototype Learning for Interpretable Object Detection Under Low-quality Conditions

## Summary
The paper introduces [[HiProto]], a prototype-driven architecture for interpretable object detection under [[Low-quality Object Detection|low-quality conditions]] (e.g., haze, low light). Instead of using complex image enhancement pipelines, HiProto integrates [[Hierarchical Prototype Learning (HPL)|Hierarchical Prototype Learning]] into a Feature Pyramid Network. This aligns multi-scale features with class-centered semantics.

## Key Contributions
- **[[HiProto]]:** An end-to-end interpretable detection architecture.
- **[[Region-to-Prototype Contrastive Loss (RPC-Loss)]]:** Promotes semantic focus on target regions using pseudo label-guided contrastive learning.
- **[[Prototype Regularization Loss (PR-Loss)]]:** Enforces orthogonality (using SVD) to enhance prototype distinctiveness.
- **[[Scale-aware Pseudo Label Generation Strategy (SPLGS)]]:** Filters out objects exceeding the valid perception range of each feature layer to preserve robust low-level prototype representations.
