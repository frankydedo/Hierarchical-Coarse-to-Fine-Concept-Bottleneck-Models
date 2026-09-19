---
title: "Coarse-to-Fine Concept Bottleneck Models"
authors: ["Konstantinos P. Panousis", "Dino Ienco", "Diego Marcos"]
year: 2024
tags: [paper]
---

# Coarse-to-Fine Concept Bottleneck Models

## Summary
The paper introduces [[Coarse-to-Fine Concept Bottleneck Models (CF-CBMs)]], a framework designed to capture both high-level image concepts and low-level patch-specific concepts. To address the problem of concept over-abundance and top-down assumption errors in standard CBMs, this model utilizes a [[Concept Hierarchy]] where high-level concepts determine the relevance of low-level patch attributes via a [[Concept Discovery Block]]. 

## Key Contributions
- **[[Coarse-to-Fine Concept Bottleneck Models (CF-CBMs)]]:** Learns hierarchical concept representations from coarse (whole image) to fine (patch-specific).
- **[[Concept Discovery Block]]:** Uses data-driven Bernoulli random variables to explicitly infer concept activation and mask irrelevant attributes.
- Introduces a Jaccard index-based metric to quantitatively assess interpretability.
- Shows that localized, low-level concept discovery yields highly interpretable and accurate models.
