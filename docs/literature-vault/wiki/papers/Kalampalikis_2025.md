---
title: "Towards Reasonable Concept Bottleneck Models"
authors: ["Nektarios Kalampalikis", "Kavya Gupta", "Georgi Vitanov", "Isabel Valera"]
year: 2026
tags: [paper]
---

# Towards Reasonable Concept Bottleneck Models

## Summary
The paper introduces Concept REAsoning Models (CREAMs), a flexible framework that explicitly encodes expert prior knowledge about concept-concept (C-C) and concept-task (C->Y) relationships. Unlike standard CBMs that assume conditional independence among concepts, CREAM utilizes a reasoning graph (often a DAG or PDAG) implemented via Structured Neural Networks (StrNNs) to enforce these dependencies. It also features a regularized side-channel to mitigate concept incompleteness while addressing [[Concept Leakage]].

## Key Contributions
- **[[Concept REAsoning Models (CREAMs)]]:** A modular framework encoding both C-C and C->Y relations.
- **[[Structured Model Reasoning]]:** Uses StrNNs to enforce logical and hierarchical constraints (like mutual exclusivity).
- Mitigation of [[Concept Leakage]] by restricting information flow through explicit reasoning graphs.
- Introduction of Concept Channel Importance (CCI), a metric to quantify interpretability when a side-channel is present.
