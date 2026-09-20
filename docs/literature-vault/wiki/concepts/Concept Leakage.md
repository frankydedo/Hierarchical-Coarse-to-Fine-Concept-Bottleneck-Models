---
tags: [concept]
---

# Concept Leakage

Concept Leakage (or Information Leakage) in Concept Bottleneck Models occurs when the model inadvertently encodes extra, unintended information from the input into the concept representations (especially when using soft, probabilistic concepts). This allows the task predictor to bypass the intended reasoning pathways, leading to high accuracy but unreliable and unfaithful explanations.

**Related Papers:**
- [[Koh_2020]]: Notes that joint training with weak concept regularization (low $\lambda$) allows models to alter concept representations to maximize task accuracy at the expense of concept alignment.
- [[Kalampalikis_2025]]: Mitigates leakage by enforcing structured reasoning graphs.
- [[Lin_2022]]: Reduces leakage in ultrasound assessment by discretizing explanations progressively via segmentations.
