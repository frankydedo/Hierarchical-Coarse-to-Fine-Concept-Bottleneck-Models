---
tags: [concept]
---

# Tree-path KL Divergence Loss

Tree-path KL Divergence Loss is a semantic consistency objective that enforces structural alignment between the model's predictions and a known hierarchical label structure. It penalizes predictions that contradict the class taxonomy (e.g., predicting a low-level category that doesn't match the predicted high-level category).

**Related Papers:**
- [[Xie_2026]]: Uses this loss to jointly train hierarchical classifiers in HIL-CBM.
