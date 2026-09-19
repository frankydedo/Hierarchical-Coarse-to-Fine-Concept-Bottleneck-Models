---
tags: [concept]
---

# Hierarchical Cross-Entropy (HXE)

Hierarchical Cross-Entropy (HXE) is a loss function that modifies standard cross-entropy by factoring a class's categorical distribution into a chain of conditional probabilities defined by its lineage in a hierarchy tree.

It reweights the cross-entropies of the conditional probabilities (edges in the tree) using a parameter $\alpha$, discounting the information further down the hierarchy. This heavily penalizes misclassifications that are far away from the ground truth class in the hierarchy.

**Related Papers:**
- [[Bertinetto_2020]]: Introduces HXE as a simple drop-in replacement to mitigate mistake severity.
