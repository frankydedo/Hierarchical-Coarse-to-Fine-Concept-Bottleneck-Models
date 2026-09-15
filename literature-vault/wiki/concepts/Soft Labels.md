---
tags: [concept]
---

# Soft Labels

Soft labels refer to training models with a target distribution that is not one-hot. Instead, probability mass is distributed across multiple classes.

When informed by [[Class Hierarchies]], soft labels can encode the semantic distance between the true class and other classes. For instance, the probability assigned to a non-target class decays exponentially based on its Lowest Common Ancestor (LCA) distance to the ground truth class. This acts as a regularizer and encourages the model to group semantically similar concepts together in its representation.

**Related Papers:**
- [[Bertinetto_2020]]: Uses soft labels derived from class hierarchy distances as an effective alternative to standard cross-entropy for reducing catastrophic classification mistakes.
