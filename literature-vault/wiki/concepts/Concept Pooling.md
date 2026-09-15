---
tags: [concept]
---

# Concept Pooling

Concept Pooling is a selective strategy used in the concept annotation phase of Concept Bottleneck Models. Analogous to max pooling in standard computer vision architectures, concept pooling identifies and selects only the most prominent or descriptive concepts from a larger generated pool. By trimming away irrelevant or redundant concepts, it minimizes the risk of the model exploiting unintended signals (hard [[Concept Leakage]]).

**Related Papers:**
- [[Sun_2024]]: Employs concept pooling to filter LLM-generated descriptive concepts in SUPCBM.
- [[Lin_2022]]: Implements concept pooling within Progressive CBMs.

**See Also:**
- [[Concept-Bottleneck Pooling]]
