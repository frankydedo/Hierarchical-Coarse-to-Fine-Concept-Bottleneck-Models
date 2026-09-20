---
tags: [concept]
---

# Test-Time Intervention

Test-Time Intervention (or Concept Intervention) is a core property of [[Concept Bottleneck Models]] where a human user or external oracle can interact with the model at inference time by inspecting and editing intermediate concept predictions $\hat{c}$ and immediately observing how the revised concepts propagate to the downstream prediction $\hat{y} = f(\hat{c})$.

In high-stakes domains (such as healthcare and medical diagnostics), test-time intervention enables effective human-AI collaboration: domain experts can correct mispredicted clinical attributes (e.g., bone spurs or joint space narrowing), enabling the model to recover from perceptual errors without needing retraining.

**Related Papers:**
- [[Koh_2020]]: Formalizes test-time intervention in independent, sequential, and joint CBMs, showing substantial task accuracy gains from correcting a small set of concepts.
- [[Hierarchically Faithful Interventions]]: Extends interventions to hierarchical concept spaces to ensure ancestor/descendant logical consistency.
- [[Intervention Matrix]]: Implements hard structural constraints on which concepts can influence specific classes during intervention.
