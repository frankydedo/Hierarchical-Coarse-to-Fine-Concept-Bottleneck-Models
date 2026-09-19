---
tags: [concept]
---

# Label-Aware Concept Annotation

Label-Aware Concept Annotation is a technique used during the training of Concept Bottleneck Models to supervise the intermediate concept layer. Instead of treating all possible concepts independently, a training instance is only annotated with the concepts that are actively involved in predicting its specific ground-truth label. This approach binds concept prediction and label prediction more tightly, significantly mitigating unintended [[Concept Leakage]].

**Related Papers:**
- [[Sun_2024]]: Proposes label-aware concept annotation to eliminate information leakage in hard CBMs.
