---
tags: [concept]
---

# Gradient-based Saliency Alignment Loss

Gradient-based Saliency Alignment Loss is a visual consistency objective used in hierarchical CBMs to encourage different abstraction layers to focus on similar spatial regions. By computing Grad-CAM saliency maps for both concept layers and minimizing the mean squared error (MSE) between them, the loss promotes spatial alignment without needing explicit relational labels between concepts.

**Related Papers:**
- [[Xie_2026]]: Introduces this loss to train the dual-level concept space in HIL-CBM.
