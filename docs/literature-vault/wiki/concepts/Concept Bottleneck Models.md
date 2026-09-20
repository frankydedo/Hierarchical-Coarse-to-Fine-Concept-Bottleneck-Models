---
tags: [concept]
---

# Concept Bottleneck Models

Concept Bottleneck Models (CBMs) are explainable-by-design neural network architectures that structure prediction into two sequential stages: first mapping raw input data $x$ to an intermediate bottleneck layer of predefined, human-understandable concepts $\hat{c} = g(x)$, and subsequently predicting the final target label $\hat{y} = f(\hat{c})$ solely from these predicted concepts ($x \xrightarrow{g} c \xrightarrow{f} y$).

By construction, CBMs enforce that all reasoning passes through the concept layer, providing inherent interpretability and enabling [[Test-Time Intervention]] where users can inspect or override predicted concepts to correct downstream model decisions.

CBMs can be trained via three primary formulations:
- **Independent Bottleneck:** Concept predictor $g$ and task predictor $f$ are trained separately (with $f$ trained directly on ground-truth concept annotations $c$, though evaluated on $\hat{g}(x)$ at test time).
- **Sequential Bottleneck:** Concept predictor $g$ is trained first on $(x, c)$, and $f$ is trained on the predicted concepts $\hat{g}(x)$.
- **Joint Bottleneck:** $g$ and $f$ are trained simultaneously with a combined loss function balancing task accuracy and concept fidelity via a weighting hyperparameter $\lambda$.

**Related Papers:**
- [[Koh_2020]]: Seminal paper proposing and benchmarking CBMs across medical imaging and fine-grained classification, demonstrating competitive accuracy and test-time intervenability.
- [[Poeta_2025]]: Comprehensive survey classifying CBMs within the Concept-based Explainable AI (C-XAI) taxonomy.
- [[Hierarchical Concept Bottleneck Models]]: Extends flat CBMs into taxonomical and multi-level semantic structures.
- [[Label-Free Concept Bottleneck Model (LF-CBM)]]: Mitigates annotation costs by using LLMs and vision-language models.
