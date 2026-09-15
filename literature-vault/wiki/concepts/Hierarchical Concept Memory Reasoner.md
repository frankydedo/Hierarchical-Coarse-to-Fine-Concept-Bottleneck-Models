---
tags: [concept]
---

# Hierarchical Concept Memory Reasoner

The Hierarchical Concept Memory Reasoner (H-CMR) is a type of Concept-Based Model (CBM) designed to provide interpretability for both the final task prediction and the intermediate concept predictions. 

Unlike standard CBMs that predict concepts using black-box neural networks and assume concepts are conditionally independent, H-CMR learns a Directed Acyclic Graph (DAG) over the concepts. It maintains a "rule memory" containing symbolic logic rules that define how concepts depend on one another. 

During inference, a neural attention mechanism selects the appropriate rules based on parent concepts and a latent embedding, and then symbolically executes these rules. This makes the reasoning process explicitly transparent and allows human interventions on a parent concept to logically propagate to its dependent children.

**Related Papers:**
- [[Debot_2025]]: Introduces H-CMR, proving it is a universal classifier while maintaining high intervenability and interpretability.
