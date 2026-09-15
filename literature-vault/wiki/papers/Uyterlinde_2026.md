---
title: "Hyperbolic Concept Bottleneck Models"
authors: ["Daniel Uyterlinde", "Swasti S. Mishra", "Pascal Mettes"]
year: 2026
tags: [paper]
---

# Hyperbolic Concept Bottleneck Models

## Summary
The paper identifies that standard CBMs embed concepts in flat Euclidean space, failing to capture the semantic hierarchies inherent in real-world concepts. It proposes [[Hyperbolic Concept Bottleneck Models (HypCBM)]], which reformulates concept activation as an asymmetric geometric containment problem in hyperbolic space. This allows for [[Hierarchically Faithful Interventions]], meaning user corrections propagate coherently through a concept tree.

## Key Contributions
- **[[Hyperbolic Concept Bottleneck Models (HypCBM)]]:** Grounds the bottleneck in hyperbolic geometry.
- **[[Hyperbolic Entailment Activation]]:** Maps image representations to sparse concept activations based on entailment cone containment.
- **[[Hierarchically Faithful Interventions]]:** An adaptive scaling law that propagates interventions down the concept hierarchy (e.g., intervening on "Animal" affects "Dog").
- Achieves strong performance in sparse regimes, improving robustness and hierarchical consistency.
