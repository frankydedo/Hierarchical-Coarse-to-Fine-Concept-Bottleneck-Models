---
title: "Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks"
authors: ["Luca Bertinetto", "Romain Mueller", "Konstantinos Tertikas", "Sina Samangooei", "Nicholas A. Lord"]
year: 2020
tags: [paper]
---

# Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks

## Summary
The paper argues that standard deep neural networks often make catastrophic errors because they treat all mistakes equally under measures like top-1 error. To mitigate mistake severity, the authors revive the idea of incorporating [[Class Hierarchies]] (like WordNet or iNaturalist) into the loss function. They propose two simple, effective variants of cross-entropy loss: [[Hierarchical Cross-Entropy]] (HXE) and [[Soft Labels]]. These methods significantly improve hierarchical error metrics (e.g., hierarchical distance of mistakes) on tieredImageNet and iNaturalist'19 while maintaining competitive top-k accuracy. 

## Key Contributions
- Review of literature on hierarchical classification methods.
- Proposition of two one-parameter drop-in generalisations of cross-entropy:
  - **[[Hierarchical Cross-Entropy]] (HXE):** Reweights the conditional probabilities of a class's lineage.
  - **[[Soft Labels]]:** A label-embedding approach that uses an exponentially decaying probability distribution based on the lowest common ancestor (LCA) distance to the ground truth.
- Extensive experiments showing a tradeoff between top-1 error and hierarchical mistake severity, demonstrating their methods subsume and outperform prior art like YOLO-v2, DeViSE, and Barz & Denzler.
- Evidence that the structure of the hierarchy matters—random hierarchies lead to worse performance.

## Insights
- Conventional top-k measures mask the actual severity of mistakes. 
- While hierarchical architectures ("divide and conquer") exist, changing the loss function alone (HXE or Soft Labels) provides a simpler, competitive way to inject prior class relationships.
- There is an inherent tension between standard top-1 accuracy and hierarchical mistake severity (similar to the tradeoff between robustness and accuracy in adversarial settings).
