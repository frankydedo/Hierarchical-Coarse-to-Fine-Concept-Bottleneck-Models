---
tags: [concept]
---

# Class Hierarchies

Class hierarchies (or taxonomies) explicitly define the relationships between different categories or concepts, often structured as a tree or directed acyclic graph (DAG). 

In computer vision and machine learning, incorporating class hierarchies into training helps models understand that some mistakes are "better" (less severe) than others. For example, mistaking a dog for a cat is semantically less catastrophic than mistaking a dog for a car.

Prominent visual hierarchies include the WordNet subset used for ImageNet and biological taxonomies like iNaturalist.

**Related Papers:**
- [[Bertinetto_2020]]: Demonstrates how to leverage class hierarchies to reduce the severity of mistakes using [[Hierarchical Cross-Entropy]] and [[Soft Labels]].
