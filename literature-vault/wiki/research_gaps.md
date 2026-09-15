---
tags: [synthesis, gaps]
---

# Research Gaps and Future Exploration Paths

Based on the project requirements outlined for **P1** and the comprehensive review of the `literature-vault` wiki, the following research gaps and exploration paths have been identified in the domain of hierarchical concept-based explainable models.

## Identified Limitations in Current Literature

### 1. The Persistence of Flat Concept Representations
Despite the documented advantages of hierarchical reasoning [[Hase_2019], [Uyterlinde_2026]], the majority of practically deployed Concept Bottleneck Models (CBMs) still rely on flat, independent sets of human-defined symbols. As highlighted in the P1 requirements, this simplification overlooks the natural *is-a* (e.g., *beak* is part of *head*) and *part-of* relationships that govern visual semantics. Even advanced models like [[Label-Free Concept Bottleneck Model (LF-CBM)]] often only operate at a single semantic tier.

### 2. Disconnect Between Concept Hierarchies and Label Hierarchies
Many existing models enforce structural relationships in the feature or concept space but fail to propagate this logic to the label space. For example, while [[Coarse-to-Fine Concept Bottleneck Models (CF-CBMs)]] [[Panousis_2023]] implement hierarchical patching, they ultimately predict a flat label. Models like [[HIL-CBM]] [[Xie_2026]] attempt to bridge this by predicting at multiple abstraction levels, but the literature still lacks unified frameworks where coarse predictions formally gate or logically dictate fine-grained outputs.

### 3. Limited Integration of Hierarchy-Aware Mistake Evaluation
Recent work emphasizes that not all classification errors are equally severe; confusing semantically similar subclasses is better than confusing distinct superclasses [[Bertinetto_2020]]. However, modern CBM frameworks rarely integrate hierarchy-aware evaluation natively. Most CBM literature evaluates interpretability metrics (like intervention accuracy) and standard predictive metrics independently, without assessing whether the *concepts themselves* cause the model to make "semantically better" mistakes.

## Proposed Paths for Exploration

To fulfill the goal of designing an explainable-by-design model where concepts are explicitly organized in a hierarchy, the following research paths are proposed:

### Path A: Ontology-Driven Multi-Level Concept Bottleneck Architectures
**Direction:** Extend an existing CBM architecture by strictly enforcing *is-a* relationships sourced from external knowledge bases.
- **Fills Gap:** 1. The Persistence of Flat Concept Representations
- **Approach:** Leverage existing ontologies (e.g., WordNet) or automatically construct a hierarchy using Large Language Models to explicitly map coarse concepts to fine concepts.
- **Related Work:** This builds upon the graph-based knowledge integration seen in [[Medical Knowledge Graphs]] [[Cai_2025]] and the structural alignment of [[Tree-path KL Divergence Loss]] [[Xie_2026]], applying them to general visual taxonomies. 

### Path B: Constrained Coarse-to-Fine Prediction Mechanics
**Direction:** Constrain the prediction process so that the activation of coarse concepts acts as a prerequisite for fine-grained concept evaluation.
- **Fills Gap:** 2. Disconnect Between Concept Hierarchies and Label Hierarchies
- **Approach:** Design an inference pipeline where high-level categories (e.g., *animal*, *vehicle*) are predicted first. The model then dynamically masks or activates subordinate concept detectors (e.g., *fur*, *wheels*) based on the coarse prediction. 
- **Related Work:** This path operationalizes the sequential reasoning of [[Progressive Concept Bottleneck Models (P-CBMs)]] [[Lin_2022]] and the rigorous constraint mechanisms of the [[Intervention Matrix]] [[Sun_2024], [Sun_2026]].

### Path C: Multi-Granularity Interventions and Explanations
**Direction:** Develop a framework capable of generating explanations and accepting human interventions at multiple levels of granularity. 
- **Fills Gap:** 3. Limited Integration of Hierarchy-Aware Mistake Evaluation (by integrating multi-level interpretability)
- **Approach:** Enable an interactive interface where a user can intervene at a coarse level (e.g., correcting the model from "Dog" to "Cat"). The model should subsequently use this intervention to automatically adjust the probabilities of subordinate concepts (e.g., deactivating "barks", activating "meows") and predict the correct fine-grained breed.
- **Related Work:** This closely aligns with the geometric propagation of [[Hierarchically Faithful Interventions]] seen in [[Hyperbolic Concept Bottleneck Models (HypCBM)]] [[Uyterlinde_2026]], but translates the concept into a Euclidean or probability-based framework suitable for standard CBMs.

## Evaluation Strategy Alignment
Whichever path is chosen, the evaluation phase must merge predictive accuracy with hierarchy-aware analysis. As mandated by P1, the chosen model should be evaluated using metrics derived from [[Hierarchical Cross-Entropy]] [[Bertinetto_2020]] to prove that explicit hierarchical concept structuring results in mistakes that are less severe and explanations that remain coherent across abstraction levels.
