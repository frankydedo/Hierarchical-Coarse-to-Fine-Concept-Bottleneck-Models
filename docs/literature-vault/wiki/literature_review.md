---
tags: [synthesis, review]
---

# Literature Review: The Evolution of Concept-Based Explainable Models

## Introduction to Concept-Based XAI
The demand for interpretable deep learning models has driven the development of [[Concept-based Explainable Artificial Intelligence (C-XAI)]]. Among these, Concept Bottleneck Models (CBMs) have emerged as a canonical approach by forcing models to predict human-understandable concepts prior to the final label prediction, thereby providing a transparent reasoning process [[Poeta_2025]]. Early CBMs, however, relied on manually annotated, flat sets of concepts. This reliance presented two major limitations: the prohibitive cost of dense annotations and the failure to capture the natural semantic structures and hierarchies that govern human reasoning. 

## The Shift to Unsupervised and Label-Free Discovery
To overcome the annotation bottleneck, researchers explored unsupervised and label-free paradigms. [[Wang_2023]] introduced the [[Bottleneck Concept Learner (BotCL)]], utilizing [[Slot Attention]] and [[Self-Supervision over Concepts]] (via contrastive learning) to discover visual concepts without explicit labels. Similarly, the [[Label-Free Concept Bottleneck Model (LF-CBM)]] eliminated manual annotation costs by leveraging Large Language Models to generate concepts and aligning them with image features using vision-language models like CLIP [[Xie_2026]]. 

## The Transition Towards Hierarchical Reasoning
Human cognition naturally organizes concepts from abstract (superordinate) to specific (subordinate) levels. Recognizing this, the literature shifted towards incorporating structure. Early work by [[Hase_2019]] proposed learning [[Hierarchical Prototypes]] for image recognition, allowing models to detect novel classes and fail more gracefully. Similarly, [[Zhang_2019]] utilized [[Hierarchical Prototype Learning (HPL)]] to bridge visual and [[Semantic Prototypes]] via [[Super-prototypes]] in transductive [[Zero-Shot Learning (ZSL)]]. 

As CBM architectures matured, models began explicitly structuring the inference process. [[Lin_2022]] developed [[Progressive Concept Bottleneck Models (P-CBMs)]] for fetal ultrasound assessment, dividing inference into a coarse [[Segmentation Bottleneck]] followed by a finer [[Property Bottleneck]]. [[Panousis_2023]] tackled the issue of concept over-abundance by introducing [[Coarse-to-Fine Concept Bottleneck Models (CF-CBMs)]], which utilize [[Concept Discovery Block]]s to hierarchically mask irrelevant patch attributes. 

## The State-of-the-Art: Hierarchical CBMs
Recent years have seen an explosion in models striving for deep structural integration to combat [[Concept Leakage]]—a phenomenon where unintended predictive signals bypass the concepts.

### Hierarchical Prototypes and Concept Pooling
[[Pittino_2023]] introduced [[Hierarchical Concept Bottleneck Models]] specifically designed for fine classification and [[Object Tracking]], utilizing a spatial hierarchy of patches. To directly combat information leakage, [[Sun_2024]] and [[Sun_2025]] proposed [[SUPCBM]], separating concept and label prediction via a hard binary [[Intervention Matrix]] and [[Concept Pooling]]. This work was extended in [[Hi-CBM]] by [[Sun_2026]], which leverages [[Concept-Bottleneck Pooling]] over a two-level concept hierarchy (perceptual and descriptive) to aggressively filter out spurious concepts. Similar robust hierarchical logic has been successfully deployed under degraded visibility conditions with the [[HiProto]] architecture [[Xiang_2026]].

### Advanced Semantic and Geometric Structures
[[Xie_2026]] proposed [[HIL-CBM]], which extends the label-free paradigm to hierarchies using [[Gradient-based Saliency Alignment Loss]] (for visual consistency) and [[Tree-path KL Divergence Loss]] (for semantic consistency), proving that multi-level explanations can be learned without relational concept labels. Concurrently, [[Uyterlinde_2026]] radically reimagined the concept space using [[Hyperbolic Concept Bottleneck Models (HypCBM)]]. By replacing flat Euclidean space with hyperbolic geometry, HypCBM utilizes [[Hyperbolic Entailment Activation]] to naturally enforce hierarchical consistency and enable [[Hierarchically Faithful Interventions]]. 

Domain-specific applications have also driven hierarchical innovation. [[Cai_2025]] introduced ProtoEHR, combining hierarchical [[Prototype Learning]] with [[Medical Knowledge Graphs]] to explain patient trajectories across [[Electronic Health Records (EHR)]].

## Hierarchy-Aware Evaluation
Alongside architectural advancements, the evaluation of these models has evolved. Traditional metrics penalize all mistakes equally, but confusing a "border collie" with a "golden retriever" is less severe than confusing it with a "cat". [[Bertinetto_2020]] spearheaded this paradigm shift by leveraging [[Class Hierarchies]] through [[Hierarchical Cross-Entropy]] and structured [[Soft Labels]], ensuring that models are penalized based on the semantic severity of their errors. 

## Conclusion
The C-XAI field has transitioned from flat, manually supervised bottlenecks to highly structured, hierarchical, and label-free architectures. Current state-of-the-art models not only provide richer, multi-granularity explanations but actively employ hierarchical reasoning to combat concept leakage, improve robustness, and fail gracefully.
