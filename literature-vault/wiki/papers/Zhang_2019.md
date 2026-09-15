---
title: "Hierarchical Prototype Learning for Zero-Shot Recognition"
authors: ["Xingxing Zhang", "Shupeng Gui", "Zhenfeng Zhu", "Yao Zhao", "Ji Liu"]
year: 2019
tags: [paper]
---

# Hierarchical Prototype Learning for Zero-Shot Recognition

## Summary
The paper addresses [[Zero-Shot Learning (ZSL)]] under a [[Transductive Zero-Shot Learning|transductive]] setting using [[Hierarchical Prototype Learning (HPL)]]. To bridge the domain gap between seen and unseen classes, the authors learn [[Visual Prototypes]] and [[Semantic Prototypes]]. They further introduce interpretable [[Super-prototypes]] that connect both visual and semantic spaces, maximizing their structural consistency to improve zero-shot recognition.

## Key Contributions
- **[[Hierarchical Prototype Learning (HPL)]]:** A transductive ZSL model enforcing discriminability on both seen and unseen domains.
- Learns [[Super-prototypes]] shared between seen and unseen classes to bridge domain gaps.
- Maximizes the structural consistency of visual and semantic prototypes to mitigate the loss of information.
- Provides an efficient iterative algorithm to solve the proposed formulation.
