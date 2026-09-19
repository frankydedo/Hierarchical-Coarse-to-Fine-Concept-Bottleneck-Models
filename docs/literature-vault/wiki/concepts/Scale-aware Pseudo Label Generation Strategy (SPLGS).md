---
tags: [concept]
---

# Scale-aware Pseudo Label Generation Strategy (SPLGS)

Scale-aware Pseudo Label Generation Strategy (SPLGS) is an approach used in multi-scale object detection (like HiProto) to generate more robust supervision signals. It filters out objects that exceed the valid perception range of a particular feature layer, preventing low-level features (which have small receptive fields) from being incorrectly supervised by large objects. This reduces label noise and preserves the robustness of prototype representations at different scales.

**Related Papers:**
- [[Xiang_2026]]: Introduces SPLGS to improve prototype robustness in HiProto.
