---
tags: [concept]
---

# Medical Knowledge Graphs

Medical Knowledge Graphs (KGs) are structured representations of medical knowledge, where entities (such as diseases, medications, and procedures) are connected by semantic relations (e.g., "treats", "causes", "co-occurs with"). 

They are used to enrich representation learning by explicitly modeling the complex relationships between medical codes, which might otherwise be missed by purely data-driven approaches on [[Electronic Health Records (EHR)]]. KGs can be sourced from established databases (like UMLS) or constructed automatically using Large Language Models (LLMs).

**Related Papers:**
- [[Cai_2025]]: Leverages LLMs (like Llama 3 and GPT-4) to automatically construct and refine a medical knowledge graph from EHR data, which is then used as a foundation for hierarchical [[Prototype Learning]].
