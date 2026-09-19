# Literature Review Implementation Plan

The goal of this phase is to conduct a systematic literature review for your XAI course project (Hierarchical Concept-Based Explainable-by-Design Models), adhering to the **PRISMA framework**, and delivering the final result as a structured **LLM Wiki** based on Andrej Karpathy's pattern.

## Proposed Approach

The plan merges the rigorous systematic review steps of PRISMA with the compounding knowledge base architecture of the LLM Wiki pattern. The final wiki will be tailored for **Obsidian**, utilizing `[[Wiki Links]]` for seamless navigation, and will serve as a persistent, compounding artifact.

### Phase 1: Protocol & Architecture Definition (Preparation)
- **PRISMA Protocol:** Define strict inclusion/exclusion criteria, search strings, and the specific research questions based on the requirements (Concept Bottleneck Models, coarse-to-fine discovery, hierarchical prototypes, hierarchy-aware evaluation). We will use the 5 references from `P1.md` as initial seeds.
- **LLM Wiki Architecture:** We will establish our knowledge base inside a `literature-vault/` directory with the following structure:
  - `raw/`: Immutable directory for downloaded PDFs and raw extracted text from ArXiv and OpenAlex.
  - `wiki/`: The persistent memory layer containing LLM-generated markdown files. It will be subdivided into:
    - `papers/`: Dedicated pages for each ingested paper.
    - `concepts/`: Dedicated pages for broader entities and methodologies.
  - `output/`: For ephemeral outputs on demand (e.g., intermediate scripts, temp analysis files). This directory must be empty at the end of the literature review phase.
  - `AGENTS.md`: The key configuration file that tells the LLM how the wiki is structured, what the conventions are, and what workflows to follow when ingesting sources, answering questions, or maintaining the wiki. This is what makes the LLM a disciplined wiki maintainer rather than a generic chatbot.
- **Indexing & Logging Files:** Initialize `index.md` (content-oriented catalog) and `log.md` (chronological append-only record of LLM operations).

### Phase 2: Identification & Screening (PRISMA Steps 1 & 2)
- **Identification:** I will run automated searches on **ArXiv** and **OpenAlex** using our defined search strings and download the metadata/abstracts.
- **Screening:** I will review titles and abstracts against our criteria, maintaining a PRISMA flow diagram log of accepted/rejected papers.

### Phase 3: Eligibility & Wiki Ingestion (PRISMA Steps 3 & 4)
For each paper passing the screening phase, we perform the **Ingest** operation:
- **Full-Text Retrieval:** Download the full text into the immutable `literature-vault/raw/` folder.
- **Ingest Operation:** I will read the full text and integrate it into the `wiki/`. The process includes:
  - Creating a dedicated **Paper Page** in `literature-vault/wiki/papers/` (e.g., `[[Bertinetto 2020]]`).
  - Extracting and creating/updating **Concept Pages** in `literature-vault/wiki/concepts/` (e.g., `[[Concept Bottleneck Models]]`, `[[Hierarchy-Aware Evaluation]]`).
  - Updating the `index.md` catalog with the new pages.
  - Appending a record to `log.md` (e.g., `## [2026-08-17] ingest | Bertinetto 2020`).

### Phase 4: Synthesis & Research Gaps (The Query Operation)
- With the wiki populated, we will use the **Query** operation to synthesize the findings.
- I will search the wiki to aggregate the limitations of current flat concept representations and coarse-to-fine modeling.
- The answers and synthesis will be filed back into the wiki as new concept or overview pages (e.g., `[[State of the Art]]` and `[[Identified Research Gaps]]`), ensuring the knowledge compounds rather than disappearing into our chat history.

### Phase 5: Health-Checks (The Lint Operation)
- Periodically, I will run a **Lint** operation over the wiki to:
  - Check for contradictions or stale claims between papers.
  - Identify orphan pages or concepts that need their own dedicated page.
  - Ensure all cross-references are valid.

## Verification Plan

### Automated/Tool Verification
- Verify that there are no broken links within the Markdown wiki during Lint passes.
- Ensure the PRISMA flow numbers match (Records identified -> screened -> assessed -> included).
- Check that `log.md` and `index.md` are correctly formatted.
- Ensure the `literature-vault/output/` directory is empty at the end of the phase.

### Manual Verification
- You will be able to open the `literature-vault/` project directly in Obsidian to browse the graph view, follow the evolution of the research in `log.md`, and verify that the compounding knowledge meets the course requirements.
