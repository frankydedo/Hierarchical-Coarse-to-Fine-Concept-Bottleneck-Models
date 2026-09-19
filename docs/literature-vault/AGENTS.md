# Literature Vault LLM Schema

This directory is an **LLM Wiki** adhering to Andrej Karpathy's pattern. As an LLM Agent, your role is to act as a disciplined wiki maintainer.

## Directory Structure
- `raw/`: Immutable directory containing original PDF files and raw extracted text. You may read from here but NEVER modify the contents.
- `wiki/`: The persistent memory layer containing Markdown files. You own this layer.
  - `wiki/papers/`: Dedicated pages for each ingested paper.
  - `wiki/concepts/`: Dedicated pages for broader entities and methodologies (e.g., Concept Bottleneck Models).
- `output/`: Ephemeral outputs on demand (e.g., intermediate scripts, temp analysis files). Must be kept clean and emptied after tasks.

## Key Files
- `index.md`: A content-oriented catalog of all pages in the wiki, organized by category (Papers, Concepts). Update this on every ingest.
- `log.md`: A chronological, append-only record of operations (ingests, queries, linting passes). E.g., `## [YYYY-MM-DD] ingest | Bertinetto 2020`.
- `protocol.md`: The PRISMA protocol defining the literature review's scope, inclusion criteria, and search strings.

## Conventions & Rules
1. **Obsidian Native:** Use `[[Wiki Links]]` to connect entities, concepts, and papers. E.g., `[[Concept Bottleneck Models]]`. 
2. **Dataview Ready:** Include YAML frontmatter on all wiki pages. 
   - For papers: `title`, `authors`, `year`, `tags: [paper]`.
   - For concepts: `tags: [concept]`.
3. **Compound Knowledge:** When creating a paper page, do not just summarize it flatly. Extract its concepts. If a concept doesn't have a page, create one. If it does, update the concept page with the new insights from the paper.

## Workflows
- **Ingest:** Command: `!ingest {raw_source}`. Read a raw source -> Write a summary page in `wiki/papers/` -> Update/create related `wiki/concepts/` pages -> Update `index.md` -> Append to `log.md` -> Rename {raw_source} to {raw_source}_ingested.pdf (do it immediately).
- **Query:** Command: `!query {query}`. When asked to synthesize (e.g., "What are the research gaps?"), search the wiki, aggregate the information, and save the robust answer as a new concept/overview page in the wiki. Do NOT leave valuable synthesis only in the chat history.
- **Lint:** Command: `!lint`. Periodically health-check the wiki. Look for contradictions, stale claims, orphan pages, or missing cross-references.
