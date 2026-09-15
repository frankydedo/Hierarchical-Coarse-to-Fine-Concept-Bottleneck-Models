#!/bin/bash
cd /Users/francescodedominicis/.gemini/config/plugins/science/skills/literature_search_arxiv/
OUT_DIR="/Users/francescodedominicis/Desktop/XAI_P1/literature-vault/raw"
mkdir -p "$OUT_DIR"

TITLES=(
  "Making better mistakes Leveraging class hierarchies with deep networks"
  "Interpretable image recognition with hierarchical prototypes"
  "Hierarchical concept bottleneck models for vision and their application"
  "Concept-based Explainable Artificial Intelligence A Survey"
  "Eliminating Information Leakage in Hard Concept Bottleneck Models"
  "Hyperbolic Concept Bottleneck Models"
  "Towards Reasonable Concept Bottleneck Models"
  "Explainable fetal ultrasound quality assessment with progressive concept bottleneck models"
  "Coarse-to-Fine Concept Bottleneck Models"
  "Hierarchical Prototype Learning for Zero-Shot Recognition"
  "ProtoEHR Hierarchical Prototype Learning for EHR-based Healthcare Predictions"
  "HiProto Hierarchical Prototype Learning for Interpretable Object Detection Under Low-quality Conditions"
)

for title in "${TITLES[@]}"; do
  echo "Searching for: $title"
  uv run scripts/search_arxiv.py --query "ti:\"$title\"" --max_results 1 > /tmp/arxiv_res.json 2>/dev/null
  ID=$(jq -r '.papers[0].id // empty' /tmp/arxiv_res.json)
  if [ -z "$ID" ]; then
    uv run scripts/search_arxiv.py --query "all:\"$title\"" --max_results 1 > /tmp/arxiv_res.json 2>/dev/null
    ID=$(jq -r '.papers[0].id // empty' /tmp/arxiv_res.json)
  fi
  if [ -n "$ID" ]; then
    echo "Found ID: $ID. Downloading..."
    SAFE_TITLE=$(echo "$title" | tr -c "a-zA-Z0-9" "_")
    uv run scripts/download_paper.py --id "$ID" --format pdf --output "$OUT_DIR/${SAFE_TITLE}.pdf" 2>/dev/null
  else
    echo "Not found on ArXiv: $title"
  fi
  sleep 4
done
