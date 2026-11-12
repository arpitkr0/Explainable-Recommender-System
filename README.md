# Explainable Hybrid Recommendation System

An explainable hybrid recommendation system with context and influence-aware
recommendations, built on the Amazon Reviews 2023 (Electronics) dataset.

## Project Structure

```
├── configs/config.yaml          # All tuneable parameters (Section 22)
├── data/
│   ├── raw/                     # Original JSONL files (not in version control)
│   ├── processed/               # Subsetted Parquet files from Section 6
│   └── splits/                  # Train/test splits
├── notebooks/                   # Phase-by-phase Jupyter notebooks
├── src/
│   ├── data/                    # Loading, cleaning, splitting, subsetting
│   ├── content/                 # Phase 1: TF-IDF features, user profiles
│   ├── ranking/                 # Phase 1: XGBoost ranking model
│   ├── explainability/          # Phase 1: SHAP + explanation engine
│   ├── collaborative/           # Phase 2: collaborative filtering
│   ├── context/                 # Phase 3: context-aware scoring
│   ├── influence/               # Phase 4: similarity-based influence
│   ├── evaluation/              # Precision/Recall/NDCG metrics
│   └── pipeline/                # Top-level recommend() API
├── models/                      # Serialised model artefacts
├── tests/                       # pytest test suite
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Subset the raw data (Section 6) — run once
python -m src.data.subset_data

# 2. Verify the subset
python -m pytest tests/ -v
```

## Phases

| Phase | Description |
|-------|-------------|
| 1     | Content-based filtering + SHAP explainability |
| 2     | + Collaborative filtering (hybrid score) |
| 3     | + Context-aware scoring |
| 4     | + Similarity-based influence engine |
