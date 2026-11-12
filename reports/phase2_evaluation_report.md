# Section 20: Phase 1 vs. Phase 2 Evaluation Report
**Dataset**: Amazon Reviews 2023 (`Electronics` subset, 66,320 items)  
**Split Protocol**: Chronological Leave-One-Out (Section 8, Zero Leakage Verified)

## Recommendation Quality Comparison

| Model | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 | Users |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Popularity Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 500 |
| Phase 1 (Content + XGBoost + SHAP) | 0.0036 | 0.0180 | 0.0123 | 0.0022 | 0.0220 | 0.0137 | 500 |
| Phase 2 Hybrid (alpha=0.7) | 0.0052 | 0.0260 | 0.0233 | 0.0028 | 0.0280 | 0.0240 | 500 |
| Phase 2 Hybrid (alpha=0.5) | 0.0048 | 0.0240 | 0.0221 | 0.0026 | 0.0260 | 0.0228 | 500 |
| Phase 2 Hybrid (alpha=0.3) | 0.0040 | 0.0200 | 0.0174 | 0.0022 | 0.0220 | 0.0181 | 500 |
