# MovieLens Recommender (Implicit ALS)

A focused, production-style recommender built on MovieLens 100K with leave-last-out evaluation, implicit ALS training, and reusable artifacts (model + mappings). Designed to be a strong portfolio project.

## Why this is solid
- Clear data pathway: download, preprocess, leave-last-out split, train, evaluate, save artifacts.
- Industrial pattern: implicit feedback with confidence scaling, ALS for retrieval-quality recs, top-K metrics (recall@K, NDCG@K).
- Reusable assets: serialized model, user/item maps, titles for UX, metrics JSON.
- Extensible: swap in BPR/LightFM or add re-rankers later; logs directory ready for experiment tracking.

## Structure
- `train.py` — end-to-end pipeline (download → prep → train → eval → save).
- `recommend.py` — load artifacts and print top-N recs for a given user.
- `src/data_prep.py` — download + load MovieLens, encode ids, build sparse matrices.
- `src/models.py` — implicit ALS training + recommend wrapper + save/load helpers.
- `src/eval.py` — recall@K, NDCG@K.
- `requirements.txt` — pinned deps (pandas, numpy, scipy, implicit, click).

## Quickstart
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # on Windows PowerShell
pip install -r recsys-app/requirements.txt
# Run end-to-end training (downloads data to recsys-app/data, saves to recsys-app/artifacts)
python recsys-app/train.py
# Get recommendations for a MovieLens user id (e.g., 42)
python recsys-app/recommend.py --user-id 42
```

Outputs land in `recsys-app/artifacts/`:
- `als_model.pkl` — implicit ALS model (pickled).
- `user_map.json`, `item_map.json`, `item_titles.json` — id mappings + titles.
- `metrics.json` — recall@10, ndcg@10.

## Implementation notes
- Split: leave-last-out per user (timestamp order) to mimic “next-item” prediction.
- Feedback: log-scaled ratings as implicit confidence, scaled by `alpha` (default 40).
- Filtering: recommendations by default ignore already-seen items (during eval).
- Metrics: recall@10 and ndcg@10 computed on held-out next item.

## Extending
- Add re-ranking (e.g., LightGBM ranker) over ALS candidates.
- Add item/content cold-start via genre embeddings or text (BERT) similarities.
- Serve via FastAPI with `/recommend?user_id=...` returning titles and ids.
- Add experiment tracking (Weights & Biases or MLflow) hooked into `train.py`.
