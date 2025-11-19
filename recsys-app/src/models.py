from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy import sparse


@dataclass
class TrainedModel:
    model: AlternatingLeastSquares
    user_map: Dict[int, int]
    item_map: Dict[int, int]
    item_titles: Dict[int, str]


def train_als(
    train_matrix: sparse.csr_matrix,
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 20,
    alpha: float = 40.0,
) -> AlternatingLeastSquares:
    # Build a confidence-weighted user-item matrix
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        calculate_training_loss=True,
    )
    confidence = train_matrix * alpha
    model.fit(confidence)
    return model


def recommend_for_users(
    model: AlternatingLeastSquares,
    train_matrix: sparse.csr_matrix,
    user_ids: List[int],
    n: int = 10,
    filter_seen: bool = True,
) -> Dict[int, List[int]]:
    recs = {}
    for u in user_ids:
        user_row = train_matrix[u]
        seen = user_row.indices if filter_seen else None
        # implicit expects user_items to have the same number of rows as user_ids; provide just this user's row
        items, _ = model.recommend(
            userid=u,
            user_items=user_row,
            N=n,
            filter_items=seen,
            recalculate_user=True,
        )
        recs[u] = list(items)
    return recs


def save_model_artifacts(
    model: AlternatingLeastSquares,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "als_model.pkl", "wb") as f:
        pickle.dump(model, f)


def load_model_artifacts(out_dir: Path) -> AlternatingLeastSquares:
    with open(out_dir / "als_model.pkl", "rb") as f:
        return pickle.load(f)
