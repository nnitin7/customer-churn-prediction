import json
from pathlib import Path
from typing import List

import click
import numpy as np
from scipy import sparse

from src.data_prep import DatasetArtifacts
from src.models import load_model_artifacts


def load_mappings(artifacts_dir: Path) -> DatasetArtifacts:
    with open(artifacts_dir / "user_map.json", "r", encoding="utf-8") as f:
        user_map = {int(k): int(v) for k, v in json.load(f).items()}
    with open(artifacts_dir / "item_map.json", "r", encoding="utf-8") as f:
        item_map = {int(k): int(v) for k, v in json.load(f).items()}
    with open(artifacts_dir / "item_titles.json", "r", encoding="utf-8") as f:
        item_titles = {int(k): v for k, v in json.load(f).items()}
    # Dummy sparse matrix with correct shape to use in recommend()
    train_matrix = sparse.csr_matrix((len(user_map), len(item_map)))
    return DatasetArtifacts(
        user_map=user_map,
        item_map=item_map,
        item_inverse_map={v: k for k, v in item_map.items()},
        item_titles=item_titles,
        train_matrix=train_matrix,
        test_pairs=[],
    )


@click.command()
@click.option("--artifacts-dir", default="recsys-app/artifacts", show_default=True, help="Path to saved artifacts.")
@click.option("--user-id", required=True, type=int, help="Original MovieLens user id (as in ratings file).")
@click.option("--top-k", default=10, show_default=True, help="Number of recommendations.")
def main(artifacts_dir: str, user_id: int, top_k: int):
    artifacts_path = Path(artifacts_dir)
    model = load_model_artifacts(artifacts_path)
    artifacts = load_mappings(artifacts_path)

    if user_id not in artifacts.user_map:
        raise SystemExit(f"Unknown user_id {user_id}.")
    user_idx = artifacts.user_map[user_id]

    # implicit.recommend requires user_items rows == number of userids passed; provide a single-row placeholder
    empty_user_row = sparse.csr_matrix((1, artifacts.train_matrix.shape[1]))
    items, scores = model.recommend(
        userid=user_idx,
        user_items=empty_user_row,
        N=top_k,
        filter_items=None,
        recalculate_user=False,
    )
    print(f"Top-{top_k} recommendations for user {user_id}:")
    for rank, (item_idx, score) in enumerate(zip(items, scores), start=1):
        title = artifacts.item_titles.get(item_idx, f"item_{item_idx}")
        print(f"{rank:2d}. {title} (score={score:.3f})")


if __name__ == "__main__":
    main()
