import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy import sparse

ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


@dataclass
class DatasetArtifacts:
    user_map: Dict[int, int]
    item_map: Dict[int, int]
    item_inverse_map: Dict[int, int]
    item_titles: Dict[int, str]
    train_matrix: sparse.csr_matrix
    test_pairs: List[Tuple[int, int]]


def download_movielens(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "ml-100k.zip"
    extracted_dir = dest_dir / "ml-100k"
    if extracted_dir.exists():
        return extracted_dir
    if not zip_path.exists():
        urlretrieve(ML_100K_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return extracted_dir


def _load_raw(data_root: Path):
    cols = ["user_id", "item_id", "rating", "timestamp"]
    ratings = pd.read_csv(data_root / "u.data", sep="\t", names=cols)
    items = pd.read_csv(
        data_root / "u.item",
        sep="|",
        encoding="latin-1",
        names=[
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ],
    )
    return ratings, items[["item_id", "title"]]


def _encode_ids(series: pd.Series) -> Dict[int, int]:
    unique_ids = series.unique()
    return {int(raw): idx for idx, raw in enumerate(unique_ids)}


def _leave_last_out(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    ratings = ratings.sort_values("timestamp")
    last = ratings.groupby("user_id").tail(1)
    train_mask = ~ratings.index.isin(last.index)
    train = ratings[train_mask]
    test_pairs = list(zip(last["user_idx"], last["item_idx"]))
    return train, test_pairs


def _build_matrix(
    ratings: pd.DataFrame, n_users: int, n_items: int, weight_strategy: str = "log"
) -> sparse.csr_matrix:
    if weight_strategy == "binary":
        weights = np.ones(len(ratings))
    elif weight_strategy == "log":
        weights = np.log1p(ratings["rating"].to_numpy())
    else:
        weights = ratings["rating"].to_numpy()
    return sparse.csr_matrix(
        (weights, (ratings["user_idx"], ratings["item_idx"])), shape=(n_users, n_items)
    )


def prepare_dataset(data_dir: Path) -> DatasetArtifacts:
    data_root = download_movielens(data_dir)
    ratings, items = _load_raw(data_root)

    user_map = _encode_ids(ratings["user_id"])
    item_map = _encode_ids(ratings["item_id"])
    ratings["user_idx"] = ratings["user_id"].map(user_map)
    ratings["item_idx"] = ratings["item_id"].map(item_map)

    train_df, test_pairs = _leave_last_out(ratings)

    n_users = len(user_map)
    n_items = len(item_map)
    train_matrix = _build_matrix(train_df, n_users, n_items, weight_strategy="log")

    items["item_idx"] = items["item_id"].map(item_map)
    item_titles = (
        items.dropna(subset=["item_idx"]).set_index("item_idx")["title"].to_dict()
    )

    artifacts = DatasetArtifacts(
        user_map=user_map,
        item_map=item_map,
        item_inverse_map={v: k for k, v in item_map.items()},
        item_titles=item_titles,
        train_matrix=train_matrix,
        test_pairs=test_pairs,
    )
    return artifacts


def save_mappings(artifacts: DatasetArtifacts, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "user_map.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.user_map, f)
    with open(out_dir / "item_map.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.item_map, f)
    with open(out_dir / "item_titles.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.item_titles, f, ensure_ascii=False)

