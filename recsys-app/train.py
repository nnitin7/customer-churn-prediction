import json
from pathlib import Path
from typing import Dict

import click

from src.data_prep import prepare_dataset, save_mappings, DatasetArtifacts
from src.eval import recall_at_k, ndcg_at_k
from src.models import train_als, recommend_for_users, save_model_artifacts


def _evaluate(model, artifacts: DatasetArtifacts, k: int = 10) -> Dict[str, float]:
    # Build recommendations for all users in test set
    user_ids = list({u for u, _ in artifacts.test_pairs})
    recs = recommend_for_users(model, artifacts.train_matrix, user_ids, n=k + 5)
    return {
        "recall@k": recall_at_k(recs, artifacts.test_pairs, k=k),
        "ndcg@k": ndcg_at_k(recs, artifacts.test_pairs, k=k),
    }


@click.command()
@click.option("--data-dir", default="recsys-app/data", show_default=True, help="Where to download/cache MovieLens.")
@click.option("--artifacts-dir", default="recsys-app/artifacts", show_default=True, help="Where to store model artifacts.")
@click.option("--factors", default=64, show_default=True, help="ALS latent factors.")
@click.option("--iterations", default=20, show_default=True, help="ALS iterations.")
@click.option("--alpha", default=40.0, show_default=True, help="Confidence scaling for implicit ALS.")
def main(data_dir: str, artifacts_dir: str, factors: int, iterations: int, alpha: float):
    data_path = Path(data_dir)
    artifacts_path = Path(artifacts_dir)

    print("Preparing dataset...")
    artifacts = prepare_dataset(data_path)
    print(f"Users: {len(artifacts.user_map)}, Items: {len(artifacts.item_map)}, Train interactions: {artifacts.train_matrix.nnz}")

    print("Training ALS model...")
    model = train_als(
        artifacts.train_matrix,
        factors=factors,
        iterations=iterations,
        alpha=alpha,
    )

    print("Evaluating...")
    metrics = _evaluate(model, artifacts, k=10)
    print(f"recall@10={metrics['recall@k']:.4f}, ndcg@10={metrics['ndcg@k']:.4f}")

    print(f"Saving artifacts to {artifacts_path}...")
    save_model_artifacts(model, artifacts_path)
    save_mappings(artifacts, artifacts_path)
    with open(artifacts_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
