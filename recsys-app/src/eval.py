import numpy as np
from typing import List, Tuple, Dict


def recall_at_k(recs: Dict[int, List[int]], heldout: List[Tuple[int, int]], k: int = 10) -> float:
    hits = 0
    total = len(heldout)
    for u, item in heldout:
        hits += int(item in recs.get(u, [])[:k])
    return hits / total if total else 0.0


def ndcg_at_k(recs: Dict[int, List[int]], heldout: List[Tuple[int, int]], k: int = 10) -> float:
    score = 0.0
    total = len(heldout)
    for u, item in heldout:
        if u not in recs:
            continue
        try:
            rank = recs[u][:k].index(item)
            score += 1.0 / np.log2(rank + 2)
        except ValueError:
            continue
    return score / total if total else 0.0
