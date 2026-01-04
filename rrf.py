from typing import List, Dict, Tuple
from collections import defaultdict
import config

def weighted_rrf(rank_lists: List[List[str]], weights: List[float], k: int = 60) -> List[Tuple[str, float]]:
    """
    Performs Weighted Reciprocal Rank Fusion on multiple rank lists.
    rank_lists: List of lists, where each inner list contains item IDs in ranked order.
    weights: List of weights corresponding to each rank list.
    k: RRF constant (usually 60).
    
    Returns: List of (item_id, score) sorted by score descending.
    """
    if len(rank_lists) != len(weights):
        raise ValueError("Number of rank lists must match number of weights")

    rrf_scores = defaultdict(float)

    for r_list, w in zip(rank_lists, weights):
        for rank, item_id in enumerate(r_list):
            # Rank is 0-indexed here, formal RRF usually 1-indexed (rank+1)
            # score = w * (1 / (k + rank + 1))
            score = w * (1.0 / (k + rank + 1))
            rrf_scores[item_id] += score

    # Sort
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items
