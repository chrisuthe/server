"""Pure similarity functions — no MA dependencies.

Centroid blending, union merging, and MMR diversity re-ranking.
All functions operate on plain lists of floats and numpy arrays.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def combine_seeds_centroid(
    seeds: list[list[float]],
    weights: list[float] | None = None,
) -> list[float]:
    """Compute weighted average of seed signature vectors.

    :param seeds: List of signature vectors (all same dimensionality).
    :param weights: Per-seed weights. If None, equal weighting.
    :raises ValueError: If seeds is empty or weights length mismatches.
    """
    if not seeds:
        msg = "Cannot compute centroid from at least one seed"
        raise ValueError(msg)
    if weights is not None and len(weights) != len(seeds):
        msg = f"weights length ({len(weights)}) must match seeds length ({len(seeds)})"
        raise ValueError(msg)

    arr = np.array(seeds, dtype=np.float64)
    if weights is None:
        centroid = arr.mean(axis=0)
    else:
        w = np.array(weights, dtype=np.float64)
        w = w / w.sum()
        centroid = (arr * w[:, np.newaxis]).sum(axis=0)

    return [float(v) for v in centroid]


def merge_union_results(
    neighborhoods: list[list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """Merge per-seed ANN results, keeping the best distance per track.

    :param neighborhoods: List of result lists, each containing (item_id, distance) pairs.
    """
    if not neighborhoods:
        return []

    best: dict[str, float] = {}
    for neighborhood in neighborhoods:
        for item_id, dist in neighborhood:
            if item_id not in best or dist < best[item_id]:
                best[item_id] = dist

    merged = list(best.items())
    merged.sort(key=lambda x: x[1])
    return merged


def apply_mmr(
    candidates: list[tuple[str, list[float], float]],
    seed_vec: list[float],
    diversity: float,
    limit: int,
) -> list[tuple[str, float]]:
    """Apply Maximal Marginal Relevance to re-rank candidates for diversity.

    :param candidates: List of (item_id, normalized_features, distance) tuples.
    :param seed_vec: The seed signature vector (normalized).
    :param diversity: MMR lambda, 0.0 = pure relevance, 1.0 = max diversity.
    :param limit: Maximum number of results to return.
    """
    if not candidates:
        return []

    cand_vecs = {cid: np.array(vec, dtype=np.float64) for cid, vec, _d in candidates}
    seed_arr = np.array(seed_vec, dtype=np.float64)
    seed_norm = float(np.linalg.norm(seed_arr))

    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    relevance: dict[str, float] = {}
    for cid, _vec, _d in candidates:
        relevance[cid] = _cosine_sim(cand_vecs[cid], seed_arr) if seed_norm > 0 else 0.0

    selected: list[tuple[str, float]] = []
    remaining = {cid for cid, _, _ in candidates}
    dist_lookup = {cid: d for cid, _, d in candidates}

    for _ in range(min(limit, len(candidates))):
        best_id: str | None = None
        best_score = -float("inf")

        for cid in remaining:
            rel = relevance[cid]
            if not selected:
                redundancy = 0.0
            else:
                redundancy = max(_cosine_sim(cand_vecs[cid], cand_vecs[sid]) for sid, _ in selected)
            score = (1.0 - diversity) * rel - diversity * redundancy
            if score > best_score:
                best_score = score
                best_id = cid

        if best_id is None:
            break
        remaining.discard(best_id)
        selected.append((best_id, dist_lookup[best_id]))

    return selected


def expand_recursive(
    initial_seeds: list[list[float]],
    searcher: Callable[
        [list[list[float]], set[str]],
        list[tuple[str, str, list[float], float]],
    ],
    depth: int,
    branch_factor: int,
) -> list[tuple[str, str, list[float], float, int]]:
    """Expand similarity search across multiple generations.

    :param initial_seeds: Seed signature vectors for generation 0.
    :param searcher: Callback that takes (seed_vectors, seen_ids) and returns
        list of (item_id, provider, features, distance).
    :param depth: Number of generations to run.
    :param branch_factor: How many top results from each generation become seeds.
    """
    all_results: list[tuple[str, str, list[float], float, int]] = []
    seen: set[str] = set()
    current_seeds = initial_seeds

    for gen in range(depth):
        gen_results = searcher(current_seeds, seen)
        new_results: list[tuple[str, str, list[float], float]] = []
        for item_id, provider, features, dist in gen_results:
            if item_id not in seen:
                seen.add(item_id)
                new_results.append((item_id, provider, features, dist))
                all_results.append((item_id, provider, features, dist, gen))

        if not new_results or gen == depth - 1:
            break

        new_results.sort(key=lambda x: x[3])
        next_seeds = [features for _, _, features, _ in new_results[:branch_factor]]
        current_seeds = next_seeds

    return all_results
