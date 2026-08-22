from typing import Dict, Iterable, List


def reciprocal_rank_fusion(rankings: List[Iterable[int]], rrf_k: int = 60) -> Dict[int, float]:
    """Fuse multiple ranked id lists into one score map (higher = better)."""
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return fused
