"""ML-based extension learning (requires slipcore[ml]).

Cluster fallback traffic to discover new anchor candidates.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from slipcore.ucr import UCR, UCRAnchor, AnchorState, Coords

try:
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def propose_extensions_ml(
    ucr: UCR,
    fallback_texts: List[str],
    embed_batch: Callable[[List[str]], np.ndarray],
    min_cluster_size: int = 3,
    max_new: int = 32,
    greedy_sim_threshold: float = 0.78,
) -> List[UCRAnchor]:
    """Learn new anchors from fallback traffic using ML clustering.

    Embeds all fallback texts, clusters them, and for each sufficiently
    large cluster, mints a new extension anchor.

    Args:
        ucr: UCR to add extensions to
        fallback_texts: List of fallback messages to cluster
        embed_batch: Function to embed a list of texts
        min_cluster_size: Minimum messages in a cluster to create anchor
        max_new: Maximum number of new anchors to create
        greedy_sim_threshold: Similarity threshold for greedy clustering

    Returns:
        List of proposed new anchors (not yet added to UCR)
    """
    buf = [t for t in fallback_texts if t and t.strip()]
    if len(buf) < min_cluster_size:
        return []

    embeds = np.asarray(embed_batch(buf), dtype=np.float32)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    embeds = embeds / (norms + 1e-12)

    clusters: List[List[int]] = []

    if _HAS_SKLEARN and len(buf) >= (min_cluster_size * 2):
        k = max(1, min(max_new, len(buf) // max(1, min_cluster_size)))
        km = MiniBatchKMeans(n_clusters=k, n_init="auto", random_state=42)
        km.fit(embeds)
        for i in range(k):
            idxs = np.where(km.labels_ == i)[0].tolist()
            if idxs:
                clusters.append(idxs)
    else:
        # Greedy cosine clustering fallback
        centroids_list: List[np.ndarray] = []
        for i, v in enumerate(embeds):
            placed = False
            for c_idx, c in enumerate(centroids_list):
                if float(np.dot(v, c)) >= greedy_sim_threshold:
                    clusters[c_idx].append(i)
                    new_c = np.mean(embeds[clusters[c_idx]], axis=0)
                    n = np.linalg.norm(new_c)
                    centroids_list[c_idx] = new_c / (n + 1e-12) if n > 0 else new_c
                    placed = True
                    break
            if not placed:
                clusters.append([i])
                centroids_list.append(v.copy())

    new_anchors: List[UCRAnchor] = []
    minted = 0

    for idxs in clusters:
        if len(idxs) < min_cluster_size or minted >= max_new:
            continue

        cluster_embeds = embeds[idxs]
        centroid = np.mean(cluster_embeds, axis=0)
        n = np.linalg.norm(centroid)
        centroid = centroid / (n + 1e-12) if n > 0 else centroid

        dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
        exemplar = buf[idxs[int(np.argmin(dists))]].strip()

        # Simple coord estimation
        coords: Coords = (3, 4, 7, 4)  # default: request, neutral, general, normal

        words = exemplar.split()[:3]
        obj_name = "".join(w.capitalize() for w in words if w.isalpha())[:20] or "Custom"

        next_idx = ucr.next_extension_index()
        if next_idx > 0xFFFF:
            break

        anchor = UCRAnchor(
            index=next_idx,
            force="Request",
            obj=obj_name,
            canonical=exemplar,
            coords=coords,
            is_core=False,
            state=AnchorState.DRAFT,
            created_by="ml_clustering",
        )
        new_anchors.append(anchor)
        minted += 1

    return new_anchors
