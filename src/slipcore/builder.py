"""
UCR Builder - Construct UCRs from Message Corpora

Builds production-ready Universal Concept References by:
1. Embedding messages using sentence-transformers
2. Clustering via MiniBatchKMeans (or greedy cosine fallback)
3. Extracting canonical templates from cluster exemplars
4. Inferring semantic coordinates for each cluster

This enables domain-specific UCR construction beyond the core anchors.

Usage:
    from slipcore.builder import UCRBuilder

    # Collect agent messages
    messages = ["Please review this code", "Task completed", ...]

    # Build UCR
    builder = UCRBuilder(n_clusters=1024)
    ucr = builder.build(messages, domain="software_engineering")
    ucr.save("my_ucr.json")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json
import re
from datetime import datetime

from .ucr import UCR, UCRAnchor, CORE_RANGE_END, create_base_ucr


# ============ Optional Dependencies ============

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _HAS_SBERT = False

try:
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SKLEARN = True
except ImportError:
    MiniBatchKMeans = None  # type: ignore
    _HAS_SKLEARN = False


# ============ Build Statistics ============

@dataclass
class BuildStats:
    """Statistics from UCR building process."""
    n_messages: int = 0
    n_clusters_requested: int = 0
    n_clusters_created: int = 0
    n_clusters_skipped: int = 0
    embedding_dim: int = 0
    embedding_time_sec: float = 0.0
    clustering_time_sec: float = 0.0
    total_time_sec: float = 0.0
    coverage_estimate: float = 0.0
    avg_cluster_size: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_messages": self.n_messages,
            "n_clusters_requested": self.n_clusters_requested,
            "n_clusters_created": self.n_clusters_created,
            "n_clusters_skipped": self.n_clusters_skipped,
            "embedding_dim": self.embedding_dim,
            "embedding_time_sec": round(self.embedding_time_sec, 2),
            "clustering_time_sec": round(self.clustering_time_sec, 2),
            "total_time_sec": round(self.total_time_sec, 2),
            "coverage_estimate": round(self.coverage_estimate, 3),
            "avg_cluster_size": round(self.avg_cluster_size, 1),
        }


# ============ Mnemonic Generation ============

def _generate_mnemonic(text: str, category: str = "", max_len: int = 24) -> str:
    """
    Generate a CamelCase mnemonic from text.

    Examples:
        "Please review this code" -> "ReviewThisCode"
        "Task completed successfully" -> "TaskCompleted"
    """
    # Extract meaningful words
    words = re.findall(r'[A-Za-z]+', text)
    # Filter common stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its',
                 'our', 'their', 'please', 'for', 'to', 'of', 'in', 'on', 'at'}
    words = [w for w in words if w.lower() not in stopwords]

    # Take first 4 words, capitalize
    mnemonic = "".join(w.capitalize() for w in words[:4])

    # Add category prefix if short
    if category and len(mnemonic) < 8:
        prefix = category.split('/')[0].capitalize()[:4]
        mnemonic = prefix + mnemonic

    # Ensure valid identifier
    mnemonic = re.sub(r'[^A-Za-z0-9]', '', mnemonic)

    return mnemonic[:max_len] if mnemonic else "Anchor"


def _infer_category(text: str) -> str:
    """Infer message category from text content."""
    low = text.lower()

    # Request patterns
    if any(k in low for k in ("please", "can you", "could you", "would you", "need you to")):
        if any(k in low for k in ("review", "check", "look at")):
            return "request/review"
        if any(k in low for k in ("help", "assist", "support")):
            return "request/help"
        return "request/task"

    # Completion patterns
    if any(k in low for k in ("completed", "finished", "done", "succeeded")):
        return "inform/complete"

    # Progress patterns
    if any(k in low for k in ("working on", "in progress", "started")):
        return "inform/progress"

    # Error patterns
    if any(k in low for k in ("error", "failed", "exception", "crash", "bug")):
        return "inform/error"

    # Question patterns
    if text.strip().endswith("?"):
        return "ask/clarify"

    # Proposal patterns
    if any(k in low for k in ("suggest", "propose", "recommend", "could try")):
        return "propose/plan"

    # Approval patterns
    if any(k in low for k in ("approved", "lgtm", "looks good", "accept")):
        return "eval/approve"

    # Rejection patterns
    if any(k in low for k in ("rejected", "denied", "cannot", "won't work")):
        return "eval/reject"

    return "general"


def _infer_coords(text: str, category: str) -> Tuple[int, int, int, int]:
    """
    Infer semantic coordinates from text and category.

    Returns (action, polarity, domain, urgency) tuple.
    """
    low = text.lower()

    # ACTION (0-7)
    action = 3  # default: request
    if category.startswith("inform"):
        action = 1
    elif category.startswith("ask"):
        action = 2
    elif category.startswith("request"):
        action = 3
    elif category.startswith("propose"):
        action = 4
    elif category.startswith("eval"):
        action = 6
    elif any(k in low for k in ("observe", "notice", "detect")):
        action = 0

    # POLARITY (0-7)
    polarity = 4  # neutral
    if any(k in low for k in ("error", "failed", "crash", "bug", "rejected", "denied")):
        polarity = 1  # negative
    elif any(k in low for k in ("success", "completed", "approved", "good", "fixed")):
        polarity = 6  # positive

    # DOMAIN (0-7)
    domain = 7  # general
    if any(k in low for k in ("task", "work", "implement", "build")):
        domain = 0
    elif any(k in low for k in ("plan", "strategy", "approach")):
        domain = 1
    elif any(k in low for k in ("review", "evaluate", "assess")):
        domain = 3
    elif any(k in low for k in ("error", "exception", "fail")):
        domain = 6

    # URGENCY (0-7)
    urgency = 4  # normal
    if any(k in low for k in ("critical", "urgent", "asap", "immediately")):
        urgency = 7
    elif any(k in low for k in ("important", "priority", "soon")):
        urgency = 5
    elif any(k in low for k in ("background", "low priority", "when possible")):
        urgency = 1

    return (action, polarity, domain, urgency)


# ============ UCR Builder ============

class UCRBuilder:
    """
    Build UCRs from message corpora via semantic clustering.

    Process:
    1. Embed messages using sentence-transformers
    2. Cluster with MiniBatchKMeans (or greedy cosine fallback)
    3. Extract exemplar (nearest to centroid) as template
    4. Infer category and coordinates
    5. Generate mnemonics

    Recommended corpus sizes:
    - 1k-10k messages: 256-1024 clusters
    - 10k-100k messages: 1024-8192 clusters
    - 100k+ messages: 8192-16384 clusters

    Example:
        builder = UCRBuilder(n_clusters=1024)
        ucr = builder.build(messages, domain="customer_service")
        ucr.save("customer_service_ucr.json")
    """

    def __init__(
        self,
        n_clusters: int = 1024,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_cluster_size: int = 2,
        include_core: bool = True,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        """
        Initialize UCR builder.

        Args:
            n_clusters: Target number of clusters (anchors)
            embedding_model: sentence-transformers model name
            min_cluster_size: Minimum messages per cluster to create anchor
            include_core: Include core UCR anchors in result
            batch_size: Embedding batch size
            random_state: Random seed for reproducibility
        """
        if not _HAS_NUMPY:
            raise ImportError("numpy is required: pip install numpy")
        if not _HAS_SBERT:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")

        self.n_clusters = n_clusters
        self.embedding_model_name = embedding_model
        self.min_cluster_size = min_cluster_size
        self.include_core = include_core
        self.batch_size = batch_size
        self.random_state = random_state

        self._model: Optional[SentenceTransformer] = None
        self._embed_dim: Optional[int] = None

    def _ensure_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)

    def _embed_batch(self, texts: List[str]) -> "np.ndarray":
        """Embed texts and return normalized vectors."""
        self._ensure_model()
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        self._embed_dim = vecs.shape[1]
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-12)

    def build(
        self,
        messages: List[str],
        domain: str = "general",
        version: str = "1.0.0",
    ) -> UCR:
        """
        Build UCR from message corpus.

        Args:
            messages: List of agent messages (10k-100k recommended)
            domain: Domain identifier for the UCR
            version: Version string

        Returns:
            Constructed UCR with clustered anchors
        """
        import time
        start_time = time.time()

        stats = BuildStats(
            n_messages=len(messages),
            n_clusters_requested=self.n_clusters,
        )

        # Filter empty messages
        messages = [m.strip() for m in messages if m and m.strip()]
        if len(messages) < self.n_clusters:
            print(f"Warning: Only {len(messages)} messages for {self.n_clusters} clusters")
            self.n_clusters = max(1, len(messages) // 3)

        print(f"\n{'='*60}")
        print(f"Building UCR: {self.n_clusters} clusters from {len(messages)} messages")
        print(f"{'='*60}\n")

        # Step 1: Embed
        print("[1/4] Generating embeddings...")
        embed_start = time.time()
        embeddings = self._embed_batch(messages)
        stats.embedding_time_sec = time.time() - embed_start
        stats.embedding_dim = embeddings.shape[1]
        print(f"      Shape: {embeddings.shape}, Time: {stats.embedding_time_sec:.1f}s\n")

        # Step 2: Cluster
        print("[2/4] Clustering...")
        cluster_start = time.time()

        if _HAS_SKLEARN:
            print(f"      Using MiniBatchKMeans (k={self.n_clusters})")
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=1024,
                max_iter=100,
                random_state=self.random_state,
                n_init="auto",
            )
            labels = kmeans.fit_predict(embeddings)
            centroids = kmeans.cluster_centers_
        else:
            print("      Using greedy cosine clustering (sklearn not available)")
            labels, centroids = self._greedy_cluster(embeddings, self.n_clusters)

        stats.clustering_time_sec = time.time() - cluster_start
        print(f"      Time: {stats.clustering_time_sec:.1f}s\n")

        # Step 3: Extract anchors
        print("[3/4] Extracting anchors...")

        # Start with core UCR if requested
        if self.include_core:
            ucr = create_base_ucr()
            ucr.version = version
            next_index = CORE_RANGE_END
        else:
            ucr = UCR(version=version)
            next_index = 0

        # Process each cluster
        anchors_created = 0
        anchors_skipped = 0
        cluster_sizes = []

        for cluster_id in range(self.n_clusters):
            # Get messages in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = len(cluster_indices)

            if cluster_size < self.min_cluster_size:
                anchors_skipped += 1
                continue

            cluster_sizes.append(cluster_size)

            # Get cluster embeddings and centroid
            cluster_embeds = embeddings[cluster_indices]
            centroid = centroids[cluster_id]

            # Normalize centroid
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            # Find exemplar (nearest to centroid)
            dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
            exemplar_idx = cluster_indices[np.argmin(dists)]
            exemplar = messages[exemplar_idx]

            # Infer category and coordinates
            category = _infer_category(exemplar)
            coords = _infer_coords(exemplar, category)

            # Generate mnemonic
            mnemonic = _generate_mnemonic(exemplar, category)

            # Ensure unique mnemonic
            base_mnemonic = mnemonic
            counter = 1
            while ucr.get_by_mnemonic(mnemonic) is not None:
                mnemonic = f"{base_mnemonic}{counter}"
                counter += 1

            # Create anchor
            anchor = UCRAnchor(
                index=next_index,
                mnemonic=mnemonic,
                canonical=exemplar[:200],  # Truncate long messages
                coords=coords,
                is_core=False,
                centroid=centroid.tolist(),
            )

            ucr.add_anchor(anchor)
            next_index += 1
            anchors_created += 1

        stats.n_clusters_created = anchors_created
        stats.n_clusters_skipped = anchors_skipped
        stats.avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0

        print(f"      Created: {anchors_created} anchors")
        print(f"      Skipped: {anchors_skipped} (below min size)\n")

        # Step 4: Validate coverage
        print("[4/4] Validating coverage...")
        coverage = self._estimate_coverage(ucr, messages[:min(1000, len(messages))], embeddings[:min(1000, len(embeddings))])
        stats.coverage_estimate = coverage
        stats.total_time_sec = time.time() - start_time

        print(f"      Coverage: {coverage:.1%}")
        print(f"      Total time: {stats.total_time_sec:.1f}s\n")

        print(f"{'='*60}")
        print(f"UCR BUILD COMPLETE")
        print(f"  Anchors: {len(ucr.anchors)} ({len(ucr.core_anchors())} core + {len(ucr.extension_anchors())} learned)")
        print(f"  Coverage: {coverage:.1%}")
        print(f"{'='*60}\n")

        # Store stats in UCR metadata (informal)
        ucr._build_stats = stats

        return ucr

    def _greedy_cluster(
        self,
        embeddings: "np.ndarray",
        n_clusters: int,
        sim_threshold: float = 0.75,
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """
        Greedy cosine clustering fallback when sklearn unavailable.

        Returns (labels, centroids).
        """
        n_samples = embeddings.shape[0]
        labels = np.full(n_samples, -1, dtype=np.int32)
        centroids_list: List[np.ndarray] = []
        cluster_members: List[List[int]] = []

        for i in range(n_samples):
            vec = embeddings[i]
            placed = False

            # Check existing centroids
            for c_idx, centroid in enumerate(centroids_list):
                if len(centroids_list) < n_clusters or c_idx < len(centroids_list):
                    sim = float(np.dot(vec, centroid))
                    if sim >= sim_threshold:
                        labels[i] = c_idx
                        cluster_members[c_idx].append(i)
                        # Update centroid
                        new_c = np.mean(embeddings[cluster_members[c_idx]], axis=0)
                        norm = np.linalg.norm(new_c)
                        centroids_list[c_idx] = new_c / (norm + 1e-12) if norm > 0 else new_c
                        placed = True
                        break

            # Create new cluster if not placed and under limit
            if not placed and len(centroids_list) < n_clusters:
                labels[i] = len(centroids_list)
                centroids_list.append(vec.copy())
                cluster_members.append([i])

        # Handle unassigned (assign to nearest)
        unassigned = np.where(labels == -1)[0]
        if len(unassigned) > 0 and len(centroids_list) > 0:
            centroids_mat = np.array(centroids_list)
            for idx in unassigned:
                sims = np.dot(centroids_mat, embeddings[idx])
                labels[idx] = np.argmax(sims)

        return labels, np.array(centroids_list)

    def _estimate_coverage(
        self,
        ucr: UCR,
        messages: List[str],
        embeddings: "np.ndarray",
        threshold: float = 0.5,
    ) -> float:
        """Estimate coverage by checking how many messages quantize well."""
        if not ucr.anchors:
            return 0.0

        # Build centroid matrix from anchors with centroids
        anchors_with_centroids = [a for a in ucr.anchors.values() if a.centroid is not None]
        if not anchors_with_centroids:
            return 0.0

        centroids = np.array([a.centroid for a in anchors_with_centroids], dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / (norms + 1e-12)

        covered = 0
        for emb in embeddings:
            sims = np.dot(centroids, emb)
            if np.max(sims) >= threshold:
                covered += 1

        return covered / len(embeddings)


# ============ Convenience Functions ============

def build_ucr_from_corpus(
    messages: List[str],
    n_clusters: int = 1024,
    domain: str = "general",
    output_path: Optional[str] = None,
) -> UCR:
    """
    Convenience function to build a UCR from messages.

    Args:
        messages: List of agent messages
        n_clusters: Target number of clusters
        domain: Domain identifier
        output_path: Optional path to save UCR JSON

    Returns:
        Constructed UCR

    Example:
        ucr = build_ucr_from_corpus(messages, n_clusters=2048, domain="coding")
    """
    builder = UCRBuilder(n_clusters=n_clusters)
    ucr = builder.build(messages, domain=domain)

    if output_path:
        ucr.save(Path(output_path))
        print(f"Saved UCR to: {output_path}")

    return ucr


# ============ CLI ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build UCR from message corpus")
    parser.add_argument("input", help="Input file (JSONL with 'message' field, or plain text)")
    parser.add_argument("-o", "--output", default="ucr.json", help="Output UCR file")
    parser.add_argument("-n", "--n-clusters", type=int, default=1024, help="Number of clusters")
    parser.add_argument("-d", "--domain", default="general", help="Domain identifier")

    args = parser.parse_args()

    # Load messages
    input_path = Path(args.input)
    messages = []

    if input_path.suffix == ".jsonl":
        with open(input_path) as f:
            for line in f:
                data = json.loads(line)
                if "message" in data:
                    messages.append(data["message"])
                elif "text" in data:
                    messages.append(data["text"])
    else:
        with open(input_path) as f:
            messages = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(messages)} messages from {input_path}")

    # Build UCR
    ucr = build_ucr_from_corpus(
        messages,
        n_clusters=args.n_clusters,
        domain=args.domain,
        output_path=args.output,
    )
