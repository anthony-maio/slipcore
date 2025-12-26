"""
Slipstream Protocol v3.0 - Semantic Manifold Quantization
Synthesizes:
1. User's 4D Semantic Structure & Extension Lifecycle
2. Research-based Vector Quantization & Clustering Engine
"""

import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

# --- ML Backend (Graceful Degradation) ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠️  ML libraries missing. Running in Heuristic Mode.")

# ==============================================================================
# 1. The Semantic Manifold (4D Structure)
# ==============================================================================

# Protocol Ranges
CORE_RANGE = (0x0000, 0x7FFF)      # 32k Universal Anchors
EXT_RANGE  = (0x8000, 0xFFFF)      # 32k Dynamic Local Anchors

@dataclass
class SemanticCoords:
    """The 4D logical position of an intent."""
    action: str   # e.g., 'REQUEST', 'INFORM', 'EVAL'
    polarity: int # -1 (Neg), 0 (Neut), 1 (Pos)
    domain: str   # e.g., 'AUTH', 'TASK', 'INFRA'
    urgency: int  # 0 (Low) to 3 (Critical)

@dataclass
class Anchor:
    """A quantized point in the semantic manifold."""
    index: int              # Wire ID: 0x004A
    mnemonic: str           # Readable: "ReqAuthCrit"
    coords: SemanticCoords  # Structure
    canonical: str          # Template: "Request auth review immediately"
    centroid: List[float]   # Vector Embedding
    usage_count: int = 0
    is_extension: bool = False

# ==============================================================================
# 2. Universal Concept Reference (UCR)
# ==============================================================================

class UCR:
    """
    The Manifold Manager.
    Handles the Vector Index and the Anchor Registry.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.anchors: Dict[int, Anchor] = {}
        self.centroids_matrix: Optional[np.ndarray] = None
        self.indices_list: List[int] = []
        
        # Extension State
        self.next_ext_index = EXT_RANGE[0]
        self.fallback_buffer: List[str] = []
        
        # ML Engine
        self.model = None
        if HAS_ML:
            print(f"🧠 Loading Quantization Engine ({model_name})...")
            self.model = SentenceTransformer(model_name)

    def add_anchor(self, anchor: Anchor):
        self.anchors[anchor.index] = anchor
        self.centroids_matrix = None # Invalidate cache

    def rebuild_index(self):
        """Prepare vector matrix for fast cosine similarity search."""
        if not self.anchors: return
        self.indices_list = list(self.anchors.keys())
        matrix = [self.anchors[i].centroid for i in self.indices_list]
        
        # Normalize for Cosine Similarity
        self.centroids_matrix = np.array(matrix)
        norms = np.linalg.norm(self.centroids_matrix, axis=1, keepdims=True)
        self.centroids_matrix = self.centroids_matrix / (norms + 1e-9)

    def quantize(self, thought: str, threshold: float = 0.55) -> Tuple[Optional[Anchor], float]:
        """
        The Core Loop: Projects a thought onto the Manifold.
        """
        if not self.model: return self._heuristic_quantize(thought)
        if self.centroids_matrix is None: self.rebuild_index()

        # 1. Embed
        vec = self.model.encode(thought)
        vec = vec / (np.linalg.norm(vec) + 1e-9)

        # 2. Search
        scores = np.dot(self.centroids_matrix, vec)
        best_idx_loc = np.argmax(scores)
        confidence = float(scores[best_idx_loc])
        
        # 3. Threshold
        if confidence < threshold:
            self._record_fallback(thought)
            return None, confidence

        anchor = self.anchors[self.indices_list[best_idx_loc]]
        anchor.usage_count += 1
        return anchor, confidence

    def _record_fallback(self, thought: str):
        self.fallback_buffer.append(thought)

    def _heuristic_quantize(self, thought: str):
        """Backup logic if ML is missing."""
        for a in self.anchors.values():
            if a.coords.action.lower() in thought.lower():
                return a, 0.5
        return None, 0.0

# ==============================================================================
# 3. Extension Manager (Evolutionary Layer)
# ==============================================================================

class ExtensionManager:
    """
    Analyzes fallback traffic to 'invent' new protocol anchors via Clustering.
    """
    def propose_extensions(self, ucr: UCR, min_cluster_size=3) -> List[Anchor]:
        if not HAS_ML or len(ucr.fallback_buffer) < min_cluster_size:
            return []

        print(f"⚙️  Analyzing {len(ucr.fallback_buffer)} fallbacks for new concepts...")
        embeddings = ucr.model.encode(ucr.fallback_buffer)
        
        # Dynamic K estimation
        n_clusters = max(1, len(ucr.fallback_buffer) // min_cluster_size)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init="auto")
        kmeans.fit(embeddings)
        
        new_anchors = []
        for i in range(n_clusters):
            indices = np.where(kmeans.labels_ == i)[0]
            if len(indices) >= min_cluster_size:
                # Use the centroid of the cluster as the anchor vector
                cluster_center = kmeans.cluster_centers_[i]
                
                # Find the representative text (closest to center)
                cluster_embeds = embeddings[indices]
                dists = np.linalg.norm(cluster_embeds - cluster_center, axis=1)
                exemplar = ucr.fallback_buffer[indices[np.argmin(dists)]]
                
                # Generate a new Anchor
                idx = ucr.next_ext_index
                ucr.next_ext_index += 1
                
                # Heuristic Mnemonic: "EXT_" + first 2 words
                words = exemplar.split()[:2]
                mnemonic = "EXT_" + "".join(w.capitalize() for w in words)
                mnemonic = "".join(c for c in mnemonic if c.isalnum())
                
                new_anchors.append(Anchor(
                    index=idx,
                    mnemonic=mnemonic,
                    coords=SemanticCoords("EXT", 0, "DYN", 1), # Generic Ext coords
                    canonical=exemplar,
                    centroid=cluster_center.tolist(),
                    is_extension=True
                ))
                
        # Clear buffer after processing
        ucr.fallback_buffer = [] 
        return new_anchors

# ==============================================================================
# 4. Wire Protocol
# ==============================================================================

def encode_wire(src: str, dst: str, anchor: Anchor, payload: List[str] = None) -> str:
    """
    SLIP v3 <src> <dst> <HEX_ID> [payload]
    """
    hex_id = f"0x{anchor.index:04X}"
    payload_str = json.dumps(payload) if payload else ""
    return f"SLIP v3 {src} {dst} {hex_id} {payload_str}".strip()

def decode_wire(wire: str, ucr: UCR) -> Dict:
    parts = wire.split(" ", 4)
    if parts[0] != "SLIP": raise ValueError("Invalid Protocol")
    
    id_part = parts[4].split(" ")[0]
    idx = int(id_part, 16)
    anchor = ucr.anchors.get(idx)
    
    return {
        "intent": anchor.canonical if anchor else "UNKNOWN",
        "coords": asdict(anchor.coords) if anchor else None,
        "is_ext": anchor.is_extension if anchor else False
    }

# ==============================================================================
# 5. Bootstrap & Demo
# ==============================================================================

def bootstrap_core(ucr: UCR):
    """Bootstraps the 4D Semantic Manifold."""
    # (Index, Mnemonic, Action, Polarity, Domain, Urgency, Template)
    core_data = [
        (0x0010, "ReqTask", "REQ", 0, "TASK", 1, "Request a new task assignment"),
        (0x0011, "ReqRev",  "REQ", 0, "QA", 2,   "Request code review"),
        (0x0020, "InfDone", "INF", 1, "TASK", 0, "Inform that task is complete"),
        (0x0030, "ErrCrit", "INF", -1, "ERR", 3, "Critical system failure detected"),
    ]
    
    if HAS_ML:
        vectors = ucr.model.encode([x[6] for x in core_data])
    else:
        vectors = [[0.0]*384 for _ in core_data]

    for i, data in enumerate(core_data):
        ucr.add_anchor(Anchor(
            index=data[0], mnemonic=data[1],
            coords=SemanticCoords(data[2], data[3], data[4], data[5]),
            canonical=data[6], centroid=vectors[i]
        ))
    ucr.rebuild_index()

if __name__ == "__main__":
    ucr = UCR()
    ext_mgr = ExtensionManager()
    bootstrap_core(ucr)
    
    print("\n" + "="*60)
    print("SLIPSTREAM v3 - SEMANTIC MANIFOLD QUANTIZATION")
    print("="*60)
    
    # 1. Core Manifold Usage
    thoughts = [
        "I finished the job successfully.",
        "System is crashing! Help!",
        "Can you review my code?"
    ]
    
    print("\n--- Phase 1: Core Manifold ---")
    for t in thoughts:
        anchor, conf = ucr.quantize(t)
        if anchor:
            print(f"Thought: '{t}'")
            print(f"  -> Mapped: {anchor.mnemonic} (Conf: {conf:.2f})")
            print(f"  -> Coords: {anchor.coords}")
            print(f"  -> Wire:   {encode_wire('A', 'B', anchor)}")
        else:
            print(f"Thought: '{t}' -> FALLBACK")
    
    # 2. Extension Learning
    print("\n--- Phase 2: Evolutionary Learning ---")
    drift_pattern = "Please scale the kubernetes cluster"
    
    # Simulate repeated traffic
    for _ in range(4):
        ucr.quantize(drift_pattern) # Will fail and record fallback
        
    new_anchors = ext_mgr.propose_extensions(ucr)
    for a in new_anchors:
        print(f"🆕 Learned Anchor 0x{a.index:04X}: '{a.canonical}'")
        ucr.add_anchor(a)
    
    # 3. Verify Learning
    ucr.rebuild_index()
    anchor, conf = ucr.quantize(drift_pattern)
    print(f"\nRetrying: '{drift_pattern}'")
    print(f"  -> Mapped: {anchor.mnemonic} (Conf: {conf:.2f})")
    print(f"  -> Wire:   {encode_wire('A', 'B', anchor)}")