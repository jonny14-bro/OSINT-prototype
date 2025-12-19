import json
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np


class FaissManager:
    """Small FAISS manager for in-memory embedding storage and search.

    - Keeps a FAISS index (IndexFlatL2 or IndexHNSWFlat when available wrapped in IndexIDMap)
    - Maps stable string IDs (UUIDs) to FAISS internal integer ids
    - Stores simple metadata per item
    - Supports saving/loading the FAISS index and metadata to disk
    """

    def __init__(self, dim: int = 512, use_hnsw: bool = True):
        self.dim = dim
        self.lock = threading.Lock()

        # core index: try HNSW for faster queries, fallback to IndexFlatL2
        if use_hnsw and hasattr(faiss, 'IndexHNSWFlat'):
            base_index = faiss.IndexHNSWFlat(dim, 32)
        else:
            base_index = faiss.IndexFlatL2(dim)

        # wrap with IDMap so we can assign our own integer ids
        self.index = faiss.IndexIDMap(base_index)

        # mappings and metadata
        self._next_idx = 1  # monotonic integer for faiss ids
        self.idx_to_uid: Dict[int, str] = {}
        self.uid_to_idx: Dict[str, int] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        # store original embeddings (uid -> list[float]) so we can reconstruct queries reliably
        self.embeddings: Dict[str, List[float]] = {}
        # map file hash (sha256) -> uid for quick duplicate detection
        self.hash_to_uid: Dict[str, str] = {}

    def add(self, embedding: List[float], metadata: Optional[Dict[str, Any]] = None, file_hash: Optional[str] = None) -> str:
        """Add an embedding and optional metadata. Returns a stable UUID string id."""
        if not embedding:
            raise ValueError('Empty embedding')
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        if vec.shape[1] != self.dim:
            raise ValueError(f'Embedding dimension {vec.shape[1]} does not match index dim {self.dim}')

        with self.lock:
            idx = self._next_idx
            self._next_idx += 1
            uid = str(uuid.uuid4())

            ids = np.array([idx], dtype=np.int64)
            self.index.add_with_ids(vec, ids)

            self.idx_to_uid[idx] = uid
            self.uid_to_idx[uid] = idx
            self.metadata[uid] = metadata or {}
            # store embedding as plain list for persistence
            self.embeddings[uid] = vec.flatten().astype(float).tolist()
            if file_hash:
                try:
                    self.hash_to_uid[file_hash] = uid
                except Exception:
                    pass

        return uid

    def search_by_vector(self, query: List[float], k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search by a query vector. Returns list of (uid, distance, metadata)."""
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError('Query dimension mismatch')

        with self.lock:
            if self.index.ntotal == 0:
                return []
            D, I = self.index.search(q, k)

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for dist, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            uid = self.idx_to_uid.get(int(idx))
            meta = self.metadata.get(uid, {}) if uid else {}
            results.append((uid, float(dist), meta))
        return results

    def get_uid_by_hash(self, file_hash: str) -> Optional[str]:
        """Return stored uid for a given file SHA256 hash, or None if not present."""
        return self.hash_to_uid.get(file_hash)

    def find_duplicate_by_embedding(self, query: List[float], threshold: float = 1e-6) -> Optional[Tuple[str, float]]:
        """Check whether a query vector has a nearby stored neighbor within `threshold` distance.

        Returns (uid, distance) if found, otherwise None.
        """
        res = self.search_by_vector(query, k=1)
        if not res:
            return None
        uid, dist, _ = res[0]
        if dist <= float(threshold):
            return uid, dist
        return None

    def search_by_uid(self, uid: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Lookup embedding by uid and search nearest neighbors (excluding the query item itself)."""
        idx = self.uid_to_idx.get(uid)
        if idx is None:
            raise KeyError('uid not found')
        # use stored embedding if available
        emb_list = self.embeddings.get(uid)
        if emb_list is None:
            raise RuntimeError('Embedding for uid not stored; cannot perform uid-based search')
        res = self.search_by_vector(emb_list, k + 1)
        # filter out self (same uid)
        filtered = [r for r in res if r[0] != uid]
        return filtered[:k]

    def save(self, index_path: str, meta_path: str) -> None:
        """Persist FAISS index and metadata to disk."""
        with self.lock:
            faiss.write_index(self.index, index_path)
            payload = {
                'next_idx': self._next_idx,
                'idx_to_uid': self.idx_to_uid,
                'metadata': self.metadata,
                'embeddings': self.embeddings,
                'hash_to_uid': self.hash_to_uid,
            }
            with open(meta_path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh)

    def load(self, index_path: str, meta_path: str) -> None:
        """Load FAISS index and metadata from disk."""
        with self.lock:
            idx = faiss.read_index(index_path)
            # wrap in IDMap if not already
            if not isinstance(idx, faiss.IndexIDMap):
                idx = faiss.IndexIDMap(idx)
            self.index = idx
            with open(meta_path, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)
            self._next_idx = int(payload.get('next_idx', 1))
            self.idx_to_uid = {int(k): v for k, v in payload.get('idx_to_uid', {}).items()}
            self.uid_to_idx = {v: int(k) for k, v in self.idx_to_uid.items()}
            self.metadata = payload.get('metadata', {})
            self.embeddings = payload.get('embeddings', {})
            # restore hash map if present
            self.hash_to_uid = payload.get('hash_to_uid', {}) or {}

    def count(self) -> int:
        return int(self.index.ntotal)


__all__ = ['FaissManager']
