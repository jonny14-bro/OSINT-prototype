from .faiss_manager import FaissManager
import os

FAISS_USE_HNSW = os.getenv('FAISS_USE_HNSW', 'false').lower() in ('1', 'true', 'yes')

# Text embeddings (MiniLM)
text_faiss = FaissManager(dim=384, use_hnsw=FAISS_USE_HNSW)

# Image / Video / Audio embeddings (CLIP)
vision_faiss = FaissManager(dim=512, use_hnsw=FAISS_USE_HNSW)

__all__ = ["text_faiss", "vision_faiss"]
