"""In-memory, session-scoped FAISS vector store and metadata mapping."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import numpy as np
from .models import Chunk


class FaissVectorStore:
    """FAISS IndexFlatL2 store; metadata remains in the matching chunk list."""
    def __init__(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            raise ValueError("Cannot create a vector store without chunks.")
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(chunks) or vectors.shape[1] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array matching the chunk count.")
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required to create a vector store.") from exc
        self.chunks = list(chunks)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors))

    @property
    def size(self) -> int:
        return self.index.ntotal

    def save(self, index_path: Path) -> None:
        """Persist this session's FAISS index; metadata is saved by the session manager."""
        import faiss
        faiss.write_index(self.index, str(index_path))

    @classmethod
    def load(cls, index_path: Path, chunks: Sequence[Chunk]) -> "FaissVectorStore":
        """Reopen a saved session index with its separately persisted metadata."""
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required to load a vector store.") from exc
        instance = cls.__new__(cls)
        instance.chunks = list(chunks)
        instance.index = faiss.read_index(str(index_path))
        if instance.index.ntotal != len(instance.chunks):
            raise ValueError("Session index and metadata have different vector counts.")
        return instance

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError("A search query must contain exactly one embedding vector.")
        distances, positions = self.index.search(np.ascontiguousarray(vector), min(top_k, self.size))
        return [(self.chunks[int(pos)], float(distance)) for distance, pos in zip(distances[0], positions[0])
                if 0 <= int(pos) < len(self.chunks)]
