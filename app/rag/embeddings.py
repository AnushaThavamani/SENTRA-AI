"""Reusable embedding adapter for the legacy embedding model."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np


class SentenceTransformerEmbedder:
    """Lazily load and reuse all-MiniLM-L6-v2 rather than loading per query."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.model_name
        self._model: object | None = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for production embeddings. Install requirements.txt.") from exc
            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(list(texts), show_progress_bar=False), dtype=np.float32)
