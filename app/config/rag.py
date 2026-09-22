"""Small, centralized configuration for the temporary RAG session store."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RAGSettings:
    session_root: Path
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    default_top_k: int = 5
    # Scores are derived from FAISS squared-L2 distances as 1 / (1 + distance).
    # This is deliberately centralized so applications do not silently use a
    # different evidence bar from the RAG layer.
    relevance_threshold: float = 0.20
    session_ttl_hours: int = 24


def get_rag_settings() -> RAGSettings:
    project_root = Path(__file__).resolve().parents[2]
    configured_root = os.getenv("SENTRA_RAG_SESSION_ROOT")
    return RAGSettings(session_root=Path(configured_root) if configured_root else project_root / "data" / "sessions")
