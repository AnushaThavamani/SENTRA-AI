from __future__ import annotations

from io import BytesIO
from pathlib import Path

import fitz
import numpy as np
import pytest

from app.config.rag import RAGSettings
from app.rag.session_manager import SessionRAGManager
from app.services import ResearchWorkspace


class Embedder:
    vocabulary = ("alpha", "methodology")
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[text.lower().split().count(word) for word in self.vocabulary] for text in texts], dtype=np.float32)


def pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    buffer = BytesIO()
    document.save(buffer)
    document.close()
    return buffer.getvalue()


def workspace(tmp_path: Path) -> ResearchWorkspace:
    manager = SessionRAGManager(settings=RAGSettings(session_root=tmp_path / "sessions"), embedder=Embedder())
    return ResearchWorkspace(manager)


def test_workspace_status_and_ingestion_are_session_scoped(tmp_path: Path) -> None:
    service = workspace(tmp_path)
    first, second = service.create_session(), service.create_session()
    assert not service.status(first).ready
    service.ingest(first, [("paper.pdf", pdf_bytes("Alpha methodology is documented."))])
    assert service.status(first).documents == ["paper.pdf"]
    assert service.status(first).chunk_count == 1
    assert not service.status(second).ready
    with pytest.raises(ValueError, match="already indexed"):
        service.ingest(first, [("paper.pdf", pdf_bytes("Alpha methodology is documented."))])


def test_workspace_deletion_does_not_affect_another_session(tmp_path: Path) -> None:
    service = workspace(tmp_path)
    first, second = service.create_session(), service.create_session()
    service.ingest(first, [("one.pdf", pdf_bytes("Alpha methodology."))])
    service.ingest(second, [("two.pdf", pdf_bytes("Alpha methodology."))])
    service.delete_session(first)
    with pytest.raises(KeyError, match="Unknown RAG session"):
        service.status(first)
    assert service.status(second).ready
