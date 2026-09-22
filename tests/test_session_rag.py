from __future__ import annotations

from io import BytesIO
from pathlib import Path

import fitz
import numpy as np

from app.config.rag import RAGSettings
from app.rag.session_manager import SessionRAGManager


class KeywordEmbedder:
    """Deterministic fixture replacing the external production model in tests."""
    vocabulary = ("alpha", "beta", "retrieval", "monitoring")

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[text.lower().split().count(word) for word in self.vocabulary] for text in texts], dtype=np.float32)


def pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    output = BytesIO()
    document.save(output)
    document.close()
    return output.getvalue()


def manager(tmp_path: Path) -> SessionRAGManager:
    settings = RAGSettings(session_root=tmp_path / "sessions", chunk_size=500, chunk_overlap=50)
    return SessionRAGManager(settings=settings, embedder=KeywordEmbedder())


def test_session_upload_creates_metadata_and_faiss_index(tmp_path: Path) -> None:
    rag = manager(tmp_path)
    session_id = rag.create_session()

    result = rag.add_documents(session_id, [("paper_a.pdf", pdf_bytes("Alpha retrieval method."))])
    evidence = rag.retrieve(session_id, "alpha retrieval", top_k=1)
    session_dir = rag.session_path(session_id)

    assert result.chunk_count == 1
    assert (session_dir / "uploads").is_dir()
    assert (session_dir / "faiss.index").is_file()
    assert (session_dir / "metadata.json").is_file()
    assert evidence[0]["filename"] == "paper_a.pdf"
    assert evidence[0]["session_id"] == session_id
    assert evidence[0]["document_id"]


def test_session_isolation_never_crosses_document_indexes(tmp_path: Path) -> None:
    rag = manager(tmp_path)
    session_a, session_b = rag.create_session(), rag.create_session()
    rag.add_documents(session_a, [("alpha.pdf", pdf_bytes("Alpha retrieval only."))])
    rag.add_documents(session_b, [("beta.pdf", pdf_bytes("Beta monitoring only."))])

    results_a = rag.retrieve(session_a, "alpha retrieval", top_k=5)
    results_b = rag.retrieve(session_b, "beta monitoring", top_k=5)

    assert results_a and all(result["filename"] == "alpha.pdf" and result["session_id"] == session_a for result in results_a)
    assert results_b and all(result["filename"] == "beta.pdf" and result["session_id"] == session_b for result in results_b)


def test_multiple_uploads_share_only_their_own_session_index(tmp_path: Path) -> None:
    rag = manager(tmp_path)
    session_id = rag.create_session()
    rag.add_documents(session_id, [
        ("first.pdf", pdf_bytes("Alpha retrieval evidence.")),
        ("second.pdf", pdf_bytes("Beta monitoring evidence.")),
    ])

    results = rag.retrieve(session_id, "beta monitoring", top_k=1)

    assert results[0]["filename"] == "second.pdf"


def test_delete_session_removes_all_temporary_resources(tmp_path: Path) -> None:
    rag = manager(tmp_path)
    session_id = rag.create_session()
    rag.add_documents(session_id, [("paper.pdf", pdf_bytes("Alpha evidence."))])
    session_dir = rag.session_path(session_id)

    rag.delete_session(session_id)

    assert not session_dir.exists()
    try:
        rag.retrieve(session_id, "alpha")
    except KeyError:
        pass
    else:
        raise AssertionError("A deleted session must not be retrievable.")
