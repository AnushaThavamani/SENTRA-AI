from __future__ import annotations

from pathlib import Path
import fitz
import numpy as np
import pytest

from app.rag.chunker import PageChunker
from app.rag.models import ExtractedPage
from app.rag.pdf_processor import PDFProcessor
from app.rag.retriever import Retriever


class KeywordEmbedder:
    """Deterministic test fixture; production uses SentenceTransformerEmbedder."""
    vocabulary = ("alpha", "beta", "retrieval", "monitoring")

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[text.lower().split().count(word) for word in self.vocabulary] for text in texts], dtype=np.float32)


def make_pdf(path: Path, pages: list[str]) -> None:
    document = fitz.open()
    for content in pages:
        page = document.new_page()
        page.insert_text((72, 72), content)
    document.save(path)
    document.close()


def build_retriever(tmp_path: Path) -> Retriever:
    alpha, beta = tmp_path / "alpha.pdf", tmp_path / "beta.pdf"
    make_pdf(alpha, ["Alpha retrieval method uses vector search."])
    make_pdf(beta, ["Beta monitoring detects runtime risk."])
    report = PDFProcessor().extract_files([alpha, beta])
    return Retriever.from_extraction_report(report, KeywordEmbedder())


def test_pdf_extraction_preserves_page_text_and_document(tmp_path: Path) -> None:
    source = tmp_path / "paper_03.pdf"
    make_pdf(source, ["First page evidence.", "Second page evidence."])
    report = PDFProcessor().extract_files([source])
    assert not report.problems
    assert [(page.document, page.page) for page in report.pages] == [("paper_03.pdf", 1), ("paper_03.pdf", 2)]
    assert "Second page evidence" in report.pages[1].text


def test_chunking_preserves_requested_provenance() -> None:
    chunks = PageChunker(chunk_size=3, overlap=1).chunk_pages([ExtractedPage("paper_03.pdf", 14, "one two three four five")])
    assert [chunk.chunk_id for chunk in chunks] == ["paper_03_p14_c01", "paper_03_p14_c02", "paper_03_p14_c03"]
    assert all(chunk.document == "paper_03.pdf" and chunk.page == 14 for chunk in chunks)


def test_faiss_index_and_retrieval_return_provenance(tmp_path: Path) -> None:
    retriever = build_retriever(tmp_path)
    results = retriever.retrieve("alpha retrieval", top_k=1)
    assert retriever.size == 2
    assert results[0]["document"] == "alpha.pdf"
    assert results[0]["page"] == 1
    assert results[0]["chunk_id"] == "alpha_p1_c01"
    assert "distance" in results[0]


def test_multiple_documents_return_relevant_document(tmp_path: Path) -> None:
    results = build_retriever(tmp_path).retrieve("runtime monitoring", top_k=1)
    assert results[0]["document"] == "beta.pdf"
    assert "runtime risk" in str(results[0]["text"])


def test_missing_and_non_extractable_inputs_are_reported(tmp_path: Path) -> None:
    blank = tmp_path / "blank.pdf"
    make_pdf(blank, [""])
    report = PDFProcessor().extract_files([tmp_path / "missing.pdf", blank])
    assert not report.has_text
    assert len(report.problems) == 2
    with pytest.raises(ValueError, match="Cannot build a RAG corpus"):
        Retriever.from_extraction_report(report, KeywordEmbedder())
