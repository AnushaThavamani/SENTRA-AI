"""Evidence-only research agent backed by one uploaded-document session."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.session_manager import SessionRAGManager


_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "according", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "paper", "proposed",
    "the", "this", "to", "was", "what", "were", "which", "with",
})


@dataclass(frozen=True)
class ResearchResult:
    """Stable hand-off shape for later agents (such as Specification Agent)."""

    session_id: str
    query: str
    answer: str
    evidence: list[dict[str, object]]
    sufficient_evidence: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "answer": self.answer,
            "evidence": self.evidence,
            "sufficient_evidence": self.sufficient_evidence,
        }


class ResearchAgent:
    """Answer only with evidence retrieved from the supplied session ID.

    There is intentionally no filesystem or global-corpus access here: all
    document access occurs through :class:`SessionRAGManager.retrieve`.
    In the absence of an application LLM, answers are extractive summaries of
    the best matching evidence, which keeps them auditable and non-hallucinatory.
    """

    def __init__(self, rag_manager: SessionRAGManager) -> None:
        self._rag_manager = rag_manager

    def run(self, *, session_id: str, query: str, top_k: int = 3) -> ResearchResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        evidence = self._rag_manager.retrieve(session_id, query, top_k)
        keywords = self._keywords(query)
        supported = [item for item in evidence if self._supports_query(str(item["text"]), keywords)]
        if not supported:
            return ResearchResult(
                session_id=session_id,
                query=query,
                answer="The uploaded documents do not provide sufficient evidence to answer this question.",
                evidence=[],
                sufficient_evidence=False,
            )

        # Preserve retrieval order and provenance. The answer is intentionally
        # extractive rather than inventing a conclusion beyond the document.
        best = supported[0]
        excerpt = self._best_sentence(str(best["text"]), keywords)
        answer = f"Based on {best['source']}, page {best['page']}: {excerpt}"
        return ResearchResult(session_id, query, answer, supported, True)

    @staticmethod
    def _keywords(query: str) -> set[str]:
        return {word for word in re.findall(r"[a-z0-9]+", query.lower())
                if len(word) > 2 and word not in _STOP_WORDS}

    @staticmethod
    def _supports_query(text: str, keywords: set[str]) -> bool:
        # A query made only of stop words should never be treated as evidence.
        if not keywords:
            return False
        terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        return bool(keywords & terms)

    @staticmethod
    def _best_sentence(text: str, keywords: set[str]) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        matching = [sentence for sentence in sentences if ResearchAgent._supports_query(sentence, keywords)]
        return (matching[0] if matching else text).strip()
