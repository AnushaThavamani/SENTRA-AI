"""Thin application-facing integration layer for the Research Agent UI."""
from __future__ import annotations

from dataclasses import dataclass

from app.agents import ResearchAgent, ResearchResult
from app.rag import SessionIngestionResult, SessionRAGManager


@dataclass(frozen=True)
class WorkspaceStatus:
    session_id: str
    documents: list[str]
    chunk_count: int
    ready: bool


class ResearchWorkspace:
    """Coordinates UI actions without duplicating RAG or agent behavior."""

    def __init__(self, manager: SessionRAGManager | None = None) -> None:
        self._manager = manager or SessionRAGManager()
        self._agent = ResearchAgent(self._manager)

    def create_session(self) -> str:
        return self._manager.create_session()

    def status(self, session_id: str) -> WorkspaceStatus:
        summary = self._manager.session_summary(session_id)
        return WorkspaceStatus(
            session_id=session_id,
            documents=list(summary["documents"]),
            chunk_count=int(summary["chunk_count"]),
            ready=bool(summary["ready"]),
        )

    def ingest(self, session_id: str, files: list[tuple[str, bytes]]) -> SessionIngestionResult:
        if not files:
            raise ValueError("Please upload at least one PDF.")
        existing = set(self.status(session_id).documents)
        unique_files = [(name, content) for name, content in files if name not in existing]
        if not unique_files:
            raise ValueError("These documents are already indexed in this session.")
        return self._manager.add_documents(session_id, unique_files)

    def ask(self, session_id: str, query: str) -> ResearchResult:
        return self._agent.run(session_id=session_id, query=query)

    def delete_session(self, session_id: str) -> None:
        self._manager.delete_session(session_id)
