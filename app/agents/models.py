"""Shared models for Specification and Orchestration Agents."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvidenceLink:
    source: str
    page: int
    chunk_id: str
    evidence_text: str


@dataclass(frozen=True)
class SpecificationRequirement:
    name: str
    status: str  # "SUPPORTED", "MISSING_EVIDENCE", "UNSUPPORTED"
    value: str | None = None
    evidence: EvidenceLink | None = None


@dataclass(frozen=True)
class SpecificationResult:
    session_id: str
    task_description: str
    items: dict[str, SpecificationRequirement] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardrailResult:
    status: str  # "DRAFT", "REQUIRES_RETRIEVAL", "REQUIRES_REVISION", "VERIFIED", "EVIDENCE_UNRESOLVED"
    attempt_number: int
    max_attempts: int
    unsupported_requirements: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    corrective_actions: dict[str, str] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        return self.status == "VERIFIED"


@dataclass(frozen=True)
class CodeGenerationInput:
    session_id: str
    verified_specification: SpecificationResult
    source_documents: list[str] = field(default_factory=list)