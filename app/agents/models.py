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
    status: str  # "SUPPORTED", "MISSING", "UNSUPPORTED"
    value: str | None = None
    evidence: EvidenceLink | None = None


@dataclass(frozen=True)
class SpecificationResult:
    session_id: str
    task_description: str
    items: dict[str, SpecificationRequirement] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    revision_required: bool
    attempt_number: int
    max_attempts: int
    unsupported_requirements: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    corrective_actions: dict[str, str] = field(default_factory=dict)