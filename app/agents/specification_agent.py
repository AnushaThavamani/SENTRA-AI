"""Specification Agent for transforming research evidence into ML specifications."""
from __future__ import annotations

from app.agents.models import EvidenceLink, SpecificationRequirement, SpecificationResult
from app.agents.research_agent import ResearchResult


class SpecificationAgent:
    """Transforms retrieved research evidence into structured ML specifications."""

    ML_DOMAINS = [
        "algorithm",
        "architecture",
        "inputs",
        "outputs",
        "loss_function",
        "optimizer",
        "learning_rate",
        "evaluation_metrics",
    ]

    def generate(
        self,
        session_id: str,
        task_description: str,
        evidence_results: dict[str, ResearchResult],
        proposed_overrides: dict[str, str] | None = None
    ) -> SpecificationResult:
        """
        Generate a SpecificationResult.
        
        Args:
            session_id: The ID of the research session.
            task_description: The original task description.
            evidence_results: A mapping of domain name to ResearchResult.
            proposed_overrides: Optional mapping to simulate a code generator attempting
                                to hallucinate/invent values without evidence.
        """
        items = {}
        proposed_overrides = proposed_overrides or {}

        for domain in self.ML_DOMAINS:
            result = evidence_results.get(domain)
            proposed_val = proposed_overrides.get(domain)

            if proposed_val is not None:
                # If there's a proposed value but no sufficient evidence to support it, it's UNSUPPORTED.
                # In a real LLM implementation, the Guardrail would check if the evidence actually entails the proposed value.
                if not result or not result.sufficient_evidence or not result.evidence:
                    items[domain] = SpecificationRequirement(
                        name=domain,
                        status="UNSUPPORTED",
                        value=proposed_val
                    )
                else:
                    # If we have evidence, we assume it's SUPPORTED for this deterministic mock.
                    best_ev = result.evidence[0]
                    items[domain] = SpecificationRequirement(
                        name=domain,
                        status="SUPPORTED",
                        value=proposed_val,
                        evidence=EvidenceLink(
                            source=str(best_ev.get("source", best_ev.get("filename", ""))),
                            page=int(best_ev.get("page", 0)),
                            chunk_id=str(best_ev.get("chunk_id", "")),
                            evidence_text=str(best_ev.get("text", ""))
                        )
                    )
            else:
                # No proposed value, so we derive entirely from evidence.
                if not result or not result.sufficient_evidence or not result.evidence:
                    items[domain] = SpecificationRequirement(
                        name=domain,
                        status="MISSING"
                    )
                else:
                    best_ev = result.evidence[0]
                    items[domain] = SpecificationRequirement(
                        name=domain,
                        status="SUPPORTED",
                        value=result.answer,
                        evidence=EvidenceLink(
                            source=str(best_ev.get("source", best_ev.get("filename", ""))),
                            page=int(best_ev.get("page", 0)),
                            chunk_id=str(best_ev.get("chunk_id", "")),
                            evidence_text=str(best_ev.get("text", ""))
                        )
                    )

        return SpecificationResult(
            session_id=session_id,
            task_description=task_description,
            items=items
        )
