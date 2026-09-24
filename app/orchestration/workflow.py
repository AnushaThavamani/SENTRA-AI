"""Bounded corrective retrieval and reconsideration workflow."""
from __future__ import annotations

from app.agents.models import GuardrailResult, SpecificationResult, CodeGenerationInput
from app.agents.research_agent import ResearchAgent
from app.agents.specification_agent import SpecificationAgent
from app.guardrails.evidence_guardrail import EvidenceGuardrail


def run_research_to_spec_workflow(
    session_id: str,
    task_description: str,
    research_agent: ResearchAgent,
    spec_agent: SpecificationAgent,
    guardrail: EvidenceGuardrail,
    proposed_overrides: dict[str, str] | None = None
) -> tuple[CodeGenerationInput | None, SpecificationResult, GuardrailResult]:
    """
    Executes the bounded revision loop for ML Specification verification.
    
    1. Initially retrieves evidence across all ML domains.
    2. Generates the Specification.
    3. Runs the Active Evidence Guardrail.
    4. If the Guardrail detects missing or unsupported items, executes targeted
       corrective retrieval.
    5. Repeats up to Guardrail.max_attempts.
    6. If verified, outputs the strict CodeGenerationInput.
    """
    evidence_results = {}
    
    spec = None
    guardrail_result = None

    for attempt in range(1, guardrail.max_attempts + 1):
        if attempt == 1:
            # Broad sweep on the first attempt
            for domain in spec_agent.ML_DOMAINS:
                query = f"{task_description} {domain}"
                evidence_results[domain] = research_agent.run(session_id=session_id, query=query)
        else:
            # Corrective retrieval based on Guardrail feedback
            for req, action_query in guardrail_result.corrective_actions.items():
                evidence_results[req] = research_agent.run(session_id=session_id, query=action_query)
                
        # Generate Specification with current evidence pool
        spec = spec_agent.generate(
            session_id=session_id,
            task_description=task_description,
            evidence_results=evidence_results,
            proposed_overrides=proposed_overrides
        )
        
        # Verify Specification against evidence
        guardrail_result = guardrail.validate(spec, attempt_number=attempt)
        
        if guardrail_result.status in ("VERIFIED", "EVIDENCE_UNRESOLVED"):
            break

    if guardrail_result.status == "VERIFIED":
        # Create final valid contract
        # In a real setup, we might also pass `source_documents` 
        # from the research_workspace or evidence links.
        source_docs = list({
            item.evidence.source 
            for item in spec.items.values() 
            if item.evidence is not None
        })
        
        code_input = CodeGenerationInput(
            session_id=session_id,
            verified_specification=spec,
            source_documents=source_docs
        )
        return code_input, spec, guardrail_result

    return None, spec, guardrail_result

