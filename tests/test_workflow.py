"""Tests for the bounded ML specification revision loop."""
import pytest
from unittest.mock import Mock, call

from app.agents.models import GuardrailResult, SpecificationResult
from app.agents.research_agent import ResearchAgent, ResearchResult
from app.agents.specification_agent import SpecificationAgent
from app.guardrails.evidence_guardrail import EvidenceGuardrail
from app.orchestration.workflow import run_research_to_spec_workflow


@pytest.fixture
def spec_agent():
    return SpecificationAgent()


@pytest.fixture
def guardrail():
    return EvidenceGuardrail(max_attempts=3)


@pytest.fixture
def session_id():
    return "test-session-123"


def create_mock_research_agent(responses: dict):
    agent = Mock(spec=ResearchAgent)
    
    def side_effect(session_id: str, query: str, top_k: int = 3):
        # Allow exact match, or fallback to default
        if query in responses:
            return responses[query]
        return ResearchResult(
            session_id=session_id,
            query=query,
            answer="No sufficient evidence.",
            evidence=[],
            sufficient_evidence=False
        )
        
    agent.run.side_effect = side_effect
    return agent


def create_evidence(text: str, filename="test.pdf", chunk_id="chunk1"):
    return {
        "text": text,
        "source": filename,
        "filename": filename,
        "page": 1,
        "chunk_id": chunk_id,
        "distance": 0.1
    }


def test_fully_supported_specification(spec_agent, guardrail, session_id):
    """Test 1: Fully supported specification -> PASS"""
    # Mock finding evidence for every domain
    responses = {}
    for domain in spec_agent.ML_DOMAINS:
        query = f"Task {domain}"
        responses[query] = ResearchResult(
            session_id=session_id, query=query, answer=f"{domain} found",
            evidence=[create_evidence(f"{domain} found", chunk_id=f"chunk_{domain}")], sufficient_evidence=True
        )
        
    agent = create_mock_research_agent(responses)
    
    code_input, spec, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        spec_agent=spec_agent,
        guardrail=guardrail
    )
    
    assert gr.status == "VERIFIED"
    assert gr.passed is True
    assert code_input is not None
    assert gr.attempt_number == 1
    assert len(gr.missing_requirements) == 0
    assert len(gr.unsupported_requirements) == 0

    # spec is code_input.verified_specification (and we also get it as second return)

    # Test 9: Provenance verification
    assert spec.items["optimizer"].evidence.source == "test.pdf"
    assert spec.items["optimizer"].evidence.page == 1
    assert spec.items["optimizer"].evidence.chunk_id == "chunk_optimizer"


def test_missing_information(spec_agent, guardrail, session_id):
    """Test 2: Missing information -> REVISION_REQUIRED, and Test 5: Evidence remains unavailable"""
    responses = {}
    for domain in spec_agent.ML_DOMAINS:
        query = f"Task {domain}"
        if domain != "optimizer":
            responses[query] = ResearchResult(
                session_id, query, "found", [create_evidence("found")], True
            )
        else:
            # Missing optimizer
            responses[query] = ResearchResult(
                session_id, query, "Not found", [], False
            )
            # Make sure targeted query also fails
            targeted_query = "Find specific evidence for the ML optimizer used in the model."
            responses[targeted_query] = ResearchResult(
                session_id, targeted_query, "Not found", [], False
            )

    agent = create_mock_research_agent(responses)
    
    code_input, spec, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        spec_agent=spec_agent,
        guardrail=guardrail
    )
    
    assert gr.status == "EVIDENCE_UNRESOLVED"
    assert gr.passed is False
    assert code_input is None
    assert gr.attempt_number == 3
    assert "optimizer" in gr.missing_requirements


def test_hallucination_control(spec_agent, guardrail, session_id):
    """Test 3: Hallucination control - verify missing parameters do not result in hallucinated defaults."""
    responses = {}
    for domain in spec_agent.ML_DOMAINS:
        query = f"Task {domain}"
        responses[query] = ResearchResult(
            session_id, query, "found", [create_evidence("found")], True
        )
    # Simulate optimizer not found in evidence
    responses["Task optimizer"] = ResearchResult(
        session_id, "Task optimizer", "Not found", [], False
    )
    responses["Find specific evidence for the ML optimizer used in the model."] = ResearchResult(
        session_id, "Find specific evidence for the ML optimizer used in the model.", "Not found", [], False
    )

    agent = create_mock_research_agent(responses)
    
    # We pass a proposed override that hallucinated the optimizer
    overrides = {"optimizer": "Adam"}
    
    code_input, spec, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        spec_agent=spec_agent,
        guardrail=guardrail,
        proposed_overrides=overrides
    )
    
    assert gr.status == "EVIDENCE_UNRESOLVED"
    assert gr.passed is False
    assert code_input is None
    assert gr.attempt_number == 3
    assert "optimizer" in gr.unsupported_requirements
    assert spec.items["optimizer"].status == "UNSUPPORTED"
    assert spec.items["optimizer"].value == "Adam"


def test_corrective_retrieval_finds_evidence(spec_agent, guardrail, session_id):
    """Test 4: Corrective retrieval finds evidence"""
    responses = {}
    for domain in spec_agent.ML_DOMAINS:
        query = f"Task {domain}"
        if domain != "optimizer":
            responses[query] = ResearchResult(
                session_id, query, "found", [create_evidence("found")], True
            )
        else:
            responses[query] = ResearchResult(
                session_id, query, "Not found", [], False
            )
            
    # Targeted query succeeds
    targeted_query = "Find specific evidence for the ML optimizer used in the model."
    responses[targeted_query] = ResearchResult(
        session_id, targeted_query, "Adam found", [create_evidence("Adam found", chunk_id="chunk_2")], True
    )

    agent = create_mock_research_agent(responses)
    
    code_input, spec, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        spec_agent=spec_agent,
        guardrail=guardrail
    )
    
    assert gr.status == "VERIFIED"
    assert gr.passed is True
    assert code_input is not None
    assert gr.attempt_number == 2
    assert spec.items["optimizer"].status == "SUPPORTED"
    assert spec.items["optimizer"].value == "Adam found"
    assert spec.items["optimizer"].evidence.chunk_id == "chunk_2"


def test_no_infinite_loop(spec_agent, guardrail, session_id):
    """Test 6 & 7: No infinite loop, same unsupported value cannot magically pass."""
    # The agent returns insufficient evidence for everything.
    agent = create_mock_research_agent({})
    
    code_input, spec, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        spec_agent=spec_agent,
        guardrail=guardrail
    )
    
    # Assert terminated exactly at max attempts
    assert gr.attempt_number == 3
    assert gr.status == "EVIDENCE_UNRESOLVED"
    assert gr.passed is False


def test_session_isolation(spec_agent, guardrail):
    """Test 8: Session isolation (verify correct session id is passed)"""
    agent = Mock(spec=ResearchAgent)
    agent.run.return_value = ResearchResult("sess-1", "q", "ans", [], False)
    
    run_research_to_spec_workflow("sess-1", "Task", agent, spec_agent, guardrail)
    
    for call_args in agent.run.call_args_list:
        assert call_args.kwargs["session_id"] == "sess-1"


def test_integration_pipeline(tmp_path):
    """Test full integration from raw PDF bytes through to VERIFIED CodeGenerationInput."""
    from tests.test_research_agent import make_agent, pdf_bytes
    
    rag, research_agent = make_agent(tmp_path)
    session_id = rag.create_session()
    
    # Supply dummy content for all ML_DOMAINS to trigger "found" state
    content = "The alpha algorithm uses beta architecture. " \
              "inputs are datasets. outputs are predictions. " \
              "loss_function is MSE. optimizer is Adam. " \
              "learning_rate is 0.01. evaluation_metrics is accuracy."
              
    rag.add_documents(session_id, [("paper.pdf", pdf_bytes([content]))])
    
    spec_agent = SpecificationAgent()
    guardrail = EvidenceGuardrail(max_attempts=3)
    
    code_input, spec, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="alpha dataset methodology",
        research_agent=research_agent,
        spec_agent=spec_agent,
        guardrail=guardrail
    )
    
    assert gr.status == "VERIFIED"
    assert code_input is not None
    assert code_input.session_id == session_id
    assert "paper.pdf" in code_input.source_documents
    
    assert spec.items["optimizer"].status == "SUPPORTED"
    assert spec.items["learning_rate"].status == "SUPPORTED"
