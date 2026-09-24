# Sentra AI

**Project title:** Multi-Agent Runtime Monitoring and Risk Management System

## Purpose

Sentra AI will support evidence-grounded implementation workflows: research evidence informs an implementation specification, the specification is verified, a human approves it, and later stages generate and test code.

## Current status

Step 2 is complete: a session-scoped RAG evidence layer can extract selectable PDF text, create page-aware overlapping chunks, embed them with `all-MiniLM-L6-v2`, create an in-memory FAISS index, and retrieve evidence with document/page/chunk provenance. No agents, guardrails, code generation, UI, database, or authentication have been implemented.

## Planned architecture

1. Research Agent — retrieves evidence from user-uploaded research PDFs through an existing RAG implementation.
2. Specification Builder — creates a structured implementation specification from the request and retrieved evidence.
3. Specification Guardrail — verifies that specification claims are supported by the retrieved evidence.
4. Human Approval Gate — allows review and approval of verified specifications.
5. Code Generator — produces code from approved specifications.
6. Testing Agent — tests generated code and reports runtime or output issues.
7. Orchestration — coordinates the end-to-end workflow.

The legacy RAG implementation was inspected and its PDF extraction, 500-word chunking with 50-word overlap, `all-MiniLM-L6-v2` embeddings, and FAISS L2 retrieval design were refactored into `app/rag/`. The legacy project was not modified. Its development PDFs are not part of this application; each future upload/session builds a separate in-memory corpus.

## Step 2B — Session-Specific RAG

User-uploaded PDFs are stored temporarily under `data/sessions/<session_id>/uploads/`. Sentra AI extracts selectable text page by page, creates provenance-preserving chunks, embeds them, and stores each session's vectors in its own FAISS index with separate metadata. Retrieval is restricted to the requested session ID, and `delete_session(session_id)` removes that session's uploads, index, and metadata. This is a retrieval pipeline built from uploaded documents; no model training occurs.

## P1 - Research Agent CLI demo

P1 adds an evidence-only `ResearchAgent`. It takes a required session ID and a question, retrieves only from that session, returns structured answer/evidence/source/page/score data, and reports insufficient evidence instead of guessing.

Install `requirements.txt`, then run the temporary-session demo against selectable-text PDFs:

```powershell
python main.py --demo .\paper1.pdf .\paper2.pdf --question "What methodology is proposed?"
```

The command creates a session, ingests the files, prints answer and provenance, and deletes the session before exiting. The first production run may download `sentence-transformers/all-MiniLM-L6-v2`.

## P2 - Streamlit UI

P2 introduces a professional, captivating Streamlit interface for the existing Research Agent, visually establishing the first stage of the future pipeline.

### Running the UI

You can launch the Streamlit UI with the following command:

```powershell
streamlit run ui/app.py
```

### Research workflow

The Streamlit UI supports the following interaction flow:

1. **Create session**: Initialize a new, isolated research space.
2. **Upload PDFs**: Drag and drop your research documents into the workspace.
3. **Session RAG**: Documents are parsed, chunked, embedded, and indexed for this specific session.
4. **Research Agent**: Query your knowledge base with specific research questions.
5. **Evidence-backed answer**: Receive a response grounded in the uploaded text, displaying specific source files and page numbers as evidence.

### Session isolation

Each research session has its own temporary, isolated knowledge base. Only documents uploaded to a specific research session are available to the Research Agent within that session. Deleting the session clears its uploads, index, and evidence history entirely.

### Current Limitations & Future Work

*   **Session State Drift:** If the Streamlit UI tab is refreshed or closed without explicitly deleting the session, the browser loses track of the `session_id`. To prevent orphaned data from accumulating on disk, the system now implements an automatic Time-To-Live (TTL) cleanup. Any session older than 24 hours is automatically swept and deleted when new sessions are created.
*   **Storage Backend:** The system currently relies on local filesystem storage (`data/sessions/`). This is ideal for the local Python execution of this MCA project demo. However, if deployed to a hosted cloud environment with multiple concurrent users or scaled horizontally, the storage layer must be abstracted to use a cloud object store (e.g., AWS S3 or Google Cloud Storage) rather than local directories.

## P3 - ML Specification Agent & Active Evidence Guardrail

P3 adds the ML Specification Agent and Active Evidence Guardrail components, transitioning the project focus explicitly to **machine-learning research papers**.

### ML-Paper Domain Scope
The system is intentionally constrained to generating specifications grounded *only* in ML research (e.g. architectures, optimizers, learning rates, evaluation metrics). It avoids generic or arbitrary software domains.

### Active Evidence Guardrail
Before any downstream code generation occurs, the generated ML specification is subjected to an **Active Evidence Guardrail**. The guardrail determines if each requirement is supported by actual evidence retrieved from the user's uploaded paper(s) in the active session.

### Bounded Revision & Hallucination Control
If the Guardrail detects missing or unsupported items (such as hallucinated ML defaults like `optimizer = Adam` when not specified in the text), the system executes targeted corrective retrieval against the session's RAG index.
- **Targeted Evidence Retrieval**: Broad queries are replaced with targeted searches for the missing requirement.
- **Bounded Reconsideration**: The Specification Agent revises the spec and the Guardrail re-checks it. This loop has a hard limit (e.g., `MAX_ATTEMPTS = 3`).
- **Hallucination-Control Principle**: The system will not silently insert common ML assumptions into the final specification.

### Terminal Unresolved State
If evidence remains unavailable after the bounded attempts, the requirement is explicitly marked as `UNSUPPORTED` or `MISSING`, and the guardrail fails (`EVIDENCE_UNRESOLVED`). The UI surfaces this failure cleanly instead of allowing a code generator to invent missing pieces.

### Session Isolation
All evidence retrieval for the ML Specification Agent and Evidence Guardrail strictly uses the `session_id`. No global knowledge base is accessed, preventing contamination between distinct user research sessions.

## P4 - Verified ML Specification Pipeline and Code Generation Handoff

P4 solidifies the boundary between the research/verification stage and the future code generation stage. The objective is to produce a strict, verified contract that prevents the downstream Coding Agent from receiving unverified or hallucinated ML parameters.

### `VerifiedMLSpecification` Contract
The pipeline output is wrapped in a final `CodeGenerationInput` data contract. This object strictly contains:
- The `session_id` used for grounding.
- The `VerifiedMLSpecification` containing only `SUPPORTED` claims with attached exact `EvidenceLink` metadata.
- An aggregation of `source_documents` that contributed to the final specification.

### Integration and Guardrail States
The Evidence Guardrail has been formalized with explicit status enumerations (`VERIFIED`, `EVIDENCE_UNRESOLVED`, `REQUIRES_RETRIEVAL`) to prevent boolean ambiguity. The central orchestrator `run_research_to_spec_workflow` only returns a valid `CodeGenerationInput` contract if the Guardrail reaches `VERIFIED` status.

### UI Progression Tracker
The Streamlit interface now includes a pipeline progression tracker, making the workflow visually explicit:
- Research Agent ✓
- ML Specification ✓
- Evidence Verification ✓ (or ✗ EVIDENCE UNRESOLVED)
- Code Generation ○ (Next Phase)
- Testing ○ (Next Phase)
