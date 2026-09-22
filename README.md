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
