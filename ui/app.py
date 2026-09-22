"""Professional Streamlit workspace for the existing SENTRA Research Agent."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

from app.services import ResearchWorkspace, WorkspaceStatus
from ui.components.evidence_panel import render_result
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from ui.components.upload_panel import render_upload_panel
from ui.styles.theme import inject_theme


@st.cache_resource
def workspace() -> ResearchWorkspace:
    """One application service; backend session IDs remain the isolation boundary."""
    return ResearchWorkspace()


def reset_ui_state() -> None:
    st.session_state.pop("sentra_session_id", None)
    st.session_state.pop("history", None)
    st.session_state.pop("confirm_delete", None)


def active_status(service: ResearchWorkspace) -> WorkspaceStatus | None:
    session_id = st.session_state.get("sentra_session_id")
    if not session_id:
        return None
    try:
        return service.status(session_id)
    except (KeyError, ValueError):
        reset_ui_state()
        st.warning("The previous research session is no longer available.")
        return None


def create_session(service: ResearchWorkspace) -> None:
    st.session_state.sentra_session_id = service.create_session()
    st.session_state.history = []
    st.session_state.confirm_delete = False
    st.success("New research session created. Upload PDF papers to build its knowledge base.")


def main() -> None:
    st.set_page_config(page_title="SENTRA AI · Research Workspace", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    inject_theme(st)
    service = workspace()
    status = active_status(service)
    render_header(st, bool(status))
    new_session, request_delete = render_sidebar(st, status)
    if new_session:
        if status:
            st.info("The current session remains available until you explicitly delete it.")
        create_session(service)
        st.rerun()
    if request_delete:
        st.session_state.confirm_delete = True

    status = active_status(service)
    if st.session_state.get("confirm_delete") and status:
        st.warning("Delete this temporary session and all of its uploaded PDFs, index, and evidence history?")
        confirm, cancel = st.columns(2)
        if confirm.button("Confirm delete", type="primary"):
            try:
                service.delete_session(status.session_id)
                reset_ui_state()
                st.success("Research session deleted.")
                st.rerun()
            except (KeyError, ValueError) as exc:
                st.error(f"Could not delete this session: {exc}")
        if cancel.button("Keep session"):
            st.session_state.confirm_delete = False
            st.rerun()

    if not status:
        st.markdown("## WELCOME TO SENTRA AI")
        st.markdown('<div class="card"><h3>Transform research papers into evidence-grounded research intelligence.</h3><p class="muted">Create a temporary research session. Your papers, FAISS index, and evidence remain isolated to that session.</p></div>', unsafe_allow_html=True)
        if st.button("Create Research Session", type="primary"):
            create_session(service)
            st.rerun()
        return

    left, right = st.columns([1, 1.75], gap="large")
    with left:
        files = render_upload_panel(st, disabled=False)
        if files:
            pending = [(file.name, file.getvalue()) for file in files]
            if st.button("Index uploaded documents", type="primary", use_container_width=True):
                try:
                    with st.spinner("Extracting text and building this session's knowledge base..."):
                        ingestion = service.ingest(status.session_id, pending)
                    if ingestion.problems:
                        st.warning("Some PDF content could not be extracted: " + "; ".join(problem.message for problem in ingestion.problems))
                    st.success(f"Knowledge Base READY — {ingestion.chunk_count} chunks indexed.")
                    st.rerun()
                except (KeyError, ValueError, RuntimeError) as exc:
                    st.error(f"The uploaded documents could not be processed: {exc}")
        st.markdown("---")
        st.markdown("#### SESSION-ISOLATED KNOWLEDGE")
        st.info(f"Only documents uploaded to this research session are available to the Research Agent.\n\nCurrent knowledge space: `{status.session_id}`")

    with right:
        st.markdown("### RESEARCH WORKSPACE")
        if not status.ready:
            st.markdown('<div class="card"><div class="label">YOUR RESEARCH SPACE IS EMPTY</div><h3>Upload one or more research papers to create your temporary knowledge base.</h3><p class="muted">Documents remain isolated within this research session.</p></div>', unsafe_allow_html=True)
            return
            
        qa_tab, spec_tab = st.tabs(["Q&A", "ML Specification"])
        
        with qa_tab:
            query = st.text_area("Ask Research Agent", placeholder="Ask something about your uploaded research papers...", height=105)
        st.caption("Examples: What methodology is proposed? · What datasets were used? · What are the main limitations?")
        if st.button("Ask Research Agent", type="primary"):
            if not query.strip():
                st.warning("Please enter a research question.")
            else:
                try:
                    with st.spinner("Research Agent is analyzing your evidence..."):
                        result = service.ask(status.session_id, query)
                    st.session_state.history = [{"query": query, "result": result}, *st.session_state.get("history", [])][:8]
                except (KeyError, ValueError, RuntimeError) as exc:
                    st.error(f"The Research Agent could not retrieve evidence: {exc}")
        history = st.session_state.get("history", [])
        if history:
            render_result(st, history[0]["result"])
            with st.expander("RECENT QUESTIONS"):
                for item in history:
                    st.caption(f"• {item['query']}")
                    
        with spec_tab:
            st.markdown("### Generate ML Specification")
            st.markdown("Extract a verified ML specification from the papers using the active evidence guardrail.")
            task_desc = st.text_area(
                "Target Task / Model Description", 
                placeholder="e.g. A Vision Transformer for image classification", 
                height=68
            )
            
            if st.button("Generate Specification", type="primary", use_container_width=True):
                if not task_desc.strip():
                    st.warning("Please provide a task description.")
                else:
                    with st.spinner("Executing bounded revision loop..."):
                        spec, gr = service.generate_ml_specification(status.session_id, task_desc)
                    st.session_state.last_spec = spec
                    st.session_state.last_guardrail = gr
            
            spec = st.session_state.get("last_spec")
            gr = st.session_state.get("last_guardrail")
            
            if spec and gr:
                st.markdown("---")
                if gr.passed:
                    st.success(f"Guardrail PASSED (Attempt {gr.attempt_number}) - Specification is fully supported by evidence.")
                else:
                    st.error(f"Guardrail FAILED (Attempt {gr.attempt_number}) - Revision loop exhausted.")
                    if gr.missing_requirements:
                        st.warning(f"Missing Requirements: {', '.join(gr.missing_requirements)}")
                    if gr.unsupported_requirements:
                        st.warning(f"Unsupported Requirements: {', '.join(gr.unsupported_requirements)}")
                
                st.markdown("#### Extracted Specification")
                for key, item in spec.items.items():
                    if item.status == "SUPPORTED":
                        st.success(f"**{item.name}**: {item.value}")
                        with st.expander("View Evidence"):
                            st.caption(f"Source: {item.evidence.source} (Page {item.evidence.page})")
                            st.markdown(f"> {item.evidence.evidence_text}")
                    elif item.status == "MISSING":
                        st.warning(f"**{item.name}**: MISSING")
                    else:
                        st.error(f"**{item.name}**: UNSUPPORTED ({item.value})")


if __name__ == "__main__":
    main()
