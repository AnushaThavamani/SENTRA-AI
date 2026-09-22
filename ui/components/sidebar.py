from __future__ import annotations

from app.services import WorkspaceStatus


def render_sidebar(st: object, status: WorkspaceStatus | None) -> tuple[bool, bool]:
    with st.sidebar:
        st.markdown("### RESEARCH SESSION")
        if status:
            st.markdown('<div class="label">CURRENT SESSION</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="session-id">{status.session_id}</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("#### KNOWLEDGE BASE")
            state = "● READY" if status.ready else "● EMPTY"
            css = "ready" if status.ready else "empty"
            st.markdown(f'<div class="{css}">{state}</div>', unsafe_allow_html=True)
            one, two = st.columns(2)
            one.metric("DOCUMENTS", len(status.documents))
            two.metric("CHUNKS", status.chunk_count)
            if status.documents:
                st.markdown("#### DOCUMENTS")
                for document in status.documents:
                    st.caption(f"✓ {document}")
            st.markdown("---")
        new_session = st.button("+ New Research Session", use_container_width=True)
        delete_session = bool(status) and st.button("Delete Session", type="secondary", use_container_width=True)
        st.markdown("---")
        st.markdown("#### PIPELINE")
        st.caption("✓ Research")
        st.caption("○ Specification · future")
        st.caption("○ Verification · future")
        st.caption("○ Code Generation · future")
        return new_session, delete_session
