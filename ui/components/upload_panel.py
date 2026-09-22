from __future__ import annotations


def render_upload_panel(st: object, disabled: bool) -> list[object] | None:
    st.markdown("#### UPLOAD RESEARCH PAPERS")
    st.caption("PDF · Multiple files supported · stored only in this research session")
    return st.file_uploader("Drag and drop PDF papers here", type=["pdf"], accept_multiple_files=True, disabled=disabled, label_visibility="visible")
