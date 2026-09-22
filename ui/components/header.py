from __future__ import annotations


def render_header(st: object, active: bool) -> None:
    status = "● RESEARCH SESSION ACTIVE" if active else "○ NO ACTIVE SESSION"
    with st.container():
        left, right = st.columns([4, 2])
        with left:
            st.markdown('<div class="sentra-header"><div class="brand">SENTRA AI</div><div class="subtitle">Multi-Agent Research-to-Code Verification System</div></div>', unsafe_allow_html=True)
        with right:
            st.markdown(f'<div class="status">{status}</div>', unsafe_allow_html=True)
