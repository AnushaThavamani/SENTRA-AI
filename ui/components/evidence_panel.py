from __future__ import annotations

from app.agents import ResearchResult


def render_result(st: object, result: ResearchResult) -> None:
    badge = "Evidence grounded" if result.sufficient_evidence else "Insufficient evidence"
    st.markdown("#### RESEARCH AGENT")
    st.markdown(f'<div class="card answer-card"><div class="label">{badge.upper()}</div></div>', unsafe_allow_html=True)
    st.write(result.answer)
    if not result.evidence:
        return
    st.markdown("#### EVIDENCE USED")
    for item in result.evidence:
        title = f"{item['source']} — Page {item['page']} · Score {item['score']:.2f}"
        with st.expander(title):
            columns = st.columns(3)
            columns[0].metric("SOURCE", str(item["source"]))
            columns[1].metric("PAGE", str(item["page"]))
            columns[2].metric("RELEVANCE", f"{item['score']:.2f}")
            st.write(item["text"])
