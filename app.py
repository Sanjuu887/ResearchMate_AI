"""ResearchMate AI Streamlit interface."""

from datetime import datetime, timezone

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from config import AppConfig, load_config
from researchmate import link_citations, quick_greeting_response
from report_exports import make_report_pdf
from research_workflow import (
    citation_number as _citation_number,
    make_json,
    make_markdown,
    run_research_direct,
)


def _load_app_config() -> AppConfig:
    try:
        return load_config(st.secrets)
    except StreamlitSecretNotFoundError:
        return load_config()


APP_CONFIG = _load_app_config()

st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="auto",
)

RESEARCH_MODES = ("Quick Answer", "Standard Research", "Deep Research")

st.markdown(
    """
    <style>
    :root {
        --rm-ink: #18243a;
        --rm-muted: #68758a;
        --rm-line: #e3e8f0;
        --rm-accent: #6257d9;
        --rm-accent-dark: #4e44c2;
        --rm-paper: #ffffff;
        --rm-canvas: #f5f7fb;
    }
    .stApp {background: var(--rm-canvas); color: var(--rm-ink);}
    [data-testid="stHeader"], [data-testid="stBottom"] {
        background:var(--rm-canvas) !important;
    }
    [data-testid="stBottom"] > div {background:var(--rm-canvas) !important;}
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        padding: 2.3rem 2.2rem 4.5rem;
    }
    [data-testid="stSidebar"] {
        background: #172238;
        border-right: 1px solid #26344d;
    }
    [data-testid="stSidebar"] * {color: #e9edf7;}
    [data-testid="stSidebar"] hr {border-color: #34415a;}
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background: #202d45;
        border-color: #3a4862;
        color: #f5f7fb;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background:#202d45; border:1px solid #3a4862; color:#eef2fa;
        min-height:2.55rem; text-align:left;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background:#2a3955; border-color:#657495; color:#fff;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
        background:#5d52d3; border-color:#5d52d3; color:#fff;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display:block; color:#e9edf7; background:#202d45; border:1px solid #3a4862;
        border-radius:9px; padding:.45rem .65rem; margin:.25rem 0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background:#30385f; border-color:#8278e4;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color:#f1f3fa; margin:0;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background:#1d2a41; border:1px solid #3a4862; border-radius:10px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] div {
        color:#e9edf7;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#b7c2d5;}
    h1, h2, h3 {color: var(--rm-ink); letter-spacing: -.035em;}
    h1 {font-size: clamp(2.25rem, 4vw, 3.2rem) !important; line-height: 1.12 !important;}
    h2 {font-size: 1.55rem !important;}
    p, li {line-height: 1.78;}
    .rm-brand {display:flex; align-items:center; gap:.78rem; margin:.3rem 0 1.3rem;}
    .rm-logo {
        display:grid; place-items:center; width:2.45rem; height:2.45rem;
        flex:0 0 2.45rem; border-radius:.78rem;
        background:linear-gradient(145deg,#8177f2,#5146c7);
        box-shadow:0 5px 14px rgba(22,17,90,.24);
    }
    .rm-brand-name {font-size:1.12rem; font-weight:750; letter-spacing:-.04em; color:#f7f8fc;}
    .rm-brand-sub {font-size:.75rem; color:#aeb9cd; margin-top:.12rem; letter-spacing:.025em;}
    .rm-brand-main .rm-brand-name {color:var(--rm-ink);}
    .rm-brand-main .rm-brand-sub {color:var(--rm-muted);}
    .rm-brand-main {gap:1rem; margin:.2rem 0 2rem;}
    .rm-brand-main .rm-logo {width:3.5rem; height:3.5rem; flex-basis:3.5rem; border-radius:1rem;}
    .rm-brand-main .rm-logo svg {width:31px; height:31px;}
    .rm-brand-main .rm-brand-name {font-size:1.7rem; line-height:1.1;}
    .rm-brand-main .rm-brand-sub {font-size:.84rem; margin-top:.22rem;}
    .rm-eyebrow {
        color:var(--rm-accent); font-size:.76rem; font-weight:700;
        letter-spacing:.12em; text-transform:uppercase; margin-bottom:.8rem;
    }
    .rm-hero {padding:.35rem 0 1.3rem;}
    .rm-subtitle {max-width:620px; color:var(--rm-muted); font-size:1.04rem; line-height:1.68; margin:.7rem 0 1.5rem;}
    .rm-prompt-label {font-size:.88rem; font-weight:650; color:var(--rm-ink); margin:.35rem 0 .65rem;}
    [data-testid="stForm"] {
        background:var(--rm-paper); border:1px solid var(--rm-line);
        border-radius:18px; padding:1.35rem 1.45rem 1.1rem;
        box-shadow:0 12px 36px rgba(28,42,69,.055);
    }
    [data-testid="stTextArea"] label, [data-testid="stRadio"] label {color:var(--rm-ink); font-weight:600;}
    [data-testid="stTextArea"] textarea {
        background:#fbfcfe; border:1px solid #dfe5ee; border-radius:12px;
        color:var(--rm-ink); line-height:1.6; padding:.9rem 1rem;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color:var(--rm-accent); box-shadow:0 0 0 2px rgba(98,87,217,.14);
    }
    [data-testid="stRadio"] [role="radiogroup"] {gap:.65rem;}
    [data-testid="stRadio"] label {
        border:1px solid var(--rm-line); background:#fbfcfe; border-radius:10px;
        padding:.55rem .85rem; margin:0;
    }
    [data-testid="stRadio"] label:has(input:checked) {
        border-color:#a8a1f3; background:#f2f0ff;
    }
    .stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a {
        border-radius:10px; font-weight:600; transition:background-color .15s ease, border-color .15s ease;
    }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
    [data-testid="stLinkButton"] a:focus-visible {
        outline:3px solid #aaa4f4; outline-offset:2px;
    }
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primaryFormSubmit"] {
        background:#5146c7 !important; border-color:#5146c7 !important; color:#fff !important;
        min-height:2.9rem;
    }
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background:#4339b5 !important; border-color:#4339b5 !important;
    }
    [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div:first-child {
        background:#6257d9 !important; border-color:#6257d9 !important;
    }
    .stButton > button[kind="secondary"] {
        color:var(--rm-ink); background:#fff; border-color:var(--rm-line);
    }
    .rm-example button {text-align:left !important; white-space:normal !important;}
    .rm-result-head {
        border-bottom:1px solid var(--rm-line); padding:0 0 1.2rem; margin:.6rem 0 1.25rem;
    }
    .rm-meta {display:flex; flex-wrap:wrap; gap:.55rem; margin:.85rem 0 0;}
    .rm-pill {
        background:#eeeffb; color:#463e9f; border:1px solid #e2e1f5;
        border-radius:999px; padding:.28rem .68rem; font-size:.78rem; font-weight:600;
    }
    .rm-answer {
        background:#fff; border:1px solid var(--rm-line); border-radius:15px;
        padding:1.35rem 1.55rem; box-shadow:0 8px 24px rgba(28,42,69,.035);
    }
    .rm-answer p {max-width:78ch; color:#28364b;}
    .rm-answer h1, .rm-answer h2, .rm-answer h3 {margin-top:1.4em;}
    .rm-answer a {color:var(--rm-accent-dark); text-decoration-thickness:1px; text-underline-offset:2px;}
    div[data-testid="stExpander"] {border:1px solid var(--rm-line); border-radius:12px; background:#fff;}
    [data-testid="stStatusWidget"] {border-radius:12px; border:1px solid var(--rm-line); background:#fff;}
    [data-testid="stAlert"] {border-radius:12px;}
    [data-testid="stChatMessage"] {max-width:760px; margin-left:auto; margin-right:auto;}
    [data-testid="stChatInput"] {max-width:760px; margin-left:auto; margin-right:auto;}
    [data-testid="stChatMessage"] p {line-height:1.68;}
    [data-testid="stChatInput"] > div {
        background:#fff !important; border:1px solid #dfe5ee !important;
        border-radius:14px !important; box-shadow:0 5px 18px rgba(28,42,69,.06);
    }
    [data-testid="stChatInput"] textarea {
        background:#fff !important; color:var(--rm-ink) !important;
        caret-color:var(--rm-accent); line-height:1.5;
    }
    [data-testid="stChatInput"] textarea::placeholder {color:#76839a !important;}
    [data-testid="stChatInput"] button {
        background:#5146c7 !important; color:#fff !important; border-radius:9px !important;
    }
    [data-testid="stChatInput"] button:disabled {
        background:#eceafc !important; color:#7770b9 !important;
    }
    .rm-footer {color:var(--rm-muted); font-size:.83rem; border-top:1px solid var(--rm-line); margin-top:2.2rem; padding-top:1rem;}
    @media (max-width: 700px) {
        [data-testid="stAppViewContainer"] .main .block-container {padding:1.4rem 1rem 2.5rem;}
        [data-testid="stForm"] {padding:1rem;}
        .rm-answer {padding:1rem;}
        .rm-brand-main {margin-bottom:1.5rem;}
        .rm-brand-main .rm-brand-name {font-size:1.45rem;}
        [data-testid="stChatMessage"], [data-testid="stChatInput"] {max-width:100%;}
        [data-testid="stRadio"] [role="radiogroup"] {flex-wrap:wrap;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "research_result" not in st.session_state:
    st.session_state.research_result = None
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "research_history" not in st.session_state:
    st.session_state.research_history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "research_mode" not in st.session_state:
    st.session_state.research_mode = "Standard Research"
if "quick_messages" not in st.session_state:
    st.session_state.quick_messages = []


def render_brand(main: bool = False) -> None:
    brand_class = "rm-brand rm-brand-main" if main else "rm-brand"
    st.markdown(
        f"""
        <div class="{brand_class}">
          <div class="rm-logo" aria-hidden="true">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
                 xmlns="http://www.w3.org/2000/svg">
              <circle cx="10.6" cy="10.6" r="6.2" stroke="white" stroke-width="1.8"/>
              <path d="M15.2 15.2L20 20M8 10.6h5.2M10.6 8v5.2"
                    stroke="white" stroke-width="1.6" stroke-linecap="round"/>
            </svg>
          </div>
          <div>
            <div class="rm-brand-name">ResearchMate AI</div>
            <div class="rm-brand-sub">Evidence, made useful</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _restore_report(index: int) -> None:
    entry = st.session_state.research_history[index]
    st.session_state.research_result = entry["result"]
    st.session_state.research_mode = entry.get("mode", "Standard Research")
    st.session_state.last_question = entry["question"]
    st.session_state.question_input = entry["question"]


def _store_report(question: str, depth: str, result: dict) -> dict:
    result["question"] = question
    result["depth"] = depth
    result["markdown"] = make_markdown(question, result)
    result["json"] = make_json(question, result, result["markdown"])
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    st.session_state.research_result = result
    st.session_state.last_question = question
    st.session_state.research_history = [
        entry for entry in st.session_state.research_history
        if entry["question"] != question
    ]
    st.session_state.research_history.insert(
        0, {"question": question, "mode": depth, "result": result}
    )
    st.session_state.research_history = st.session_state.research_history[:5]
    return result


with st.sidebar:
    render_brand()
    st.caption("A focused workspace for questions worth investigating.")
    st.divider()

    if st.button("＋  New research", use_container_width=True, type="primary"):
        st.session_state.research_result = None
        st.session_state.last_question = ""
        st.session_state.question_input = ""
        if st.session_state.research_mode == "Quick Answer":
            st.session_state.quick_messages = []
        st.rerun()

    st.radio(
        "Research mode",
        options=RESEARCH_MODES,
        key="research_mode",
        help=(
            "Quick Answer is chat, Standard Research is a concise investigation, "
            "and Deep Research creates a comprehensive report."
        ),
    )

    if (
        st.session_state.research_mode != "Quick Answer"
        and st.session_state.research_history
    ):
        st.markdown("#### Recent reports")
        for index, entry in enumerate(st.session_state.research_history[:5]):
            label = entry["question"].strip()
            if len(label) > 42:
                label = label[:39].rstrip() + "…"
            st.button(
                label,
                key=f"history_{index}",
                use_container_width=True,
                help=f"{entry.get('mode', 'Research')}: {entry['question']}",
                on_click=_restore_report,
                args=(index,),
            )

    with st.expander("Settings and workflow"):
        st.caption(
            "Planning → search → page extraction → evidence analysis → answer. "
            "The research workflow runs inside this app; no MCP server is needed."
        )
        st.caption("Fetched webpages are untrusted evidence, not instructions.")
        st.caption(
            "Gemini: configured"
            if APP_CONFIG.gemini_api_key
            else "Gemini: missing GEMINI_API_KEY"
        )
        st.caption(
            "Tavily: configured"
            if APP_CONFIG.tavily_api_key
            else "Tavily: missing TAVILY_API_KEY"
        )
        st.caption("Credential values are never displayed.")

mode = st.session_state.research_mode
if mode == "Quick Answer":
    render_brand(main=True)
    st.markdown(
        '<div class="rm-eyebrow">Quick Answer · Conversation</div>',
        unsafe_allow_html=True,
    )
    st.title("Ask a question. Keep exploring.")
    st.markdown(
        '<div class="rm-subtitle">Short, natural answers for everyday questions. '
        'When a question needs current evidence, ResearchMate searches and links its sources.</div>',
        unsafe_allow_html=True,
    )
    messages = st.session_state.quick_messages
    if not messages:
        st.info(
            "Start with a question or explanation. Ask follow-ups in this conversation; "
            "switch to Standard or Deep Research for structured reports."
        )
    for message in messages:
        avatar = (
            ":material/person:"
            if message["role"] == "user"
            else ":material/travel_explore:"
        )
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                st.markdown(
                    link_citations(message["content"], message.get("sources", []))
                )
                if message.get("sources"):
                    with st.expander(f"Sources used ({len(message['sources'])})"):
                        for source in message["sources"]:
                            number = _citation_number(source, 1)
                            st.markdown(
                                f"[{number}] [{source.get('title', 'Source')}]"
                                f"({source['url']})"
                            )
                            st.caption(
                                "Fetched page content"
                                if source.get("retrieved_as") == "page"
                                else "Search snippet only"
                            )
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask a quick question or follow up…", max_chars=1200)
    if prompt and prompt.strip():
        prompt = prompt.strip()
        prior_history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.quick_messages[-8:]
        ]
        st.session_state.quick_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=":material/travel_explore:"):
            status = None
            try:
                instant_answer = quick_greeting_response(prompt)
                if instant_answer:
                    result = {
                        "answer": instant_answer,
                        "sources": [],
                        "note": None,
                    }
                else:
                    status = None
                    with st.status("Planning your answer", expanded=False) as status:
                        result = run_research_direct(
                            prompt,
                            "Quick Answer",
                            status,
                            APP_CONFIG,
                            prior_history,
                        )
                        status.update(
                            label="Answer ready", state="complete", expanded=False
                        )
                st.markdown(
                    link_citations(result["answer"], result.get("sources", []))
                )
                if result.get("note"):
                    st.caption(result["note"])
                if result.get("sources"):
                    with st.expander(f"Sources used ({len(result['sources'])})"):
                        for source in result["sources"]:
                            number = _citation_number(source, 1)
                            st.markdown(
                                f"[{number}] [{source.get('title', 'Source')}]"
                                f"({source['url']})"
                            )
                            st.caption(
                                "Fetched page content"
                                if source.get("retrieved_as") == "page"
                                else "Search snippet only"
                            )
                st.session_state.quick_messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "note": result.get("note"),
                    }
                )
            except (RuntimeError, ValueError, KeyError) as exc:
                if status is not None:
                    status.update(label="Answer could not be generated", state="error")
                st.error(str(exc))
                st.session_state.quick_messages.append(
                    {
                        "role": "assistant",
                        "content": f"Unable to answer: {exc}",
                        "sources": [],
                    }
                )
    st.markdown(
        '<div class="rm-footer">Quick Answer keeps this conversation separate '
        'from research reports and preserves it for this Streamlit session.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

current = st.session_state.research_result
if current and current.get("depth") != mode:
    current = None
if current is None:
    render_brand(main=True)
    st.markdown('<div class="rm-hero">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="rm-eyebrow">ResearchMate AI · {mode}</div>',
        unsafe_allow_html=True,
    )
    st.title(
        "Build a focused research brief."
        if mode == "Standard Research"
        else "Investigate the question in depth."
    )
    mode_description = (
        "A concise, source-backed investigation designed to answer the key question."
        if mode == "Standard Research"
        else "A broader review with more sources, detailed synthesis, and optional lexical topic grouping."
    )
    st.markdown(
        f'<div class="rm-subtitle">{mode_description} Explore the evidence '
        "and take the report with you.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="rm-prompt-label">Choose a starting point</div>', unsafe_allow_html=True)
    examples = [
        ("EXPLAIN", "How do large language models learn to generate text?"),
        ("COMPARE", "Compare PostgreSQL and MongoDB for a small SaaS product."),
        ("TECHNOLOGY", "What are the main trade-offs of small modular reactors?"),
        ("CURRENT", "What are the latest developments in quantum error correction?"),
    ]
    example_cols = st.columns(2)
    for index, (category, prompt) in enumerate(examples):
        with example_cols[index % 2]:
            st.caption(category)
            if st.button(
                prompt,
                key=f"example_{index}",
                use_container_width=True,
            ):
                st.session_state.question_input = prompt
                st.rerun()

    with st.form("research_question_form"):
        question = st.text_area(
            "Your research question",
            placeholder=(
                "What would you like to investigate?"
                if mode == "Standard Research"
                else "What question deserves a deeper evidence review?"
            ),
            max_chars=1200,
            height=130,
            key="question_input",
        )
        st.caption(f"Selected mode: **{mode}** · Change modes in the sidebar.")
        submitted = st.form_submit_button(
            "Start research  →" if mode == "Standard Research" else "Start deep research  →",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.error("Enter a question to continue.")
        else:
            try:
                with st.status("Planning your research", expanded=True) as status:
                    result = run_research_direct(
                        question.strip(), mode, status, APP_CONFIG
                    )
                    status.update(label="Research complete", state="complete")
                _store_report(question.strip(), mode, result)
                st.rerun()
            except (RuntimeError, ValueError, KeyError) as exc:
                st.error(str(exc))
else:
    render_brand(main=True)
    st.markdown(
        f'<div class="rm-eyebrow">{current.get("depth", mode)} · Research report</div>',
        unsafe_allow_html=True,
    )
    st.subheader(current.get("question", st.session_state.last_question))
    completed_at = current.get("generated_at")
    if completed_at:
        try:
            completion_label = datetime.fromisoformat(completed_at).astimezone().strftime(
                "%b %d, %Y · %I:%M %p"
            )
        except ValueError:
            completion_label = "Completed"
    else:
        completion_label = "Completed"
    depth_label = current.get("depth", "Standard Research")
    source_count = len(current.get("sources", []))
    research_label = "Web research" if current.get("searched") else "Direct answer"
    st.markdown(
        f'<div class="rm-meta"><span class="rm-pill">{depth_label}</span>'
        f'<span class="rm-pill">{research_label}</span>'
        f'<span class="rm-pill">{source_count} source(s)</span>'
        f'<span class="rm-pill">Completed {completion_label}</span></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    if current.get("note"):
        st.warning(current["note"])
    if current.get("searched") and not current.get("sources"):
        st.warning(
            "No relevant sources were retrieved. The answer should identify "
            "where the available evidence is insufficient."
        )

    st.markdown("#### Findings")
    with st.container(border=True):
        st.markdown(
            link_citations(current["answer"], current.get("sources", [])),
            help="Inline citation numbers link to the corresponding retrieved sources.",
        )

    if current.get("clusters"):
        with st.expander("Optional lexical topic grouping"):
            st.caption(
                "TF-IDF/K-Means groups shared word usage; it does not establish semantic relationships."
            )
            for cluster in current["clusters"]:
                st.markdown(
                    f"**{cluster.get('label', 'Topic')}** — "
                    f"{', '.join(cluster.get('keywords', []))}"
                )

    sources = current.get("sources", [])
    if sources:
        st.markdown("#### Evidence and references")
        st.caption(
            "Citation numbers in the answer map to these retrieved sources. "
            "Search snippets are labeled when page content was unavailable."
        )
        for index, source in enumerate(sources, 1):
            citation_number = _citation_number(source, index)
            title = source.get("title") or source.get("url") or f"Source {index}"
            with st.expander(f"[{citation_number}]  {title}"):
                provenance = (
                    "Fetched page content"
                    if source.get("retrieved_as") == "page"
                    else "Search-result snippet only; page content was not fetched"
                )
                if source.get("published_date"):
                    provenance += f" · Published {source['published_date']}"
                st.caption(provenance)
                st.link_button(
                    "Open original source",
                    source["url"],
                    key=f"source_{index}_{source['id']}",
                )
                excerpt = str(source.get("content", "")).strip()
                if excerpt:
                    st.markdown("**Retrieved excerpt**")
                    st.text(excerpt[:900] + ("…" if len(excerpt) > 900 else ""))
                else:
                    st.caption("No source text or search snippet was available.")

    if current.get("searched") and not sources:
        st.info("There are no retrieved references to display for this answer.")

    st.markdown("#### Export this report")
    markdown_col, json_col, pdf_col = st.columns(3)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_col.download_button(
        "Download Markdown",
        data=current["markdown"],
        file_name=f"researchmate_{stamp}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    json_col.download_button(
        "Download JSON",
        data=current["json"],
        file_name=f"researchmate_{stamp}.json",
        mime="application/json",
        use_container_width=True,
    )
    pdf_col.download_button(
        "Download PDF",
        data=make_report_pdf(current.get("question", ""), current),
        file_name=f"researchmate_{stamp}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

st.markdown(
    '<div class="rm-footer">ResearchMate AI makes evidence easier to inspect. '
    'Generated answers can still be wrong; verify important claims against the sources.</div>',
    unsafe_allow_html=True,
)
