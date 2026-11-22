import os
import re

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Agent Research Studio", page_icon="🔎", layout="wide")

# ---------- styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .hero h1 {font-size: 2.2rem; margin-bottom: 0.2rem;}
    .hero p {color: #6b7280; margin-top: 0;}
    .score-badge {display:inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
                  font-weight: 700; font-size: 1.1rem; color: white;}
    </style>
    """,
    unsafe_allow_html=True,
)

STEP_META = {
    "search": ("🔎", "Search Agent", "Finds recent, relevant sources via Tavily"),
    "reader": ("📖", "Reader Agent", "Scrapes the most relevant page for depth"),
    "writer": ("✍️", "Writer Chain", "Drafts the structured research report"),
    "critic": ("🧐", "Critic Chain", "Scores and critiques the report"),
}

# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")

    openai_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY") or "",
    )
    tavily_key = st.text_input(
        "Tavily API key", type="password", value=os.getenv("TAVILY_API_KEY") or ""
    )
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)

    st.divider()
    st.subheader("Pipeline")
    for key, (icon, name, desc) in STEP_META.items():
        st.markdown(f"{icon} **{name}**  \n<small>{desc}</small>", unsafe_allow_html=True)

    st.divider()
    if st.session_state.get("history"):
        st.subheader("🕘 History")
        for i, item in enumerate(reversed(st.session_state["history"])):
            if st.button(item["topic"][:40], key=f"hist_{i}", use_container_width=True):
                st.session_state["state"] = item["state"]
                st.session_state["topic"] = item["topic"]
        if st.button("Clear history", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

# ---------- header ----------
st.markdown(
    '<div class="hero"><h1>🔎 Multi-Agent Research Studio</h1>'
    "<p>Search → Read → Write → Critique. Four agents, one report.</p></div>",
    unsafe_allow_html=True,
)

topic = st.text_input(
    "Research topic",
    value=st.session_state.get("topic", ""),
    placeholder="e.g. State of retrieval-augmented generation in 2026",
)
run = st.button("🚀 Run research", type="primary", disabled=not topic.strip())

# ---------- run ----------
if run:
    if not openai_key or not tavily_key:
        st.error("Add both your OpenAI and Tavily API keys in the sidebar.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["TAVILY_API_KEY"] = tavily_key

    # Imported after keys are set so clients pick them up.
    from pipeline import run_research_pipeline

    status_boxes = {}
    progress = st.progress(0.0, text="Starting…")
    cols = st.columns(4)
    for col, (key, (icon, name, _)) in zip(cols, STEP_META.items()):
        status_boxes[key] = col.empty()
        status_boxes[key].info(f"{icon} {name}\n\nWaiting")

    done_count = {"n": 0}

    def on_step(step, status, output=None):
        icon, name, _ = STEP_META[step]
        if status == "start":
            status_boxes[step].warning(f"{icon} {name}\n\nWorking…")
            progress.progress(done_count["n"] / 4, text=f"{name} is working…")
        else:
            done_count["n"] += 1
            status_boxes[step].success(f"{icon} {name}\n\nDone")
            progress.progress(done_count["n"] / 4, text=f"{name} finished")

    try:
        with st.spinner("Agents are researching…"):
            state = run_research_pipeline(topic.strip(), model=model, on_step=on_step)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    progress.progress(1.0, text="Complete ✅")
    st.session_state["state"] = state
    st.session_state["topic"] = topic.strip()
    st.session_state.setdefault("history", []).append({"topic": topic.strip(), "state": state})

# ---------- results ----------
state = st.session_state.get("state")
if state:
    st.divider()
    tab_report, tab_critic, tab_search, tab_scrape = st.tabs(
        ["📄 Report", "🧐 Critic review", "🔎 Search results", "📖 Scraped content"]
    )

    with tab_report:
        st.markdown(state["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=state["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_critic:
        m = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", state["feedback"])
        if m:
            score = float(m.group(1))
            color = "#16a34a" if score >= 8 else "#d97706" if score >= 6 else "#dc2626"
            st.markdown(
                f'<span class="score-badge" style="background:{color}">{score:g}/10</span>',
                unsafe_allow_html=True,
            )
            st.write("")
        st.markdown(state["feedback"])

    with tab_search:
        st.markdown(state["search_results"])

    with tab_scrape:
        st.markdown(state["scraped_content"])
