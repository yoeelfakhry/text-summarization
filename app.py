import time
import streamlit as st
from inference import summarize_with_bart

st.set_page_config(
    page_title="Text Summarizer — BART vs. Pegasus",
    page_icon="📰",
    layout="wide",
)

# Styling — dark technical theme, functional color-coding per model
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }

    .stApp {
        background-color: #0F1419;
        color: #E8E6E1;
    }

    .hero-title {
        font-family: 'Newsreader', serif;
        font-weight: 600;
        font-size: 2.6rem;
        color: #F4F2ED;
        margin-bottom: 0.2rem;
        letter-spacing: -0.01em;
    }

    .hero-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem;
        color: #8B92A0;
        max-width: 640px;
        line-height: 1.6;
        margin-bottom: 1.8rem;
    }

    .model-card {
        border-radius: 4px;
        padding: 1.4rem 1.5rem;
        background-color: #161C24;
        min-height: 260px;
    }

    .model-card.bart { border-left: 3px solid #C9A227; }
    .model-card.pegasus { border-left: 3px solid #3FA796; }

    .model-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }

    .model-label.bart { color: #C9A227; }
    .model-label.pegasus { color: #3FA796; }

    .model-desc {
        font-size: 0.8rem;
        color: #8B92A0;
        margin-bottom: 1rem;
    }

    .summary-text {
        font-family: 'Newsreader', serif;
        font-size: 1.05rem;
        line-height: 1.65;
        color: #E8E6E1;
    }

    .metrics-row {
        margin-top: 1.1rem;
        padding-top: 0.9rem;
        border-top: 1px solid #262E3A;
        font-size: 0.78rem;
        color: #8B92A0;
        display: flex;
        gap: 1.4rem;
    }

    .metrics-row span.value {
        color: #E8E6E1;
    }

    section[data-testid="stSidebar"] {
        background-color: #0B0F14;
        border-right: 1px solid #1B222C;
    }

    .stButton > button {
        background-color: #C9A227;
        color: #0F1419;
        border: none;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
    }

    .stButton > button:hover {
        background-color: #E0BC3A;
        color: #0F1419;
    }

    textarea {
        background-color: #161C24 !important;
        color: #E8E6E1 !important;
        font-family: 'Newsreader', serif !important;
        font-size: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar — generation controls + project context
with st.sidebar:
    st.markdown("### Configuration")
    max_length = st.slider("Max summary length (tokens)", 32, 200, 128, step=8)
    num_beams = st.slider("Beam search width", 1, 8, 4, step=1)

    st.markdown("---")
    st.markdown("### About this project")
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#8B92A0; line-height:1.6;">
        BART-base was fine-tuned by me on CNN/DailyMail
        (20k examples, 3 epochs). Pegasus runs zero-shot —
        pretrained by Google specifically for summarization.
        <br><br>
        Full training log, 4 documented experiments, and the
        rationale for stopping hyperparameter search are in the
        project README.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.78rem;">'
        '<a href="https://github.com/yoeel/text-summarization" style="color:#C9A227;">GitHub repo →</a><br>'
        '<a href="https://huggingface.co/yoeel/bart-cnn-summarizer-20k-3ep" style="color:#3FA796;">BART model →</a>'
        '</div>',
        unsafe_allow_html=True,
    )

# Hero
st.markdown('<div class="hero-title">Summarization: fine-tuned vs. specialist</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'A news article goes through two models side by side — a BART-base I fine-tuned '
    'myself, and Pegasus, pretrained specifically for this task. Same input, same '
    'decoding settings, different training history.'
    '</div>',
    unsafe_allow_html=True,
)

# Example articles — quick-load for demoing without hunting for text
EXAMPLES = {
    "— Paste your own —": "",
    "Short: local event": (
        "Residents of Riverside gathered Saturday for the annual clean-up "
        "drive along the county's main waterway. Organizers said turnout "
        "roughly doubled from last year, with over 300 volunteers collecting "
        "an estimated two tons of debris. The event was funded in part by a "
        "grant from the state environmental agency, and organizers say they "
        "plan to expand it to two additional sites next spring."
    ),
    "Long: policy story": (
        "Lawmakers in the state capital advanced a package of energy bills "
        "Thursday after months of committee hearings, setting up a final vote "
        "expected before the legislative session ends next week. The central "
        "measure would require utilities to source a growing share of "
        "electricity from renewable sources over the next decade, with "
        "interim targets reviewed every two years. Industry groups have "
        "argued the timeline is too aggressive given current grid "
        "infrastructure, while environmental advocates say the targets do "
        "not go far enough. A separate provision in the package would create "
        "a state fund to help low-income households cover the cost of energy "
        "efficiency upgrades, funded through a small surcharge on commercial "
        "utility bills. The bill's sponsor said negotiations over the "
        "surcharge's size delayed the package by nearly two months, and "
        "further amendments are still possible before the floor vote."
    ),
}

example_choice = st.selectbox("Load an example, or paste your own text below:", list(EXAMPLES.keys()))
default_text = EXAMPLES[example_choice]

article_text = st.text_area(
    "Article text",
    value=default_text,
    height=220,
    placeholder="Paste a news article here...",
    label_visibility="collapsed",
)

generate = st.button("Summarize")

# Results
if generate:
    if not article_text or not article_text.strip():
        st.warning("Please enter or select some article text first.")
    else:
        with st.spinner("Running BART..."):
            start = time.perf_counter()
            bart_summary = summarize_with_bart(article_text, max_length=max_length, num_beams=num_beams)
            bart_time = time.perf_counter() - start

        st.markdown(
            f"""
            <div class="model-card bart">
                <div class="model-label bart">BART · FINE-TUNED</div>
                <div class="model-desc">bart-base, 20k CNN/DailyMail examples, 3 epochs</div>
                <div class="summary-text">{bart_summary}</div>
                <div class="metrics-row">
                    <div>WORDS <span class="value">{len(bart_summary.split())}</span></div>
                    <div>TIME <span class="value">{bart_time:.2f}s</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )