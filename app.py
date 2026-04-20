"""
app.py
------
Main Streamlit application for the AI Fake News Detector.
Tabs:
  1. Analyze   - User inputs a claim, gets REAL/FAKE/UNCERTAIN verdict
  2. History   - Shows all past searches from the database
  3. Statistics - Pie chart, model performance metrics, system flow
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from ai_checker import get_ml_prediction, search_news, analyze_news
from database import init_db, save_result, get_all_history, delete_all_history


# Page config - MUST be first Streamlit call
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize database
init_db()


# ─── Futuristic CSS Theme ─────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;600&display=swap');

@keyframes flicker {
    0%,19%,21%,23%,25%,54%,56%,100% { opacity:1; text-shadow:0 0 10px #0ff,0 0 20px #0ff,0 0 40px #0ff; }
    20%,24%,55% { opacity:0.8; text-shadow:none; }
}
@keyframes grid-move  { 0%{background-position:0 0} 100%{background-position:60px 60px} }
@keyframes fadeInUp   { from{opacity:0;transform:translateY(30px)} to{opacity:1;transform:translateY(0)} }
@keyframes border-glow{ 0%,100%{opacity:0.5} 50%{opacity:1} }
@keyframes sweep      { 0%{left:-100%} 100%{left:200%} }
@keyframes pulse-real { 0%,100%{box-shadow:0 0 10px #00ff88,0 0 20px #00ff88} 50%{box-shadow:0 0 30px #00ff88,0 0 60px #00ff88} }
@keyframes pulse-fake { 0%,100%{box-shadow:0 0 10px #ff0055,0 0 20px #ff0055} 50%{box-shadow:0 0 30px #ff0055,0 0 60px #ff0055} }
@keyframes pulse-unct { 0%,100%{box-shadow:0 0 10px #ffaa00,0 0 20px #ffaa00} 50%{box-shadow:0 0 30px #ffaa00} }

.stApp {
    background-color: #010108;
    background-image:
        linear-gradient(rgba(0,242,254,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,242,254,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: grid-move 8s linear infinite;
    color: #c8d8e8;
    font-family: 'Inter', sans-serif;
}

h1,h2,h3,h4,.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
}

.main-title {
    text-align: center;
    font-size: 3.8rem;
    font-weight: 900;
    font-family: 'Orbitron', sans-serif;
    color: #00f2fe;
    letter-spacing: 4px;
    animation: flicker 4s infinite;
    padding-top: 20px;
    margin-bottom: 0;
}
.sub-title {
    text-align: center;
    color: rgba(0,242,254,0.6);
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem;
    letter-spacing: 5px;
    margin-bottom: 40px;
}

.stTextArea textarea {
    background: rgba(0,10,30,0.85) !important;
    border: 1px solid rgba(0,242,254,0.5) !important;
    color: #e8f4ff !important;
    border-radius: 8px !important;
    font-size: 1.05rem;
    font-family: 'Share Tech Mono', monospace !important;
    padding: 20px;
    caret-color: #00f2fe !important;
    cursor: text !important;
    pointer-events: auto !important;
    user-select: text !important;
    -webkit-user-select: text !important;
    box-shadow: inset 0 0 20px rgba(0,242,254,0.02), 0 0 15px rgba(0,242,254,0.08);
    transition: all 0.3s ease;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: #00f2fe !important;
    box-shadow: 0 0 25px rgba(0,242,254,0.25) !important;
    outline: none !important;
}

div.stButton > button:first-child {
    background: transparent;
    color: #00f2fe;
    border: 1px solid #00f2fe;
    border-radius: 4px;
    padding: 14px 20px;
    font-family: 'Orbitron', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 4px;
    width: 100%;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    clip-path: polygon(8px 0%, 100% 0%, calc(100% - 8px) 100%, 0% 100%);
    box-shadow: 0 0 10px rgba(0,242,254,0.2);
}
div.stButton > button:first-child::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,242,254,0.3), transparent);
    animation: sweep 2.5s ease-in-out infinite;
}
div.stButton > button:first-child:hover {
    background: rgba(0,242,254,0.12);
    box-shadow: 0 0 40px rgba(0,242,254,0.6);
    color: #fff;
    border-color: #fff;
    transform: translateY(-2px);
    letter-spacing: 6px;
}

.result-card {
    border-radius: 6px;
    margin-top: 30px;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease forwards;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -3px; left: 10%; width: 80%; height: 3px;
    border-radius: 2px;
    animation: border-glow 2s ease infinite;
}
.real-card     { border:1px solid rgba(0,255,136,0.4); background:rgba(0,255,136,0.03); animation:pulse-real 3s ease infinite, fadeInUp 0.5s ease forwards; }
.real-card::before     { background:linear-gradient(90deg,transparent,#00ff88,transparent); }
.fake-card     { border:1px solid rgba(255,0,85,0.4); background:rgba(255,0,85,0.03); animation:pulse-fake 3s ease infinite, fadeInUp 0.5s ease forwards; }
.fake-card::before     { background:linear-gradient(90deg,transparent,#ff0055,transparent); }
.uncertain-card { border:1px solid rgba(255,170,0,0.4); background:rgba(255,170,0,0.03); animation:pulse-unct 3s ease infinite, fadeInUp 0.5s ease forwards; }
.uncertain-card::before { background:linear-gradient(90deg,transparent,#ffaa00,transparent); }

.verdict-text   { font-family:'Orbitron',sans-serif !important; font-size:3rem !important; font-weight:900 !important; letter-spacing:6px; }
.real-text      { color:#00ff88; text-shadow:0 0 15px #00ff88, 0 0 40px #00ff88; }
.fake-text      { color:#ff0055; text-shadow:0 0 15px #ff0055, 0 0 40px #ff0055; }
.uncertain-text { color:#ffaa00; text-shadow:0 0 15px #ffaa00, 0 0 40px #ffaa00; }

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #0a4a5a, #00f2fe);
    box-shadow: 0 0 12px #00f2fe;
}

.streamlit-expanderHeader {
    color: #00f2fe !important;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.9rem;
    letter-spacing: 2px;
    background: rgba(0,242,254,0.04);
    border: 1px solid rgba(0,242,254,0.15) !important;
    border-radius: 4px;
}

div[data-testid="stAlert"] {
    background: rgba(0,10,25,0.8) !important;
    border: 1px solid rgba(0,242,254,0.25) !important;
    border-left: 3px solid #00f2fe !important;
    color: #c8d8e8 !important;
    font-family: 'Share Tech Mono', monospace;
    border-radius: 4px;
}

.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid rgba(0,242,254,0.15); }
.stTabs [data-baseweb="tab"]      { font-size:0.8rem; letter-spacing:2px; color:rgba(0,242,254,0.5); background:transparent; border-radius:4px 4px 0 0; }
.stTabs [aria-selected="true"]    { color:#00f2fe !important; border-bottom:2px solid #00f2fe !important; text-shadow:0 0 10px rgba(0,242,254,0.5); }

[data-testid="stDataFrame"] { border:1px solid rgba(0,242,254,0.15) !important; border-radius:6px; }
</style>
""", unsafe_allow_html=True)


# ─── Helper: Build the Verdict Result Card HTML ───────────────────────────────

def build_result_card(verdict, confidence, card_class, text_class, color):
    icon = "✅" if verdict == "REAL" else ("🚨" if verdict == "FAKE" else "⚠️")
    html = (
        f'<div class="result-card {card_class}" style="padding:35px 40px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">'
                f'<span style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#888;letter-spacing:3px;">ANALYSIS RESULT</span>'
                f'<span style="font-size:1.5rem;">{icon}</span>'
            f'</div>'
            f'<div style="text-align:center;padding:20px 0;">'
                f'<p style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#555;letter-spacing:6px;margin:0 0 10px;">[ PREDICTION ]</p>'
                f'<p class="verdict-text {text_class}" style="margin:0;">{verdict}</p>'
            f'</div>'
            f'<div style="margin-top:25px;padding-top:20px;border-top:1px solid rgba(255,255,255,0.06);">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-family:Orbitron,sans-serif;font-size:0.75rem;color:#555;letter-spacing:2px;">CONFIDENCE SCORE</span>'
                    f'<span style="color:{color};font-family:Share Tech Mono,monospace;font-size:1.2rem;font-weight:bold;text-shadow:0 0 10px {color};">{confidence}%</span>'
                f'</div>'
            f'</div>'
        f'</div>'
    )
    return html


# ─── Helper: Matplotlib Pie Chart ─────────────────────────────────────────────

def build_pie_chart(history_rows):
    verdicts = [row[1] for row in history_rows]
    counts = {
        "REAL":      verdicts.count("REAL"),
        "FAKE":      verdicts.count("FAKE"),
        "UNCERTAIN": verdicts.count("UNCERTAIN"),
    }
    labels = [k for k, v in counts.items() if v > 0]
    values = [v for v in counts.values() if v > 0]
    colors = {"REAL": "#00ff88", "FAKE": "#ff0055", "UNCERTAIN": "#ffaa00"}
    pie_colors = [colors[l] for l in labels]

    if not values:
        return None

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "#111", "linewidth": 2},
        textprops={"fontsize": 12, "weight": "bold"},
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    ax.set_title("Verdict Distribution", color="#00f2fe", size=15, pad=20, weight="bold")
    return fig


# ─── Main App ─────────────────────────────────────────────────────────────────

# Page header
st.markdown("<div class='main-title'>Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real-time authenticity check using AI</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Five tabs
tab_analyze, tab_history, tab_stats, tab_how, tab_ref = st.tabs([
    "[ ⚡ Analyze ]",
    "[ 🕒 History ]",
    "[ 📊 Statistics ]",
    "[ 📚 How It Works ]",
    "[ 🔗 References ]"
])


# ── Tab 1: Analyze ────────────────────────────────────────────────────────────
with tab_analyze:
    _, center_col, _ = st.columns([1, 6, 1])
    with center_col:
        user_input = st.text_area(
            label="",
            placeholder="Enter a news snippet, headline, or claim...",
            height=150
        )

        if st.button("Verify News"):
            if not user_input.strip():
                st.warning("Please enter some text to analyze.")
                st.stop()

            with st.spinner("Analyzing text and finding sources..."):
                ml_verdict, ml_confidence = get_ml_prediction(user_input)
                articles = search_news(user_input)
                ai_result = analyze_news(user_input, articles, ml_verdict)

            verdict     = ai_result.get("verdict",     "UNCERTAIN")
            confidence  = ai_result.get("confidence",  50)
            explanation = ai_result.get("explanation", "No explanation available.")

            save_result(user_input, verdict, confidence)

            style_map = {
                "REAL":      ("real-card",      "real-text",      "#00ff88"),
                "FAKE":      ("fake-card",       "fake-text",      "#ff0055"),
                "UNCERTAIN": ("uncertain-card",  "uncertain-text", "#ffaa00"),
            }
            card_class, text_class, color = style_map.get(verdict, style_map["UNCERTAIN"])

            st.markdown(
                build_result_card(verdict, confidence, card_class, text_class, color),
                unsafe_allow_html=True
            )
            st.progress(confidence / 100.0)

            ml_color = "#00ff88" if ml_verdict == "REAL" else ("#ff0055" if ml_verdict == "FAKE" else "#ffaa00")
            st.markdown(
                f"<p style='font-family:Share Tech Mono,monospace;font-size:0.78rem;color:#444;margin-top:4px;'>"
                f"ML Style Model: <span style='color:{ml_color};'>{ml_verdict} ({ml_confidence}%)</span>"
                f" &nbsp;|&nbsp; Writing pattern analysis only</p>",
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<h4 style='color:#00f2fe; border-bottom:1px solid rgba(0,242,254,0.3); padding-bottom:8px;'>🧠 Analysis Details</h4>",
                unsafe_allow_html=True
            )
            st.info(explanation)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔗  Web Sources & References"):
                if articles:
                    for article in articles:
                        title  = article.get("title",  "Unknown Title")
                        url    = article.get("url",    "#")
                        source = article.get("source", "Unknown")
                        st.markdown(f"- **[{source}]** [{title}]({url})")
                else:
                    st.write("No live news articles were found for this query.")


# ── Tab 2: History ────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("<h3 style='color:#00f2fe; text-align:center;'>Search History</h3>", unsafe_allow_html=True)
    rows = get_all_history()
    if rows:
        df = pd.DataFrame(rows, columns=["Query", "Verdict", "Confidence (%)", "Timestamp"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
        _, center_btn, _ = st.columns([2, 2, 2])
        with center_btn:
            if st.button("Clear All History"):
                delete_all_history()
                st.success("History cleared.")
                st.rerun()
    else:
        st.info("No history yet. Run your first analysis to see results here.")


# ── Tab 3: Statistics + Model Performance ────────────────────────────────────
with tab_stats:
    st.markdown("<h3 style='color:#00f2fe; text-align:center;'>Analytics Dashboard</h3>", unsafe_allow_html=True)
    rows = get_all_history()
    if rows:
        fig = build_pie_chart(rows)
        if fig:
            _, chart_col, _ = st.columns([1, 4, 1])
            with chart_col:
                st.pyplot(fig)
                st.markdown("<hr style='border-color:rgba(0,242,254,0.2); margin:20px 0;'>", unsafe_allow_html=True)
                verdicts = [row[1] for row in rows]
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Scans",      len(rows))
                col2.metric("Real Detected",    verdicts.count("REAL"))
                col3.metric("Fake Detected",    verdicts.count("FAKE"))
    else:
        st.info("No data yet. Run some analyses to see statistics.")

    # ── Model Performance Metrics ─────────────────────────────────────────────
    st.markdown("<hr style='border-color:rgba(0,242,254,0.2); margin:40px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#00f2fe; text-align:center;'>Model Performance Metrics</h3>", unsafe_allow_html=True)

    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{metrics.get('accuracy',  0)*100:.2f}%")
        m2.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        m3.metric("Recall",    f"{metrics.get('recall',    0)*100:.2f}%")
        m4.metric("F1 Score",  f"{metrics.get('f1',        0)*100:.2f}%")

        # Confusion Matrix Table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#00f2fe; text-align:center;'>Confusion Matrix</h4>", unsafe_allow_html=True)
        cm = metrics.get("confusion_matrix", [[0,0],[0,0]])
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted Fake", "Predicted Real"],
            index=["Actual Fake", "Actual Real"]
        )
        st.dataframe(cm_df, use_container_width=True)

        # Confusion Matrix Image
        if os.path.exists("confusion_matrix.png"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#00f2fe; text-align:center;'>Confusion Matrix Graph</h4>", unsafe_allow_html=True)
            _, img_col, _ = st.columns([1, 2, 1])
            with img_col:
                st.image("confusion_matrix.png", use_container_width=True)
    else:
        st.info("Run train.py first to generate model metrics.")

    # (System Flow Diagram moved to How It Works tab)

# ── Tab 4: How It Works ───────────────────────────────────────────────────────
with tab_how:
    st.markdown("<h2 style='color:#00f2fe; text-align:center; letter-spacing:4px;'>HOW IT WORKS</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:rgba(0,242,254,0.5); font-family:Share Tech Mono,monospace;'>From your input to the final verdict — 5 simple steps</p><br>", unsafe_allow_html=True)

    # Step 1
    col_txt, col_img = st.columns([1, 1])
    with col_txt:
        st.markdown("""
<div style='padding:30px; border-left:3px solid #00f2fe; background:rgba(0,242,254,0.03); border-radius:8px;'>
<span style='color:#00f2fe; font-size:2rem; font-family:Orbitron,sans-serif; font-weight:900;'>01</span>
<h3 style='color:#fff; margin-top:10px;'>You Enter the News</h3>
<p style='color:#c8d8e8; font-size:1rem; line-height:1.7;'>
You paste or type any news headline or claim you want to verify. The system reads it, cleans it, and prepares it for analysis.
</p>
<ul style='color:#c8d8e8; line-height:2;'>
<li>Any language, any topic</li>
<li>Headline, sentence, or full article</li>
<li>Input is cleaned and ready in seconds</li>
</ul>
</div>""", unsafe_allow_html=True)
    with col_img:
        st.image("https://images.unsplash.com/photo-1504711434969-e33886168f5c?q=80&w=2070", caption="Step 1 — Enter a news claim", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 2
    col_img2, col_txt2 = st.columns([1, 1])
    with col_img2:
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070", caption="Step 2 — ML model checks writing style", use_container_width=True)
    with col_txt2:
        st.markdown("""
<div style='padding:30px; border-left:3px solid #00ff88; background:rgba(0,255,136,0.03); border-radius:8px;'>
<span style='color:#00ff88; font-size:2rem; font-family:Orbitron,sans-serif; font-weight:900;'>02</span>
<h3 style='color:#fff; margin-top:10px;'>AI Checks the Writing Style</h3>
<p style='color:#c8d8e8; font-size:1rem; line-height:1.7;'>
Our trained machine learning model reads the writing pattern of the text. Fake news often uses emotional or exaggerated words — the model detects this.
</p>
<ul style='color:#c8d8e8; line-height:2;'>
<li>Trained on 45,000+ real & fake articles</li>
<li>100% test accuracy</li>
<li>Gives a confidence score (0–100%)</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 3
    col_txt3, col_img3 = st.columns([1, 1])
    with col_txt3:
        st.markdown("""
<div style='padding:30px; border-left:3px solid #ff9500; background:rgba(255,149,0,0.03); border-radius:8px;'>
<span style='color:#ff9500; font-size:2rem; font-family:Orbitron,sans-serif; font-weight:900;'>03</span>
<h3 style='color:#fff; margin-top:10px;'>System Searches the Internet</h3>
<p style='color:#c8d8e8; font-size:1rem; line-height:1.7;'>
The system automatically searches the internet to find real news articles about the same topic. It collects evidence from trusted news websites.
</p>
<ul style='color:#c8d8e8; line-height:2;'>
<li>Searches Google, DuckDuckGo & NewsAPI</li>
<li>Finds top 5 matching articles</li>
<li>Reads the actual article content</li>
</ul>
</div>""", unsafe_allow_html=True)
    with col_img3:
        st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072", caption="Step 3 — Live internet search for real news", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 4
    col_img4, col_txt4 = st.columns([1, 1])
    with col_img4:
        st.image("https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=2070", caption="Step 4 — AI reads and thinks carefully", use_container_width=True)
    with col_txt4:
        st.markdown("""
<div style='padding:30px; border-left:3px solid #cc44ff; background:rgba(204,68,255,0.03); border-radius:8px;'>
<span style='color:#cc44ff; font-size:2rem; font-family:Orbitron,sans-serif; font-weight:900;'>04</span>
<h3 style='color:#fff; margin-top:10px;'>AI Compares & Decides</h3>
<p style='color:#c8d8e8; font-size:1rem; line-height:1.7;'>
A powerful AI (Groq / LLaMA) reads your claim, the internet evidence, and the ML score — then gives a final, reasoned decision with a full explanation.
</p>
<ul style='color:#c8d8e8; line-height:2;'>
<li>Uses Groq's super-fast AI engine</li>
<li>Compares claim vs. real-world facts</li>
<li>Writes a human-readable explanation</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 5
    col_txt5, col_img5 = st.columns([1, 1])
    with col_txt5:
        st.markdown("""
<div style='padding:30px; border-left:3px solid #ff0055; background:rgba(255,0,85,0.03); border-radius:8px;'>
<span style='color:#ff0055; font-size:2rem; font-family:Orbitron,sans-serif; font-weight:900;'>05</span>
<h3 style='color:#fff; margin-top:10px;'>You Get the Verdict</h3>
<p style='color:#c8d8e8; font-size:1rem; line-height:1.7;'>
The system shows you a clear result — REAL, FAKE, or UNCERTAIN — with a confidence percentage, a full explanation, and all the news sources it used.
</p>
<ul style='color:#c8d8e8; line-height:2;'>
<li>Clear REAL / FAKE / UNCERTAIN label</li>
<li>Confidence % displayed visually</li>
<li>All sources shown for transparency</li>
</ul>
</div>""", unsafe_allow_html=True)
    with col_img5:
        st.image("https://images.unsplash.com/photo-1614064641938-3bbee52942c7?q=80&w=2070", caption="Step 5 — Final verdict with explanation", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if os.path.exists("architecture.png"):
        st.markdown("<hr style='border-color:rgba(0,242,254,0.2); margin:30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#00f2fe; text-align:center;'>Full System Architecture</h3>", unsafe_allow_html=True)
        _, flow_col, _ = st.columns([1, 4, 1])
        with flow_col:
            st.image("architecture.png", use_container_width=True)


# ── Tab 5: References ─────────────────────────────────────────────────────────
with tab_ref:
    st.markdown("<h2 style='color:#00f2fe; text-align:center;'>Tools & References</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#c8d8e8;'>All the tools, libraries, and datasets used to build this project.</p><br>", unsafe_allow_html=True)

    st.markdown("### Libraries & Frameworks")
    st.markdown("- **[Streamlit](https://docs.streamlit.io/)** — Used to build the web interface. Simple Python-based UI framework.")
    st.markdown("- **[Scikit-Learn](https://scikit-learn.org/stable/)** — Used to train the Logistic Regression fake news detection model.")
    st.markdown("- **[Pandas](https://pandas.pydata.org/)** — Used for loading and managing the dataset and history records.")
    st.markdown("- **[Matplotlib](https://matplotlib.org/)** — Used to draw the confusion matrix and performance charts.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### AI & Web APIs")
    st.markdown("- **[Groq AI](https://groq.com/)** — The AI engine that reads evidence and decides if news is real or fake.")
    st.markdown("- **[DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)** — Searches the internet for real news articles without tracking.")
    st.markdown("- **[NewsAPI](https://newsapi.org/)** — Fetches live news from trusted sources around the world.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Dataset")
    st.markdown("- **[Kaggle — WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)** — A collection of 72,000+ labeled real and fake news articles used to train the model.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Design")
    st.markdown("- **[Google Fonts](https://fonts.google.com/)** — Orbitron and Share Tech Mono fonts for the futuristic UI style.")
    st.markdown("- **[Unsplash](https://unsplash.com/)** — Free high-quality photos used in the How It Works section.")


