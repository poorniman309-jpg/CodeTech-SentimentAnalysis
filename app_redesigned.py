import nltk
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SentimentScope AI",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #080d1c 0%, #10152b 55%, #17102b 100%);
    color: #eef0ff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10172e, #17122e);
    border-right: 1px solid #292747;
}

[data-testid="stSidebar"] * {
    color: #e7e9ff !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

h1, h2, h3, p, label {
    color: #f1f2ff;
}

.hero {
    background: linear-gradient(115deg, #171c3b, #302052);
    border: 1px solid #49366d;
    border-radius: 22px;
    padding: 28px 32px;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #b9bbdc;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    margin: 20px 0 14px;
}

.metric-card {
    background: linear-gradient(145deg, #171e38, #201a3b);
    border: 1px solid #333358;
    border-radius: 17px;
    padding: 21px;
    min-height: 125px;
}

.metric-label {
    color: #a9afd0;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #ffffff;
    font-size: 29px;
    font-weight: 800;
    margin-top: 9px;
}

.metric-note {
    color: #9b91d9;
    font-size: 12px;
    margin-top: 5px;
}

.panel {
    background: #141a30;
    border: 1px solid #2b3151;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

div.stButton > button {
    background: linear-gradient(100deg, #7657ee, #a04ff0);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    min-height: 44px;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: linear-gradient(100deg, #8c70ff, #b66aff);
    color: white;
    border: 0;
}

.stTextArea textarea {
    background: #0e1428;
    color: #f1f2ff;
    border: 1px solid #34365c;
    border-radius: 12px;
}

[data-testid="stFileUploader"] {
    background: #11172c;
    border: 1px dashed #55517e;
    border-radius: 12px;
    padding: 10px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #303452;
    border-radius: 12px;
}

div[data-testid="stMetric"] {
    background: #171d36;
    padding: 15px;
    border-radius: 13px;
    border: 1px solid #303452;
}

hr {
    border-color: #2c3150;
}

.small-muted {
    color: #a4aac8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SENTIMENT MODEL ----------------
@st.cache_resource
def load_analyzer():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


analyzer = load_analyzer()


def predict_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, compound, scores


# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 💜 SentimentScope")
    st.caption("AI-Powered Sentiment Intelligence")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Analyze Text",
            "Batch Analysis",
            "Analysis History"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("### ⚙️ Model Information")
    st.caption("Model: VADER Sentiment Analyzer")
    st.caption("Technology: Python + NLTK")
    st.caption("Classes: Positive, Neutral, Negative")

    if st.button("🗑️ Clear Analysis History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ---------------- HELPERS ----------------
def metric_card(label, value, note, icon):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{icon} &nbsp; {label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_history_df():
    if not st.session_state.history:
        return pd.DataFrame(
            columns=["Time", "Text", "Sentiment", "Score"]
        )
    return pd.DataFrame(st.session_state.history)


def sentiment_chart(data):
    counts = (
        data["Sentiment"]
        .value_counts()
        .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        .reset_index()
    )
    counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        hole=0.68,
        color="Sentiment",
        color_discrete_map={
            "Positive": "#35d6a1",
            "Neutral": "#8a8ff7",
            "Negative": "#ff6685"
        }
    )

    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        marker=dict(line=dict(color="#141a30", width=4))
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e9eaff",
        showlegend=False,
        margin=dict(t=15, b=15, l=15, r=15),
        height=330
    )
    return fig


def trend_chart(data):
    plot_data = data.copy()
    plot_data["Time"] = pd.to_datetime(plot_data["Time"])

    fig = px.line(
        plot_data.sort_values("Time"),
        x="Time",
        y="Score",
        markers=True,
        color_discrete_sequence=["#a17aff"]
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8)
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#b9bfdf",
        height=300,
        margin=dict(t=15, b=15, l=10, r=10),
        xaxis_title="Analysis Time",
        yaxis_title="Compound Score",
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#292e4a"),
        xaxis=dict(gridcolor="#292e4a")
    )
    return fig


# ---------------- HERO ----------------
st.markdown(
    """
    <div class="hero">
        <h1>💜 SentimentScope AI</h1>
        <p>Understand emotions, discover sentiment patterns, and transform text
        into actionable insights with Natural Language Processing.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- OVERVIEW ----------------
if page == "Overview":
    history = get_history_df()

    total = len(history)
    positive = int((history["Sentiment"] == "Positive").sum()) if total else 0
    neutral = int((history["Sentiment"] == "Neutral").sum()) if total else 0
    negative = int((history["Sentiment"] == "Negative").sum()) if total else 0

    st.markdown('<div class="section-title">📊 Sentiment Overview</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Texts Analyzed", total, "Total text analyses this session", "📝")
    with c2:
        metric_card("Positive", positive, "Positive sentiment detected", "😊")
    with c3:
        metric_card("Neutral", neutral, "Neutral sentiment detected", "😐")
    with c4:
        metric_card("Negative", negative, "Negative sentiment detected", "☹️")

    left, right = st.columns([1, 1.5])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Sentiment Distribution")

        if total:
            st.plotly_chart(sentiment_chart(history), use_container_width=True)
        else:
            st.info("Analyze some text to populate the sentiment distribution.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Sentiment Trend")

        if total:
            st.plotly_chart(trend_chart(history), use_container_width=True)
        else:
            st.info("Your sentiment trend will appear here after analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🕘 Recent Analysis</div>',
                unsafe_allow_html=True)

    if total:
        recent = history.tail(5).iloc[::-1]
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No analysis yet. Select **Analyze Text** from the sidebar to begin.")

    st.markdown(
        '<p class="small-muted">Note: Statistics are based on analyses in the current session.</p>',
        unsafe_allow_html=True
    )


# ---------------- ANALYZE TEXT ----------------
elif page == "Analyze Text":
    st.markdown('<div class="section-title">✍️ Analyze Your Text</div>',
                unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Enter Text")
        user_text = st.text_area(
            "Paste a tweet, review, comment, or social media post",
            height=210,
            placeholder="Example: I absolutely love this product! It made my day..."
        )

        analyze_clicked = st.button(
            "✨ Analyze Sentiment",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### How it works")
        st.write("1. Enter text in the analysis box.")
        st.write("2. Click **Analyze Sentiment**.")
        st.write("3. Review the predicted sentiment and score.")
        st.write("4. Your analysis appears in the dashboard history.")
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze_clicked:
        if not user_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            label, score, scores = predict_sentiment(user_text)

            st.session_state.history.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Text": user_text,
                "Sentiment": label,
                "Score": score
            })

            st.markdown('<div class="section-title">🎯 Prediction Result</div>',
                        unsafe_allow_html=True)

            a, b, c = st.columns(3)

            with a:
                metric_card("Predicted Sentiment", label, "VADER classification", "💡")
            with b:
                metric_card("Compound Score", f"{score:.3f}", "Range: -1.0 to +1.0", "📈")
            with c:
                metric_card("Score Strength", f"{abs(score) * 100:.1f}%", "Absolute compound score", "⚡")

            if label == "Positive":
                st.success("😊 This text expresses positive sentiment.")
            elif label == "Negative":
                st.error("☹️ This text expresses negative sentiment.")
            else:
                st.info("😐 This text is classified as neutral sentiment.")

            with st.expander("View detailed sentiment scores"):
                st.json(scores)

            st.caption(
                "The score strength is the absolute compound score, not a calibrated probability."
            )


# ---------------- BATCH ANALYSIS ----------------
elif page == "Batch Analysis":
    st.markdown('<div class="section-title">📁 Batch Sentiment Analysis</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write("Upload a CSV file containing reviews, tweets, comments, or other text.")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            if data.empty:
                st.warning("The uploaded CSV file is empty.")
            else:
                st.markdown("#### Preview")
                st.dataframe(data.head(10), use_container_width=True)

                text_column = st.selectbox(
                    "Select the column containing text",
                    data.columns.tolist()
                )

                if st.button("🚀 Analyze CSV", use_container_width=True):
                    cleaned = data.dropna(subset=[text_column]).copy()
                    cleaned[text_column] = cleaned[text_column].astype(str)

                    predictions = cleaned[text_column].apply(predict_sentiment)
                    cleaned["Sentiment"] = predictions.apply(lambda x: x[0])
                    cleaned["Compound Score"] = predictions.apply(lambda x: x[1])

                    st.session_state.batch_results = cleaned

                    st.success(f"Successfully analyzed {len(cleaned)} rows.")

        except Exception as error:
            st.error(f"Unable to read this CSV: {error}")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.batch_results is not None:
        results = st.session_state.batch_results

        st.markdown("### 📊 Batch Results")

        x1, x2, x3 = st.columns(3)
        x1.metric("Total Rows Analyzed", len(results))
        x2.metric("Positive", int((results["Sentiment"] == "Positive").sum()))
        x3.metric("Negative", int((results["Sentiment"] == "Negative").sum()))

        st.dataframe(results, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Analysis Results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )


# ---------------- HISTORY ----------------
elif page == "Analysis History":
    st.markdown('<div class="section-title">🕘 Analysis History</div>',
                unsafe_allow_html=True)

    history = get_history_df()

    if not history.empty:
        st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download History CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No analyses yet. Go to **Analyze Text** to get started.")


# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "💜 SentimentScope AI | Codec Technologies Internship Project | "
    "Python • Streamlit • NLTK • Pandas • Plotly"
)