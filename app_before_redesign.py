import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download sentiment model if it is not already installed
@st.cache_resource
def load_analyzer():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

st.set_page_config(
    page_title="SentimentScope AI",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #18243b);
    color: #f3f6fc;
}
h1, h2, h3, p, label {
    color: #f3f6fc !important;
}
.metric-card {
    background: #202e45;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #344763;
}
div.stButton > button {
    background: #4f7cff;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 22px;
}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("💬 SentimentScope AI")
st.write("### Twitter / X Sentiment Analysis Dashboard")
st.caption("Explore text sentiment using Natural Language Processing (NLP).")

with st.sidebar:
    st.title("⚙️ Controls")
    st.write("Analyze individual posts or upload a CSV file.")
    st.info("Labels: Positive, Neutral and Negative")
    st.caption("Model: VADER Sentiment Analyzer")

tab1, tab2, tab3 = st.tabs([
    "✍️ Analyze Text",
    "📁 Batch Analysis",
    "📊 Dashboard"
])

def predict_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, compound, scores

with tab1:
    st.subheader("Analyze a post")
    user_text = st.text_area(
        "Enter a tweet or social media post",
        placeholder="Example: I love this new product! It works amazingly.",
        height=140
    )

    if st.button("🔍 Analyze Sentiment", key="analyze"):
        if user_text.strip():
            label, score, scores = predict_sentiment(user_text)
            st.session_state.history.append({
                "Text": user_text,
                "Sentiment": label,
                "Score": score
            })

            st.markdown("### Prediction Result")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Sentiment", label)
            with col2:
                st.metric("Compound Score", f"{score:.3f}")
            with col3:
                st.metric("Confidence Indicator", f"{abs(score) * 100:.1f}%")

            if label == "Positive":
                st.success("😊 This text expresses a positive sentiment.")
            elif label == "Negative":
                st.error("☹️ This text expresses a negative sentiment.")
            else:
                st.info("😐 This text expresses a neutral sentiment.")

            st.write("**Detailed sentiment scores**")
            st.json(scores)
        else:
            st.warning("Please enter some text first.")

with tab2:
    st.subheader("Upload a CSV file")
    st.write("Your CSV must contain a column with tweet or text content.")

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data")
            st.dataframe(data.head(10), use_container_width=True)

            columns = data.columns.tolist()
            text_column = st.selectbox("Select the text column", columns)

            if st.button("Analyze CSV"):
                data = data.dropna(subset=[text_column]).copy()
                data[text_column] = data[text_column].astype(str)

                predictions = data[text_column].apply(predict_sentiment)
                data["Sentiment"] = predictions.apply(lambda x: x[0])
                data["Compound Score"] = predictions.apply(lambda x: x[1])

                st.session_state.batch_results = data
                st.success(f"Analyzed {len(data)} rows.")

        except Exception as error:
            st.error(f"Could not read this CSV: {error}")

    if "batch_results" in st.session_state:
        results = st.session_state.batch_results
        st.dataframe(results, use_container_width=True)

        st.download_button(
            "⬇️ Download results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )

with tab3:
    st.subheader("Sentiment Overview")

    if st.session_state.history:
        history = pd.DataFrame(st.session_state.history)
        counts = history["Sentiment"].value_counts()

        col1, col2, col3 = st.columns(3)
        col1.metric("Texts Analyzed", len(history))
        col2.metric(
            "Positive",
            int((history["Sentiment"] == "Positive").sum())
        )
        col3.metric(
            "Negative",
            int((history["Sentiment"] == "Negative").sum())
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        counts.reindex(
            ["Positive", "Neutral", "Negative"],
            fill_value=0
        ).plot(kind="bar", ax=ax)

        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Texts")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Analysis History")
        st.dataframe(history, use_container_width=True)

        st.download_button(
            "⬇️ Download history",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_history.csv",
            mime="text/csv"
        )
    else:
        st.info("Analyze some text in the first tab to populate your dashboard.")

st.divider()
st.caption(
    "SentimentScope AI | Codec Technologies Internship Project | "
    "Python • Streamlit • NLTK • Pandas"
)