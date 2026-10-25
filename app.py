from pathlib import Path
from datetime import datetime
from collections import Counter
import io
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from wordcloud import WordCloud, STOPWORDS

from emotion_detector import detect_emotions

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "twitter_sentiment_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.joblib"

LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOJIS = {"Negative": "😞", "Neutral": "😐", "Positive": "😊"}

st.set_page_config(
    page_title="SentimentScope AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- MODEL --------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python train_model.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return joblib.load(METRICS_PATH)
    return {}


def predict_sentiment(model, text):
    probabilities = model.predict_proba([text])[0]
    classes = model.named_steps["classifier"].classes_

    best_index = int(np.argmax(probabilities))
    label_id = int(classes[best_index])
    sentiment = LABELS.get(label_id, "Neutral")

    probability_map = {
        LABELS.get(int(class_id), str(class_id)): float(probability)
        for class_id, probability in zip(classes, probabilities)
    }

    return sentiment, float(probabilities[best_index]), probability_map


def make_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def make_pdf(title, content):
    try:
        from reportlab.lib.pagesizes import letter # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet # type: ignore
        from reportlab.platypus import ( # type: ignore
            SimpleDocTemplate,
            Paragraph,
            Spacer,
        )
    except ImportError:
        return None

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "Generated on " + datetime.now().strftime("%Y-%m-%d %H:%M"),
            styles["Normal"],
        ),
        Spacer(1, 14),
    ]

    for line in content:
        safe_line = (
            str(line)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        story.extend([
            Paragraph(safe_line, styles["BodyText"]),
            Spacer(1, 8),
        ])

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -------------------- KEYWORDS --------------------

def extract_keywords(texts, top_n=15):
    stop_words = set(STOPWORDS)
    stop_words.update({
        "the", "and", "for", "that", "this", "with", "you", "your",
        "are", "was", "were", "have", "has", "had", "but", "not",
        "from", "they", "their", "its", "about", "just", "very",
        "can", "will", "would", "could", "should", "our", "all",
        "out", "get", "got", "one", "like", "really", "im",
        "dont", "didnt", "doesnt", "ive", "weve", "youre",
    })

    words = []

    for text in texts:
        if not isinstance(text, str):
            continue

        tokens = re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())
        words.extend(
            word for word in tokens
            if word not in stop_words and len(word) > 2
        )

    return Counter(words).most_common(top_n)


def create_wordcloud(text):
    cloud = WordCloud(
        width=1200,
        height=500,
        background_color="white",
        colormap="viridis",
        stopwords=set(STOPWORDS),
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# -------------------- STYLING --------------------

st.markdown("""
<style>
:root { --app-line: rgba(120,132,153,.24); }
.block-container { max-width: 1380px; }
h1, h2, h3, h4 { letter-spacing: -.025em; }
[data-testid="stMetric"] {
  background: transparent;
  padding: 17px 19px;
  border-radius: 15px;
  border: 1px solid var(--app-line);
  box-shadow: 0 8px 24px rgba(30,41,59,.06);
}
[data-testid="stMetricLabel"] { opacity: .76; font-weight: 600; }
[data-testid="stMetricValue"] { font-weight: 760; }
.hero {
  position: relative; overflow: hidden; padding: 30px 34px;
  border-radius: 22px; color: white; margin: 0 0 24px;
  background: linear-gradient(112deg, #121a2e 0%, #20245a 55%, #4932a2 100%);
  border: 1px solid rgba(167,139,250,.32);
  box-shadow: 0 18px 48px rgba(30,27,75,.18);
}
.hero:after {
  content: ""; position: absolute; right: -70px; top: -145px;
  width: 280px; height: 280px; border-radius: 50%;
  border: 1px solid rgba(255,255,255,.15);
  box-shadow: 0 0 0 28px rgba(255,255,255,.035), 0 0 0 58px rgba(255,255,255,.025);
  pointer-events: none;
}
.hero .eyebrow { color: #c4b5fd; font-size: .72rem; font-weight: 750; letter-spacing: .16em; margin-bottom: 8px; }
.hero h1 { color: #fff; margin: 0 0 8px; font-size: clamp(1.8rem, 3vw, 2.45rem); }
.hero p { color: #dce4ff; font-size: 1rem; margin: 0; max-width: 820px; }
.brand-lockup { display:flex; align-items:center; gap:12px; padding:8px 0 14px; }
.brand-icon { display:grid; place-items:center; width:40px; height:40px; border-radius:13px; color:#fff; font-weight:800; background:linear-gradient(135deg,#635bdb,#8b5cf6); box-shadow:0 7px 20px rgba(99,91,219,.25); }
.brand-title { font-size:1.12rem; font-weight:760; letter-spacing:-.02em; }
.brand-caption { font-size:.75rem; opacity:.7; margin-top:2px; }
section[data-testid="stSidebar"] [role="radiogroup"] { gap: .3rem; }
section[data-testid="stSidebar"] [role="radiogroup"] label {
  border-radius: 11px; padding: .47rem .6rem; transition: background .16s ease, border-color .16s ease;
  border: 1px solid transparent;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: color-mix(in srgb, #7c3aed 8%, transparent);
}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: color-mix(in srgb, #7c3aed 12%, transparent);
  border-color: color-mix(in srgb, #7c3aed 24%, transparent);
}
.stTextArea textarea, .stTextInput input, .stNumberInput input { border-radius: 11px; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  min-height: 42px; border-radius: 11px; font-weight: 650;
  transition: transform .16s ease, border-color .16s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 15px; }
[data-testid="stAlert"] { border-radius: 13px; }
@media (max-width: 768px) { .hero { padding: 23px 21px; border-radius: 17px; } }
</style>
""", unsafe_allow_html=True)


# -------------------- SIDEBAR BRAND --------------------

with st.sidebar:
    st.markdown("""
    <div class="brand-lockup">
      <div class="brand-icon">SS</div>
      <div><div class="brand-title">SentimentScope AI</div>
      <div class="brand-caption">TEXT ANALYTICS</div></div>
    </div>
    """, unsafe_allow_html=True)


PAGE_HEROES = {
    "🏠 Overview": ("SentimentScope AI", "A focused workspace for understanding sentiment, emotions, and language patterns.", "YOUR TEXT ANALYTICS WORKSPACE"),
    "🔎 Analyze Text": ("Analyze text", "Explore sentiment and emotion signals in a sentence, review, tweet, or comment.", "SINGLE-TEXT ANALYSIS"),
    "📁 Batch Analysis": ("Batch analysis", "Apply the sentiment model to a CSV and download the enriched results.", "ANALYZE AT SCALE"),
    "📈 Dashboard": ("Sentiment dashboard", "Track session predictions, confidence, and sentiment patterns.", "SESSION INSIGHTS"),
    "☁️ Word Cloud & Keywords": ("Word cloud & keywords", "Discover recurring terms and themes across your analyzed text.", "LANGUAGE PATTERNS"),
    "🧪 Model Evaluation": ("Model evaluation", "Review the saved performance metrics for your trained model.", "MODEL QUALITY"),
    "💬 Ask Your Data": ("Ask your data", "Get quick answers about sentiment counts and confidence in this session.", "SESSION ASSISTANT"),
}
# -------------------- LOAD DATA --------------------

try:
    model = load_model()
    metrics = load_metrics()
except Exception as error:
    st.error(f"Could not load the sentiment model: {error}")
    st.info("Check that your trained model files are inside the models folder.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

if "batch_result" not in st.session_state:
    st.session_state.batch_result = None


def render_page(page):
    with st.sidebar:
        st.divider()
        st.caption("MODEL")
        st.markdown("TF-IDF + Logistic Regression")
        st.caption("DATASET")
        st.markdown("TweetEval sentiment")
        st.divider()
        st.caption(f"SESSION · {len(st.session_state.get('history', [])):,} analyses")
        st.caption("Predictions are estimates, not facts.")

    hero_title, hero_subtitle, hero_eyebrow = PAGE_HEROES[page]
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{hero_eyebrow}</div>
        <h1>{hero_title}</h1>
        <p>{hero_subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

    # ==================================================
    # OVERVIEW
    # ==================================================

    if page == "🏠 Overview":
        st.subheader("At a glance")
        accuracy = metrics.get("test_accuracy")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training tweets", f'{metrics.get("train_size", 0):,}')
        col2.metric("Test tweets", f'{metrics.get("test_size", 0):,}')
        col3.metric("Test accuracy", f"{accuracy:.2%}" if accuracy is not None else "N/A")
        col4.metric("Session analyses", f'{len(st.session_state.history):,}')

        st.subheader("Choose where to start")
        card1, card2, card3 = st.columns(3)
        with card1.container(border=True, height="stretch"):
            st.subheader("🔎 Analyze text")
            st.write("Classify an individual message and review its sentiment probabilities.")
            st.caption("Best for a quick check")
        with card2.container(border=True, height="stretch"):
            st.subheader("📁 Analyze a CSV")
            st.write("Score a full dataset and export the results for further review.")
            st.caption("Best for larger datasets")
        with card3.container(border=True, height="stretch"):
            st.subheader("📈 Explore insights")
            st.write("Review session trends, common keywords, and model evaluation metrics.")
            st.caption("Best for understanding patterns")

        st.subheader("This session")
        history = st.session_state.history
        if history:
            current_counts = {
                label: sum(item["Sentiment"] == label for item in history)
                for label in ["Positive", "Neutral", "Negative"]
            }
            with st.container(border=True):
                summary_cols = st.columns(4)
                summary_cols[0].metric("Total analyses", f"{len(history):,}")
                summary_cols[1].metric("😊 Positive", f'{current_counts["Positive"]:,}')
                summary_cols[2].metric("😐 Neutral", f'{current_counts["Neutral"]:,}')
                summary_cols[3].metric("😞 Negative", f'{current_counts["Negative"]:,}')
        else:
            st.caption("Your session summary will appear here after the first analysis.")

        with st.expander("About this model"):
            st.write("Algorithm: TF-IDF features with Logistic Regression.")
            st.write("Training dataset: TweetEval sentiment.")
            st.write("Classes: Negative, Neutral, and Positive.")
            st.caption("Predictions are estimates. Sarcasm, slang, and missing context can affect results.")


    # ==================================================
    # ANALYZE TEXT AND EMOTIONS
    # ==================================================

    elif page == "🔎 Analyze Text":
        st.subheader("Analyze Text")
        st.write("Enter a tweet, review, comment, or sentence.")

        text_input = st.text_area(
            "Your text",
            placeholder="I love this product! It makes me so happy.",
            height=140,
        )

        if st.button("🔍 Analyze Sentiment and Emotion", type="primary"):
            if not text_input.strip():
                st.warning("Please enter some text first.")
            else:
                text_value = text_input.strip()

                sentiment, confidence, probabilities = predict_sentiment(
                    model, text_value
                )

                st.session_state.history.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Text": text_value,
                    "Sentiment": sentiment,
                    "Confidence (%)": round(confidence * 100, 2),
                })

                st.markdown(f"## {EMOJIS[sentiment]} {sentiment}")
                st.metric("Sentiment Model Confidence", f"{confidence:.1%}")
                st.caption("Confidence is a model estimate, not a guarantee.")

                st.subheader("Sentiment Probabilities")
                probability_df = pd.DataFrame({
                    "Sentiment": list(probabilities.keys()),
                    "Probability (%)": [
                        value * 100 for value in probabilities.values()
                    ],
                })
                st.bar_chart(
                    probability_df.set_index("Sentiment"),
                    height=250,
                )

                st.divider()
                st.subheader("🧠 Emotion Detection")

                with st.spinner("Loading emotion model and analyzing text..."):
                    try:
                        emotion_results = detect_emotions(text_value)

                        if emotion_results:
                            top_emotion = emotion_results[0]
                            st.markdown(
                                f"### {top_emotion['emotion']} "
                                f"({top_emotion['confidence']:.1%})"
                            )

                            emotion_df = pd.DataFrame(emotion_results)
                            emotion_df["Confidence (%)"] = (
                                emotion_df["confidence"] * 100
                            )

                            st.bar_chart(
                                emotion_df.set_index("emotion")["Confidence (%)"],
                                height=300,
                            )
                            st.dataframe(
                                emotion_df[["emotion", "Confidence (%)"]].round(2),
                                width="stretch",
                            )

                            st.caption(
                                "Emotion labels are pretrained-model predictions. "
                                "They do not establish how someone actually feels."
                            )

                    except Exception as error:
                        st.error(f"Emotion detection failed: {error}")
                        st.info(
                            "Check your internet connection and pretrained model setup."
                        )


    # ==================================================
    # BATCH ANALYSIS
    # ==================================================

    elif page == "📁 Batch Analysis":
        st.subheader("Batch Sentiment Analysis")
        st.write("Upload a CSV file containing a column of text.")

        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="batch_csv",
        )

        if uploaded is not None:
            try:
                dataframe = pd.read_csv(uploaded)

                if dataframe.empty:
                    st.warning("The uploaded CSV is empty.")
                else:
                    text_column = st.selectbox(
                        "Choose the text column",
                        dataframe.columns.tolist(),
                    )

                    if st.button("Analyze CSV", type="primary"):
                        result = dataframe.copy()
                        texts = result[text_column].fillna("").astype(str)

                        sentiments = []
                        confidences = []
                        progress = st.progress(0.0)
                        total = len(texts)

                        for index, text_value in enumerate(texts):
                            if not text_value.strip():
                                sentiments.append("Empty")
                                confidences.append(0.0)
                            else:
                                label, score, _ = predict_sentiment(
                                    model, text_value
                                )
                                sentiments.append(label)
                                confidences.append(round(score * 100, 2))

                            progress.progress((index + 1) / total)

                        result["Predicted Sentiment"] = sentiments
                        result["Confidence (%)"] = confidences
                        st.session_state.batch_result = result

                        # Add batch predictions to the shared session history so
                        # Dashboard, Ask Your Data, and session keywords can use them.
                        analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for text_value, label, confidence in zip(
                            texts, sentiments, confidences
                        ):
                            if text_value.strip() and label != "Empty":
                                st.session_state.history.append({
                                    "Time": analyzed_at,
                                    "Text": text_value,
                                    "Sentiment": label,
                                    "Confidence (%)": confidence,
                                })

                        st.success(
                            f"Analyzed {total:,} rows. Non-empty predictions were "
                            "added to the dashboard and Ask Your Data."
                        )

            except Exception as error:
                st.error(f"Could not process CSV: {error}")

        result = st.session_state.batch_result

        if result is not None:
            st.subheader("Analysis Results")
            st.dataframe(result, width="stretch")

            st.download_button(
                "⬇️ Download Results CSV",
                data=make_csv(result),
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
            )

            st.subheader("Sentiment Distribution")
            st.bar_chart(result["Predicted Sentiment"].value_counts())


    # ==================================================
    # PROFESSIONAL DASHBOARD
    # ==================================================

    elif page == "📈 Dashboard":
        history = pd.DataFrame(st.session_state.history)

        if history.empty:
            st.info("🔎 Analyze some text first to populate your dashboard.")
            st.write("Open **Analyze Text**, enter a sentence, and run the analysis.")
        else:
            total = len(history)
            positive = int((history["Sentiment"] == "Positive").sum())
            neutral = int((history["Sentiment"] == "Neutral").sum())
            negative = int((history["Sentiment"] == "Negative").sum())

            st.subheader("Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📋 Total Analyses", f"{total:,}")
            c2.metric("😊 Positive", f"{positive:,}")
            c3.metric("😐 Neutral", f"{neutral:,}")
            c4.metric("😞 Negative", f"{negative:,}")

            st.divider()

            st.subheader("📈 Sentiment Trends Over Time")

            trend = history.copy()
            trend["Time"] = pd.to_datetime(trend["Time"], errors="coerce")
            trend = trend.dropna(subset=["Time"]).sort_values("Time")

            if not trend.empty:
                trend["Analysis"] = range(1, len(trend) + 1)
                trend["Positive"] = (
                    trend["Sentiment"].eq("Positive").astype(int).cumsum()
                )
                trend["Neutral"] = (
                    trend["Sentiment"].eq("Neutral").astype(int).cumsum()
                )
                trend["Negative"] = (
                    trend["Sentiment"].eq("Negative").astype(int).cumsum()
                )

                trend_chart = trend.set_index("Analysis")[
                    ["Positive", "Neutral", "Negative"]
                ]
                st.line_chart(trend_chart, height=320)

                st.caption(
                    "Cumulative number of predictions in each category, "
                    "in the order they were analyzed during this session."
                )

            left, right = st.columns(2)

            with left:
                st.subheader("🎯 Sentiment Distribution")
                distribution = pd.DataFrame({
                    "Sentiment": ["Positive", "Neutral", "Negative"],
                    "Predictions": [positive, neutral, negative],
                })
                st.bar_chart(
                    distribution.set_index("Sentiment"),
                    height=280,
                )

            with right:
                st.subheader("📊 Prediction Confidence")
                confidence_data = history[["Confidence (%)"]].reset_index(drop=True)
                confidence_data.index = range(1, len(confidence_data) + 1)
                confidence_data.index.name = "Analysis"
                st.line_chart(confidence_data, height=280)

                average_confidence = history["Confidence (%)"].mean()
                st.metric("Average Confidence", f"{average_confidence:.2f}%")

            st.divider()
            st.subheader("🕒 Analysis History")

            st.dataframe(
                history.iloc[::-1],
                width="stretch",
                hide_index=True,
            )

            st.subheader("📥 Download Reports")
            download_col, pdf_col, clear_col = st.columns(3)

            with download_col:
                st.download_button(
                    "⬇️ History CSV",
                    data=make_csv(history),
                    file_name="sentiment_session_history.csv",
                    mime="text/csv",
                    width="stretch",
                )

            pdf_bytes = make_pdf(
                "SentimentScope AI - Session Report",
                [
                    f"Total analyses: {total}",
                    f"Positive predictions: {positive}",
                    f"Neutral predictions: {neutral}",
                    f"Negative predictions: {negative}",
                    f"Average confidence: {history['Confidence (%)'].mean():.2f}%",
                ],
            )

            with pdf_col:
                if pdf_bytes:
                    st.download_button(
                        "📄 PDF Report",
                        data=pdf_bytes,
                        file_name="sentiment_session_report.pdf",
                        mime="application/pdf",
                        width="stretch",
                    )
                else:
                    st.caption("Install reportlab to enable PDF reports.")

            with clear_col:
                if st.button("🗑️ Clear History", width="stretch"):
                    st.session_state.history = []
                    st.rerun()


    # ==================================================
    # WORD CLOUD AND KEYWORDS
    # ==================================================

    elif page == "☁️ Word Cloud & Keywords":
        st.subheader("☁️ Word Cloud & Trending Keywords")
        st.write(
            "Discover frequent words in your analyzed text or in an uploaded dataset."
        )

        source = st.radio(
            "Choose text source",
            ["Analyzed session", "Upload CSV"],
            horizontal=True,
        )

        collected_texts = []

        if source == "Analyzed session":
            collected_texts = [
                item.get("Text", "")
                for item in st.session_state.history
                if item.get("Text", "").strip()
            ]

            if not collected_texts:
                st.info("Analyze some text first, or select Upload CSV.")
        else:
            keyword_file = st.file_uploader(
                "Upload a CSV for keyword analysis",
                type=["csv"],
                key="keyword_csv",
            )

            if keyword_file is not None:
                try:
                    keyword_df = pd.read_csv(keyword_file)

                    if not keyword_df.empty:
                        keyword_column = st.selectbox(
                            "Select text column",
                            keyword_df.columns.tolist(),
                            key="keyword_column",
                        )
                        collected_texts = (
                            keyword_df[keyword_column]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )
                    else:
                        st.warning("The uploaded CSV is empty.")

                except Exception as error:
                    st.error(f"Could not read the CSV: {error}")

        if collected_texts:
            combined_text = " ".join(
                text for text in collected_texts if text.strip()
            )

            if combined_text.strip():
                st.metric("Texts Included", len(collected_texts))

                st.subheader("☁️ Word Cloud")

                try:
                    fig = create_wordcloud(combined_text)
                    st.pyplot(fig)
                    plt.close(fig)
                except ValueError:
                    st.info("Not enough usable words to generate a word cloud.")

                st.subheader("🔥 Trending Keywords")
                keyword_counts = extract_keywords(collected_texts, top_n=15)

                if keyword_counts:
                    keyword_df = pd.DataFrame(
                        keyword_counts,
                        columns=["Keyword", "Frequency"],
                    )

                    col1, col2 = st.columns([2, 3])

                    with col1:
                        st.dataframe(
                            keyword_df,
                            hide_index=True,
                            width="stretch",
                        )

                    with col2:
                        st.bar_chart(
                            keyword_df.set_index("Keyword"),
                            height=400,
                        )

                    st.download_button(
                        "⬇️ Download Keyword Frequencies CSV",
                        data=make_csv(keyword_df),
                        file_name="trending_keywords.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No meaningful keywords found. Try longer text.")


    # ==================================================
    # MODEL EVALUATION
    # ==================================================

    elif page == "🧪 Model Evaluation":
        st.subheader("🧪 Model Evaluation")

        if not metrics:
            st.warning("Evaluation metrics were not found.")
        else:
            col1, col2 = st.columns(2)

            col1.metric(
                "Validation Accuracy",
                f'{metrics.get("validation_accuracy", 0):.2%}',
            )
            col2.metric(
                "Test Accuracy",
                f'{metrics.get("test_accuracy", 0):.2%}',
            )

            st.subheader("Classification Report")
            report = metrics.get("classification_report", {})
            st.dataframe(
                pd.DataFrame(report).T.round(3),
                width="stretch",
            )

            matrix = metrics.get("confusion_matrix", [])
            labels = metrics.get(
                "labels",
                ["Negative", "Neutral", "Positive"],
            )

            if len(matrix):
                matrix_array = np.asarray(matrix, dtype=int)

                fig, ax = plt.subplots(figsize=(7, 5))
                display = ConfusionMatrixDisplay(
                    confusion_matrix=matrix_array,
                    display_labels=labels,
                )
                display.plot(ax=ax, cmap="Blues", colorbar=False)
                ax.set_title("TweetEval Test Set Confusion Matrix")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.caption(
                "Metrics describe performance on the held-out test dataset. "
                "Real-world results may differ."
            )


    # ==================================================
    # ASK YOUR DATA
    # ==================================================

    elif page == "💬 Ask Your Data":
        st.subheader("💬 Ask Your Data")

        st.caption(
            "This is a rule-based helper for the current session, "
            "not a generative AI chatbot."
        )

        question = st.text_input(
            "Ask a question",
            placeholder="How many positive predictions are there?",
        )

        if st.button("Get Answer", type="primary"):
            history = pd.DataFrame(st.session_state.history)
            q = question.lower().strip()
            sentiment_words = {
                "positive": "Positive",
                "negative": "Negative",
                "neutral": "Neutral",
            }
            requested_sentiment = next(
                (label for word, label in sentiment_words.items()
                 if re.search(rf"\b{word}\b", q)),
                None,
            )

            if not q:
                st.warning("Enter a question.")
            elif history.empty:
                st.info("Analyze text or run a batch analysis first.")
            elif "confidence" in q or "average" in q or "mean" in q:
                average = history["Confidence (%)"].mean()
                st.success(f"Average model confidence: {average:.2f}%.")
            elif requested_sentiment:
                count = int((history["Sentiment"] == requested_sentiment).sum())
                st.success(
                    f"{count} {requested_sentiment.lower()} prediction(s) "
                    "in this session."
                )
            elif any(word in q for word in ["how many", "total", "count"]):
                st.success(f"Total analyses in this session: {len(history)}.")
            else:
                st.info(
                    "Try asking about total analyses, a positive/neutral/negative "
                    "count, or average confidence."
                )


    # -------------------- FOOTER --------------------

    st.divider()
    st.caption(
        "SentimentScope AI | TweetEval-trained baseline model | "
        "Educational and research use"
    )

# -------------------- GROUPED NAVIGATION --------------------

def route_page(route):
    return lambda: render_page(route)


selected_page = st.navigation(
    {
        "Workspace": [
            st.Page(
                route_page("🏠 Overview"),
                title="Overview",
                url_path="overview",
                icon=":material/home:",
                default=True,
            ),
        ],
        "Analysis": [
            st.Page(
                route_page("🔎 Analyze Text"),
                title="Analyze text",
                url_path="analyze-text",
                icon=":material/search:",
            ),
            st.Page(
                route_page("📁 Batch Analysis"),
                title="Batch analysis",
                url_path="batch-analysis",
                icon=":material/upload_file:",
            ),
        ],
        "Insights": [
            st.Page(
                route_page("📈 Dashboard"),
                title="Dashboard",
                url_path="dashboard",
                icon=":material/bar_chart:",
            ),
            st.Page(
                route_page("☁️ Word Cloud & Keywords"),
                title="Keywords",
                url_path="keywords",
                icon=":material/cloud:",
            ),
            st.Page(
                route_page("🧪 Model Evaluation"),
                title="Model evaluation",
                url_path="model-evaluation",
                icon=":material/science:",
            ),
            st.Page(
                route_page("💬 Ask Your Data"),
                title="Ask your data",
                url_path="ask-your-data",
                icon=":material/forum:",
            ),
        ],
    },
    position="sidebar",
)
selected_page.run()
