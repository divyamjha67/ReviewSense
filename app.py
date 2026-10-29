"""ReviewSense: a small, inspectable sentiment analysis workbench."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from reviewsense.io import parse_reviews_csv, scored_csv
from reviewsense.service import load_scorer


st.set_page_config(
    page_title="ReviewSense | Review sentiment",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Static presentation only. Review text is rendered through native Streamlit elements.
st.html("""
<style>
    .stApp { background: #f7f8f2; }
    .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1230px; }
    h1 { letter-spacing: -0.055em; font-weight: 750 !important; }
    h2, h3 { letter-spacing: -0.025em; }
    [data-testid="stSidebar"] { border-right: 1px solid #dce6db; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #dce6db; border-radius: 10px;
        padding: 1.15rem; min-height: 112px;
    }
    [data-testid="stMetricValue"] { color: #193d34; }
    [data-testid="stCaptionContainer"] { color: #61766b; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 7px; }
    [data-testid="stTextArea"] textarea { line-height: 1.7; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #dce6db; }
    @media (max-width: 1200px) {
        .block-container { padding-left: 1.2rem !important; padding-right: 1.2rem !important; }
    }
    @media (max-width: 1100px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 1rem !important;
            align-items: stretch !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 100% !important;
        }
        [data-testid="stMetricValue"] { font-size: 2rem !important; }
        [data-testid="stMetricValue"] > div {
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: nowrap !important;
        }
        h1, h2, h3 { overflow-wrap: normal !important; word-break: normal !important; }
    }
    @media (max-width: 700px) {
        .block-container { padding-top: 1.5rem; padding-left: 1rem !important; padding-right: 1rem !important; }
        h1 { font-size: 2.2rem !important; }
    }
</style>
""")

EXAMPLES = {
    "Positive example": "The food was delicious and the staff were friendly. I would happily come back.",
    "Negative example": "The service was slow and the food was cold. I was very disappointed.",
}

DEMO_REVIEWS = [
    ("demo-01", "The food was delicious and the service was excellent."),
    ("demo-02", "The product broke after one day and customer support was terrible."),
    ("demo-03", "A wonderful movie with a great cast and an interesting story."),
    ("demo-04", "The food was cold and the staff were rude."),
    ("demo-05", "Good quality and easy to use. I am happy with this purchase."),
    ("demo-06", "A boring movie with poor acting and a disappointing ending."),
]


def demo_csv() -> bytes:
    """Six authored examples for the interface, not an evaluation dataset."""
    return pd.DataFrame(DEMO_REVIEWS, columns=["review_id", "review"]).to_csv(index=False).encode("utf-8")


@st.cache_resource(show_spinner=False)
def get_scorer():
    # Cache the trusted fitted artifact, never uploaded review payloads.
    return load_scorer()


def populate_example(text: str) -> None:
    st.session_state["review_text"] = text


def heading(eyebrow: str, title: str, description: str) -> None:
    st.caption(eyebrow)
    st.title(title)
    st.write(description)


def single_review(scorer) -> None:
    heading(
        "01 / SINGLE REVIEW",
        "Find the sentiment in a sentence.",
        "Paste an English review to explore the sentiment learned by a simple text classification model.",
    )
    st.write("")
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("What did the customer say?")
            st.caption("Try an example, or use a review of your own.")
            examples = st.columns(2)
            for column, (label, text) in zip(examples, EXAMPLES.items()):
                column.button(label, on_click=populate_example, args=(text,), width="stretch")
            review = st.text_area(
                "Review text",
                key="review_text",
                placeholder="The service was excellent and I loved the food...",
                height=190,
                max_chars=10000,
                help="English review text. Up to 10,000 characters. Neutral sentiment is not a trained class.",
            )
            analyze = st.button("Analyze sentiment", type="primary", width="stretch")
            st.caption("A new edit clears the previous result. Click Analyze sentiment to score the current text.")
    with right:
        with st.container(border=True):
            st.caption("MODEL RESULT")
            if not analyze:
                st.subheader("A small signal. A clearer view.")
                st.write("Your prediction will appear here after you analyze a review.")
                st.divider()
                st.caption("The model learns word and phrase patterns from labeled review sentences. It does not understand every context or emotion.")
            else:
                try:
                    prediction = scorer.predict([review])[0]
                except ValueError as error:
                    st.error(str(error))
                    return
                probability = prediction["positive_probability"]
                if probability is None or prediction["sentiment"] == "Not scored":
                    st.info("No vocabulary match")
                    st.write("The model could not match this text to its learned vocabulary. Try a fuller English review.")
                    st.caption("Not scored is an input-coverage status, not a neutral sentiment label.")
                else:
                    sentiment = prediction["sentiment"]
                    st.subheader(f"{sentiment} sentiment")
                    st.metric("Positive-class score", f"{probability:.1%}")
                    st.progress(min(1.0, max(0.0, float(probability))))
                    st.caption("Closer to 100% favors Positive; closer to 0% favors Negative. The classification cutoff is 50%.")
                    st.divider()
                    st.write("Treat this as a model estimate, not a certainty.")
                    st.caption("Scores are not certified confidence levels. Sarcasm, mixed opinions, and unfamiliar wording can be misclassified.")
    st.write("")
    with st.container(border=True):
        st.caption("BUILT TO BE UNDERSTOOD")
        first, second, third = st.columns(3)
        first.markdown("**Text to numbers**\n\nTF-IDF represents words and short phrases.")
        second.markdown("**A simple classifier**\n\nLogistic regression separates two sentiment classes.")
        third.markdown("**Evidence, not guesses**\n\nThe model report shows held-out results and mistakes.")


def batch_reviews(scorer) -> None:
    heading(
        "02 / BATCH REVIEW",
        "Turn a review file into a summary.",
        "Upload a CSV, inspect the sentiment split, and download every scored review.",
    )
    with st.container(border=True):
        left, right = st.columns([1.4, 1], gap="large")
        with left:
            st.subheader("Add your reviews")
            uploaded = st.file_uploader("Review CSV", type=["csv"], help="UTF-8 CSV, at most 1,000 reviews and 1,000,000 bytes.")
            st.caption("Required column: review. Optional column: review_id. Invalid files are rejected as a whole.")
        with right:
            st.subheader("Start with a sample")
            st.write("Download six authored example reviews to try the upload workflow.")
            st.download_button("Download example CSV", data=demo_csv(), file_name="reviewsense-examples.csv", mime="text/csv", on_click="ignore", width="stretch")
            st.caption("These examples are for demonstration, not model evaluation.")
        analyze = st.button("Analyze CSV", type="primary", disabled=uploaded is None)
    if not analyze:
        st.info("Choose a CSV and click Analyze CSV to build the summary.")
        return
    if uploaded is None:
        return
    if uploaded.size > 1_000_000:
        st.error("The CSV must be at most 1,000,000 bytes.")
        return
    try:
        reviews = parse_reviews_csv(uploaded.getvalue())
        with st.spinner("Analyzing your reviews..."):
            predictions = scorer.predict(reviews["review"].tolist())
        result = pd.concat([reviews.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
        download = scored_csv(reviews, predictions)
    except ValueError as error:
        st.error(str(error))
        return
    st.write("")
    st.subheader("Your sentiment summary")
    positive = int((result["sentiment"] == "Positive").sum())
    negative = int((result["sentiment"] == "Negative").sum())
    unscored = int((result["sentiment"] == "Not scored").sum())
    columns = st.columns(4)
    for column, label, number in zip(columns, ["Reviews", "Positive", "Negative", "Not scored"], [len(result), positive, negative, unscored]):
        column.metric(label, number)
    chart, context = st.columns([1, 1], gap="large")
    with chart:
        counts = pd.DataFrame({"Sentiment": ["Positive", "Negative"], "Reviews": [positive, negative]}).set_index("Sentiment")
        st.bar_chart(counts, color="#16765D", height=245)
    with context:
        with st.container(border=True):
            st.markdown("**Read the summary in context**")
            scored = positive + negative
            if scored:
                st.write(f"{positive / scored:.1%} of the {scored:,} scored reviews are classified as Positive.")
            else:
                st.write("None of these reviews matched the learned vocabulary, so no sentiment ratio is available.")
            st.caption("This is a prediction summary of the uploaded text, not a measured customer satisfaction rate.")
            if unscored:
                st.caption(f"{unscored:,} reviews had no vocabulary match and are excluded from the sentiment ratio.")
            st.caption("This application processes uploads in memory and does not write them to a review database.")
    st.dataframe(
        result,
        hide_index=True,
        width="stretch",
        column_config={
            "review_id": st.column_config.TextColumn("Review ID"),
            "review": st.column_config.TextColumn("Review", width="large"),
            "sentiment": st.column_config.TextColumn("Sentiment"),
            "positive_probability": st.column_config.NumberColumn("Positive-class score", format="percent"),
            "vocabulary_matches": st.column_config.NumberColumn("Vocabulary matches", format="%d"),
        },
    )
    st.download_button("Download scored CSV", data=download, file_name="reviewsense-scored.csv", mime="text/csv", on_click="ignore", type="primary")
    st.caption("The export includes every review, its model score, and its sentiment or coverage status. Formula-like CSV cells are escaped for safer spreadsheet use.")


def model_report(scorer) -> None:
    report = scorer.report
    heading(
        "03 / MODEL REPORT",
        "Know what the model can do.",
        "A documented benchmark, a held-out test set, and visible mistakes. The numbers below come from the saved training report.",
    )
    metrics = report["metrics"]
    columns = st.columns(4)
    for column, title, key in zip(columns, ["Accuracy", "Positive precision", "Positive recall", "Positive F1"], ["accuracy", "precision", "recall", "f1"]):
        column.metric(title, f"{metrics[key]:.1%}")
    st.caption(f"{report['test_rows']:,} held-out sentences · cutoff 50% · ROC-AUC {metrics['roc_auc']:.3f}")
    st.caption(report["metric_scope"])
    st.caption(f"Test sentences with zero vocabulary matches: {report['test_zero_vocabulary']:,}. The app returns Not scored for these; the benchmark metrics above include the underlying classifier's predictions for all held-out rows.")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("How the model was selected")
            st.write("Compare complete text pipelines on training data. The final test set is kept separate.")
            comparison = pd.DataFrame(report["comparison"])
            comparison = comparison[["name", "cv_f1_mean", "cv_f1_std"]].rename(columns={"name": "Candidate", "cv_f1_mean": "Mean CV F1", "cv_f1_std": "Fold SD"})
            st.dataframe(comparison, hide_index=True, width="stretch")
            st.caption(f"Selected: {report['model_name']}. Fold spread is not a confidence interval.")
            st.caption("The prior-only baseline predicts the training class prior. Its hard class prediction chooses the more frequent class; the recorded training folds favor Negative, so positive-class CV F1 is 0. This is not a random-guess baseline.")
    with right:
        with st.container(border=True):
            st.subheader("Confusion matrix")
            matrix = pd.DataFrame(metrics["confusion_matrix"], index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"])
            st.dataframe(matrix, width="stretch")
            st.caption("Diagonal cells are correct classifications. Off-diagonal cells are mistakes.")
    with st.container(border=True):
        st.subheader("The dataset and its boundaries")
        st.write("The UCI Sentiment Labelled Sentences benchmark contains short review sentences from Amazon, Yelp, and IMDb. It is curated for binary Positive/Negative sentiment; neutral sentiment is not a trained class.")
        columns = st.columns(3)
        columns[0].metric("Source rows", f"{report['rows_raw']:,}")
        columns[1].metric("After duplicate checks", f"{report['rows_clean']:,}")
        columns[2].metric("Training sentences", f"{report['train_rows']:,}")
        st.caption(f"Duplicate rows removed: {report['duplicates_removed']:,}. Conflicting-label rows removed: {report['conflicting_removed']:,}. Rows without alphabetic words removed: {report['invalid_removed']:,}.")
        st.markdown("[Dataset and attribution](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)")
        st.caption("Balanced, selected sentences are not a representative sample of all incoming business reviews. Scores may not transfer to long reviews, different languages, sarcasm, or a new domain.")
    st.subheader("Performance by source")
    domains = pd.DataFrame(report["domain_metrics"])[["source", "rows", "accuracy", "f1"]]
    st.dataframe(domains.rename(columns={"source": "Source", "rows": "Test sentences", "accuracy": "Accuracy", "f1": "Positive F1"}), hide_index=True, width="stretch")
    st.caption("These are source subsets of the held-out benchmark, not separate external-domain validation datasets.")
    st.write("")
    st.subheader("Patterns the model learned")
    st.caption("Largest signed logistic-regression coefficients. These are global word/phrase associations, not causal explanations or a complete explanation of one review.")
    left, right = st.columns(2, gap="large")
    for column, category, title in [(left, "positive", "Positive associations"), (right, "negative", "Negative associations")]:
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                terms = pd.DataFrame(report["top_terms"].get(category, []))
                if terms.empty:
                    st.caption("No coefficient summary is available for this model.")
                else:
                    st.dataframe(terms.rename(columns={"term": "Word or phrase", "weight": "Coefficient"}), hide_index=True, width="stretch")
    st.subheader("Learn from the mistakes")
    st.caption("Recorded misclassified test sentences. Inspect them to understand ambiguity and failure modes; do not tune repeatedly against this seen test set.")
    errors = pd.DataFrame(report["error_examples"])
    if errors.empty:
        st.info("No misclassified examples were recorded in this report.")
    else:
        st.dataframe(errors, hide_index=True, width="stretch")
    with st.expander("What this version does not establish"):
        st.write("There is no trained Neutral class, automatic sarcasm detector, multilingual guarantee, or measured business impact. The Positive-class score is an uncalibrated model estimate. A model can confidently make mistakes, especially on unfamiliar wording or a different review population.")
        st.write("Use this as a small NLP classification workbench. Operational use needs representative labeled reviews, appropriate validation, privacy controls, and monitoring.")


def main() -> None:
    with st.sidebar:
        st.title("ReviewSense")
        st.caption("WORDS INTO SIGNALS")
        st.divider()
        section = st.radio("Workspace", ["Single review", "Batch review", "Model report"], label_visibility="collapsed")
        st.divider()
        st.markdown("**Simple NLP, made visible.**")
        st.caption("TF-IDF + logistic regression")
        st.caption("Two classes: Positive and Negative")
        st.caption("English review sentences")
        st.write("")
        st.caption("No paid AI API required.")
    try:
        scorer = get_scorer()
    except (FileNotFoundError, ValueError) as error:
        heading("MODEL SETUP", "The model is not ready yet.", "Train the project model, then restart the app.")
        st.error(str(error))
        st.code("python scripts/download_data.py\npython -m reviewsense.train", language="shell")
        st.stop()
    with st.sidebar:
        st.caption(f"Model v{scorer.report['model_version']} · Ready")
    if section == "Single review":
        single_review(scorer)
    elif section == "Batch review":
        batch_reviews(scorer)
    else:
        model_report(scorer)
    st.divider()
    st.caption("ReviewSense · A documented binary sentiment benchmark and local review analysis application.")


if __name__ == "__main__":
    main()
