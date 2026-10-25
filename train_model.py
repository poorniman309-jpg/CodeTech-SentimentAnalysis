from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def load_data(text_filename, label_filename):
    text_path = DATA_DIR / text_filename
    label_path = DATA_DIR / label_filename

    if not text_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Dataset file missing: {text_path} or {label_path}"
        )

    texts = text_path.read_text(encoding="utf-8").splitlines()

    labels = [
        int(value.strip())
        for value in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    if len(texts) != len(labels):
        raise ValueError(
            f"Different numbers of texts and labels: "
            f"{text_filename}"
        )

    return texts, labels


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("       SENTIMENTSCOPE AI - MODEL TRAINING")
    print("=" * 50)

    print("\nLoading TweetEval dataset...")

    X_train, y_train = load_data(
        "train_text.txt",
        "train_labels.txt",
    )

    X_val, y_val = load_data(
        "val_text.txt",
        "val_labels.txt",
    )

    X_test, y_test = load_data(
        "test_text.txt",
        "test_labels.txt",
    )

    print(f"Training tweets:   {len(X_train):,}")
    print(f"Validation tweets: {len(X_val):,}")
    print(f"Testing tweets:    {len(X_test):,}")

    print("\nCreating machine-learning pipeline...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=100_000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining model. Please wait...")

    model.fit(X_train, y_train)

    print("\nEvaluating model...")

    validation_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    validation_accuracy = accuracy_score(
        y_val,
        validation_predictions,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    print(
        f"\nValidation accuracy: "
        f"{validation_accuracy:.2%}"
    )

    print(f"Test accuracy: {test_accuracy:.2%}")

    print("\nClassification report:")

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1, 2],
        target_names=[
            LABEL_NAMES[0],
            LABEL_NAMES[1],
            LABEL_NAMES[2],
        ],
        output_dict=True,
        zero_division=0,
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1, 2],
            target_names=[
                LABEL_NAMES[0],
                LABEL_NAMES[1],
                LABEL_NAMES[2],
            ],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1, 2],
    ).tolist()

    model_path = MODEL_DIR / "twitter_sentiment_model.joblib"
    metrics_path = MODEL_DIR / "model_metrics.joblib"

    joblib.dump(model, model_path)

    metrics = {
        "model_name": "TF-IDF + Logistic Regression",
        "dataset": "TweetEval sentiment",
        "validation_accuracy": float(validation_accuracy),
        "test_accuracy": float(test_accuracy),
        "classification_report": report,
        "confusion_matrix": matrix,
        "labels": [
            LABEL_NAMES[0],
            LABEL_NAMES[1],
            LABEL_NAMES[2],
        ],
        "train_size": len(X_train),
        "validation_size": len(X_val),
        "test_size": len(X_test),
    }

    joblib.dump(metrics, metrics_path)

    print("\nModel saved:", model_path)
    print("Metrics saved:", metrics_path)
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()