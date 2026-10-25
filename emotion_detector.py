from transformers import pipeline

# A pretrained model that classifies text into six emotions.
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

_classifier = None


def load_emotion_model():
    """Load the pretrained emotion model once when needed."""
    global _classifier

    if _classifier is None:
        _classifier = pipeline(
            task="text-classification",
            model=MODEL_NAME,
            top_k=None,
            truncation=True,
        )

    return _classifier


def detect_emotions(text):
    """Return emotion labels and their model confidence scores."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide some text for emotion detection.")

    classifier = load_emotion_model()
    results = classifier(text.strip())[0]

    # Sort emotions from highest to lowest confidence.
    results = sorted(
        results,
        key=lambda item: item["score"],
        reverse=True,
    )

    return [
        {
            "emotion": item["label"].capitalize(),
            "confidence": float(item["score"]),
        }
        for item in results
    ]