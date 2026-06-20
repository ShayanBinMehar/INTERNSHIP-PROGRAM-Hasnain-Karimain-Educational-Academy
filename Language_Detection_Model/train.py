"""
Language Detection Model - Training Script

Trains a multilingual text classifier using character-level TF-IDF features
and Multinomial Naive Bayes. Evaluates on a held-out test set and saves
the trained pipeline for inference.
"""

import os
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "confusion_matrix.png")

# Regex for symbols that do not carry language signal (keep letters from all scripts)
SYMBOL_PATTERN = re.compile(r"[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", re.UNICODE)


def preprocess_text(text: str) -> str:
    """
    Normalize text while preserving language-specific Unicode characters.

    - Converts to lowercase (safe for Latin scripts; Urdu is caseless)
    - Strips extra whitespace
    - Removes punctuation/symbols but keeps letters from all scripts
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower().strip()
    text = SYMBOL_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_preprocess(path: str) -> tuple[pd.Series, pd.Series]:
    """Load dataset CSV and apply preprocessing."""
    df = pd.read_csv(path)

    if "Text" not in df.columns or "Language" not in df.columns:
        raise ValueError("dataset.csv must contain 'Text' and 'Language' columns.")

    df = df.dropna(subset=["Text", "Language"])
    df["Text"] = df["Text"].astype(str).apply(preprocess_text)
    df = df[df["Text"].str.len() > 0]

    return df["Text"], df["Language"]


def build_pipeline() -> Pipeline:
    """
    Character n-gram TF-IDF + Multinomial Naive Bayes.

    Character features work well for short multilingual snippets and
    languages with non-Latin scripts (e.g. Urdu).
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )


def plot_confusion_matrix(y_true, y_pred, labels, save_path: str) -> None:
    """Save a labeled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Language Detection - Confusion Matrix")
    plt.xlabel("Predicted Language")
    plt.ylabel("True Language")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def compare_with_langdetect(texts: pd.Series, true_labels: pd.Series) -> None:
    """Optional comparison against the langdetect library on test samples."""
    try:
        from langdetect import DetectorFactory, detect
        from langdetect.lang_detect_exception import LangDetectException

        DetectorFactory.seed = 0

        # Map ISO 639-1 codes to our full language names
        code_to_language = {
            "en": "English",
            "ur": "Urdu",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
        }

        correct = 0
        total = 0

        for text, true_lang in zip(texts, true_labels):
            try:
                detected_code = detect(text)
                predicted = code_to_language.get(detected_code, detected_code)
                if predicted == true_lang:
                    correct += 1
                total += 1
            except LangDetectException:
                continue

        if total > 0:
            accuracy = correct / total
            print(f"\nlangdetect comparison (test set): {accuracy:.2%} ({correct}/{total})")
        else:
            print("\nlangdetect comparison: no valid predictions on test set.")

    except ImportError:
        print("\nlangdetect not installed — skipping comparison.")


def train() -> Pipeline:
    """Full training workflow: load data, train, evaluate, save."""
    print("=" * 60)
    print("Language Detection Model - Training")
    print("=" * 60)

    X, y = load_and_preprocess(DATASET_PATH)
    print(f"\nDataset loaded: {len(X)} samples, {y.nunique()} languages")
    print(f"Languages: {', '.join(sorted(y.unique()))}")
    print(f"\nSamples per language:\n{y.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    print("\nTraining Multinomial Naive Bayes with char TF-IDF (2-4 grams)...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    labels = sorted(y.unique())
    plot_confusion_matrix(y_test, y_pred, labels, CONFUSION_MATRIX_PATH)

    compare_with_langdetect(X_test, y_test)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Also save label order for predict.py convenience
    metadata = {"labels": labels, "accuracy": accuracy}
    joblib.dump(metadata, os.path.join(BASE_DIR, "model_metadata.pkl"))

    print("\nTraining complete.")
    return pipeline


if __name__ == "__main__":
    train()
