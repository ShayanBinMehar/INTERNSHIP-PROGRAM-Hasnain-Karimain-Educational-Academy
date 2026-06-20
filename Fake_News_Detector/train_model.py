"""
Fake News Detection - Training Script

Loads news text data, preprocesses with NLTK, trains TF-IDF + classifiers,
compares models, evaluates metrics, visualizes top features, and saves
the best pipeline for inference.
"""

import os
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.pkl")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance.png")
MODEL_COMPARISON_PATH = os.path.join(BASE_DIR, "model_comparison.png")

PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

_lemmatizer = None
_stop_words = None


def _ensure_nltk_data() -> None:
    """Download NLTK resources needed for preprocessing."""
    resources = ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4")
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


def _get_preprocessor():
    """Lazy-load lemmatizer and stopwords after NLTK data is available."""
    global _lemmatizer, _stop_words
    if _lemmatizer is None:
        _ensure_nltk_data()
        _lemmatizer = WordNetLemmatizer()
        _stop_words = set(stopwords.words("english"))
    return _lemmatizer, _stop_words


def preprocess_text(text: str) -> str:
    """
    Clean and normalize news text for classification.

    Handles short headlines, long articles, and noisy social-style text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    lemmatizer, stop_words = _get_preprocessor()

    text = text.lower().strip()
    text = URL_PATTERN.sub(" ", text)
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 1 and not token.isdigit()
    ]

    return " ".join(tokens)


def load_and_preprocess(path: str) -> tuple[pd.Series, pd.Series]:
    """Load dataset and apply text preprocessing."""
    df = pd.read_csv(path)

    if "Text" not in df.columns or "Label" not in df.columns:
        raise ValueError("dataset.csv must contain 'Text' and 'Label' columns.")

    df = df.dropna(subset=["Text", "Label"])
    df["Label"] = df["Label"].astype(str).str.strip().str.title()
    df = df[df["Label"].isin(["Fake", "Real"])]

    df["Text"] = df["Text"].astype(str).apply(preprocess_text)
    df = df[df["Text"].str.len() > 0]

    return df["Text"], df["Label"]


def build_pipeline(classifier_name: str = "logistic") -> Pipeline:
    """Build TF-IDF vectorizer + classifier pipeline."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    classifiers = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "naive_bayes": MultinomialNB(alpha=0.1),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    if classifier_name not in classifiers:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifiers[classifier_name]),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    """Compute classification metrics for a trained pipeline."""
    y_pred = pipeline.predict(X_test)

    return {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="Fake", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="Fake", zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label="Fake", zero_division=0),
        "y_pred": y_pred,
    }


def plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    """Save confusion matrix heatmap."""
    labels = ["Fake", "Real"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Fake News Detection - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_model_comparison(results: list[dict], save_path: str) -> None:
    """Bar chart comparing model metrics."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = [r["name"] for r in results]
    x = range(len(model_names))
    width = 0.2

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        offset = (i - 1.5) * width
        plt.bar([pos + offset for pos in x], values, width=width, label=metric.title())

    plt.xticks(list(x), model_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison - Fake News Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Model comparison saved to: {save_path}")


def plot_feature_importance(pipeline: Pipeline, save_path: str, top_n: int = 20) -> None:
    """Visualize top TF-IDF features associated with Fake vs Real."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(classifier, "coef_"):
        # Logistic Regression: positive coef -> Fake, negative -> Real
        coef = classifier.coef_[0]
        fake_indices = coef.argsort()[-top_n:][::-1]
        real_indices = coef.argsort()[:top_n]

        fake_words = [(feature_names[i], coef[i]) for i in fake_indices]
        real_words = [(feature_names[i], abs(coef[i])) for i in real_indices]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].barh(
            [w for w, _ in reversed(fake_words)],
            [v for _, v in reversed(fake_words)],
            color="#e74c3c",
        )
        axes[0].set_title(f"Top {top_n} Words → Fake")
        axes[0].set_xlabel("Coefficient")

        axes[1].barh(
            [w for w, _ in reversed(real_words)],
            [v for _, v in reversed(real_words)],
            color="#27ae60",
        )
        axes[1].set_title(f"Top {top_n} Words → Real")
        axes[1].set_xlabel("|Coefficient|")

        plt.suptitle("Most Important Words (Logistic Regression)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Feature importance saved to: {save_path}")
    else:
        print("Feature importance plot skipped (classifier has no coef_ attribute).")


def train() -> Pipeline:
    """Full training workflow."""
    print("=" * 60)
    print("Fake News Detection - Training")
    print("=" * 60)

    X, y = load_and_preprocess(DATASET_PATH)
    print(f"\nDataset loaded: {len(X)} samples")
    print(f"Label distribution:\n{y.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_configs = [
        ("Logistic Regression", "logistic"),
        ("Naive Bayes", "naive_bayes"),
        ("Random Forest", "random_forest"),
    ]

    results = []
    best_pipeline = None
    best_score = -1.0
    best_name = ""

    print("\nTraining and evaluating models...")
    for display_name, key in model_configs:
        pipeline = build_pipeline(key)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(display_name, pipeline, X_test, y_test)
        results.append(metrics)

        print(f"\n{display_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.2%}")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall:    {metrics['recall']:.2%}")
        print(f"  F1-score:  {metrics['f1']:.2%}")

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_pipeline = pipeline
            best_name = display_name
            best_pred = metrics["y_pred"]

    print(f"\nBest model (by F1): {best_name}")

    print("\nDetailed classification report (best model):")
    print(classification_report(y_test, best_pred, target_names=["Fake", "Real"]))

    plot_confusion_matrix(y_test, best_pred, CONFUSION_MATRIX_PATH)
    plot_model_comparison(results, MODEL_COMPARISON_PATH)
    plot_feature_importance(best_pipeline, FEATURE_IMPORTANCE_PATH)

    joblib.dump(best_pipeline, MODEL_PATH)
    metadata = {
        "best_model": best_name,
        "labels": ["Fake", "Real"],
        "metrics": {r["name"]: {k: r[k] for k in ("accuracy", "precision", "recall", "f1")} for r in results},
        "best_f1": best_score,
    }
    joblib.dump(metadata, METADATA_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Training complete.")
    return best_pipeline


if __name__ == "__main__":
    train()
