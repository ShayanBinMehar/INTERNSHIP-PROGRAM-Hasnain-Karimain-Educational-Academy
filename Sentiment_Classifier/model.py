"""Multi-class sentiment analysis system with training, evaluation, and prediction."""

import os
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
from sklearn.pipeline import Pipeline

from utils import preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "predictions.csv")

SENTIMENT_LABELS = ["Positive", "Negative", "Neutral", "Mixed"]
SENTIMENT_EMOJI = {
    "Positive": "😊",
    "Negative": "😡",
    "Neutral": "😐",
    "Mixed": "🤔",
}


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the labeled dataset."""
    df = pd.read_csv(path)
    required_cols = {"Text", "Sentiment"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")
    invalid = set(df["Sentiment"].unique()) - set(SENTIMENT_LABELS)
    if invalid:
        raise ValueError(f"Invalid sentiment labels found: {invalid}")
    return df


def build_pipeline() -> Pipeline:
    """Create TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Train the sentiment classifier and return metrics."""
    print("\n📊 Dataset Overview:")
    print(f"   Total samples: {len(df)}")
    print(df["Sentiment"].value_counts().to_string(header=False))

    df = df.copy()
    df["Processed_Text"] = df["Text"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["Processed_Text"],
        df["Sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["Sentiment"],
    )

    pipeline = build_pipeline()
    print("\n🔄 Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "y_test": y_test,
        "y_pred": y_pred,
    }

    print("\n✅ Model Evaluation (80/20 split):")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return pipeline, metrics


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Display confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=SENTIMENT_LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SENTIMENT_LABELS,
        yticklabels=SENTIMENT_LABELS,
    )
    plt.title("Confusion Matrix – Sentiment Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n📈 Confusion matrix saved to {save_path}")
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    plt.close()


def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    """Persist trained model to disk."""
    joblib.dump(pipeline, path)
    print(f"💾 Model saved to {path}")


def load_model(path: str = MODEL_PATH) -> Pipeline:
    """Load a previously trained model."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Train the model first (menu option 1)."
        )
    return joblib.load(path)


def predict_sentiment(pipeline: Pipeline, text: str) -> dict:
    """Predict sentiment for a single sentence with probabilities."""
    processed = preprocess_text(text)
    prediction = pipeline.predict([processed])[0]
    probabilities = pipeline.predict_proba([processed])[0]
    prob_dict = {
        label: round(prob, 4)
        for label, prob in zip(pipeline.classes_, probabilities)
    }
    return {
        "text": text,
        "sentiment": prediction,
        "emoji": SENTIMENT_EMOJI.get(prediction, ""),
        "probabilities": prob_dict,
    }


def predict_batch(pipeline: Pipeline, texts: list[str]) -> list[dict]:
    """Predict sentiment for multiple sentences."""
    return [predict_sentiment(pipeline, text) for text in texts]


def export_results(results: list[dict], path: str = RESULTS_PATH) -> None:
    """Export prediction results to CSV."""
    rows = []
    for r in results:
        row = {
            "Text": r["text"],
            "Predicted_Sentiment": r["sentiment"],
            "Confidence": max(r["probabilities"].values()),
        }
        row.update({f"Prob_{k}": v for k, v in r["probabilities"].items()})
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"📁 Results exported to {path}")


def display_prediction(result: dict) -> None:
    """Pretty-print a single prediction result."""
    print(f"\n   Input:  \"{result['text']}\"")
    print(
        f"   Predicted Sentiment: {result['sentiment']} "
        f"{result['emoji']}"
    )
    print("   Probabilities:")
    for label, prob in sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(prob * 20)
        print(f"      {label:10s} {prob:.2%}  {bar}")


def single_prediction_menu(pipeline: Pipeline) -> None:
    """Interactive single-sentence prediction."""
    text = input("\nEnter your sentence: ").strip()
    if not text:
        print("⚠️  Empty input. Please enter a sentence.")
        return
    result = predict_sentiment(pipeline, text)
    display_prediction(result)


def batch_prediction_menu(pipeline: Pipeline) -> None:
    """Batch prediction for multiple sentences."""
    print("\nEnter sentences (one per line). Type 'done' when finished:")
    texts = []
    while True:
        line = input("  > ").strip()
        if line.lower() == "done":
            break
        if line:
            texts.append(line)

    if not texts:
        print("⚠️  No sentences provided.")
        return

    results = predict_batch(pipeline, texts)
    print(f"\n📊 Batch Results ({len(results)} sentences):")
    for i, result in enumerate(results, 1):
        print(f"\n   [{i}] {result['text']}")
        print(
            f"       → {result['sentiment']} {result['emoji']} "
            f"(confidence: {max(result['probabilities'].values()):.2%})"
        )

    export = input("\nExport results to CSV? (y/n): ").strip().lower()
    if export == "y":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(BASE_DIR, f"predictions_{timestamp}.csv")
        export_results(results, path)


def cli_menu() -> None:
    """Main CLI menu for the sentiment analysis system."""
    pipeline = None

    print("=" * 55)
    print("  🎯 Multi-Class Sentiment Analysis System")
    print("  Categories: Positive 😊 | Negative 😡 | Neutral 😐 | Mixed 🤔")
    print("=" * 55)

    while True:
        print("\n─── Menu ───")
        print("  1. Train model")
        print("  2. Load saved model")
        print("  3. Predict single sentence")
        print("  4. Batch prediction")
        print("  5. Evaluate & show confusion matrix")
        print("  6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == "1":
            df = load_dataset()
            pipeline, metrics = train_model(df)
            save_model(pipeline)
            plot_confusion_matrix(metrics["y_test"], metrics["y_pred"])

        elif choice == "2":
            try:
                pipeline = load_model()
                print("✅ Model loaded successfully.")
            except FileNotFoundError as e:
                print(f"❌ {e}")

        elif choice == "3":
            if pipeline is None:
                try:
                    pipeline = load_model()
                except FileNotFoundError:
                    print("❌ No model available. Train or load a model first.")
                    continue
            single_prediction_menu(pipeline)

        elif choice == "4":
            if pipeline is None:
                try:
                    pipeline = load_model()
                except FileNotFoundError:
                    print("❌ No model available. Train or load a model first.")
                    continue
            batch_prediction_menu(pipeline)

        elif choice == "5":
            if pipeline is None:
                try:
                    pipeline = load_model()
                except FileNotFoundError:
                    print("❌ No model available. Train or load a model first.")
                    continue
            df = load_dataset()
            df["Processed_Text"] = df["Text"].apply(preprocess_text)
            _, X_test, _, y_test = train_test_split(
                df["Processed_Text"],
                df["Sentiment"],
                test_size=0.2,
                random_state=42,
                stratify=df["Sentiment"],
            )
            y_pred = pipeline.predict(X_test)
            plot_confusion_matrix(y_test, y_pred)

        elif choice == "6":
            print("\n👋 Goodbye!")
            break

        else:
            print("⚠️  Invalid option. Please choose 1-6.")


if __name__ == "__main__":
    cli_menu()
