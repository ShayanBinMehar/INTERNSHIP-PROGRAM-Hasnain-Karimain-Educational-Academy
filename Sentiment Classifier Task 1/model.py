"""Multi-class sentiment analysis system with training, evaluation, and prediction."""

import json
import os
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from utils import preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
TEST_SAMPLES_PATH = os.path.join(BASE_DIR, "test_samples.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
MODEL_METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")
RESULTS_PATH = os.path.join(BASE_DIR, "predictions.csv")
CV_FOLDS = 5

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


def build_word_char_features() -> FeatureUnion:
    """Combine word n-grams and character n-grams for richer text features."""
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 3),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    max_features=12000,
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """TF-IDF settings tuned for short review-style text."""
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,
    )


def build_tuning_pipelines() -> dict[str, Pipeline]:
    """Pipelines with hyperparameters tuned via grid search."""
    features = build_word_char_features()
    return {
        "logistic_regression": Pipeline(
            [
                ("features", features),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=8000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("features", build_word_char_features()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        LinearSVC(class_weight="balanced", random_state=42),
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def tune_pipeline(
    base_pipeline: Pipeline, param_grid: dict, X_train: pd.Series, y_train: pd.Series
) -> tuple[Pipeline, dict]:
    """Run grid search and return the best fitted pipeline."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def select_best_pipeline(X_train: pd.Series, y_train: pd.Series) -> tuple[Pipeline, str, float, dict]:
    """Pick the best model using tuned grid search and cross-validation."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    best_name = ""
    best_pipeline = None
    best_score = -1.0
    best_params: dict = {}

    tuning_configs = {
        "logistic_regression": {
            "classifier__C": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
        },
        "linear_svc": {
            "classifier__estimator__C": [0.1, 0.25, 0.5, 1.0, 2.0],
        },
    }

    print("\n🔍 Tuning models with 5-fold cross-validation (accuracy)...")
    for name, base_pipeline in build_tuning_pipelines().items():
        tuned_pipeline, params = tune_pipeline(
            base_pipeline, tuning_configs[name], X_train, y_train
        )
        scores = cross_val_score(
            tuned_pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        mean_score = scores.mean()
        print(
            f"   {name:22s} accuracy: {mean_score:.4f} (+/- {scores.std():.4f}) "
            f"params: {params}"
        )
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipeline = tuned_pipeline
            best_params = params

    print(f"\n✅ Selected model: {best_name} (CV accuracy: {best_score:.4f})")
    print(f"   Best params: {best_params}")
    return best_pipeline, best_name, best_score, best_params


def evaluate_holdout(
    pipeline: Pipeline, texts: pd.Series, labels: pd.Series, title: str
) -> dict:
    """Evaluate model on a labeled hold-out set."""
    y_pred = pipeline.predict(texts)
    metrics = {
        "accuracy": accuracy_score(labels, y_pred),
        "precision": precision_score(labels, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(labels, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(labels, y_pred, average="weighted", zero_division=0),
        "y_test": labels,
        "y_pred": y_pred,
    }
    print(f"\n📋 {title}")
    print(f"   Samples:   {len(labels)}")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    print("\n   Classification Report:")
    print(classification_report(labels, y_pred, zero_division=0))
    return metrics


def evaluate_external_test_set(pipeline: Pipeline, path: str = TEST_SAMPLES_PATH) -> dict | None:
    """Evaluate on held-out samples never used during training."""
    if not os.path.exists(path):
        print(f"\n⚠️  External test file not found: {path}")
        return None

    df = load_dataset(path)
    df["Processed_Text"] = df["Text"].apply(preprocess_text)
    return evaluate_holdout(
        pipeline,
        df["Processed_Text"],
        df["Sentiment"],
        "External Test Set (test_samples.csv)",
    )


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Train the sentiment classifier and return metrics."""
    print("\n📊 Training Dataset Overview:")
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

    pipeline, model_name, cv_accuracy, best_params = select_best_pipeline(X_train, y_train)
    print("\n🔄 Training tuned model on training split...")
    pipeline.fit(X_train, y_train)

    metrics = evaluate_holdout(
        pipeline, X_test, y_test, "Validation Split (20% hold-out from training data)"
    )
    metrics["model_name"] = model_name
    metrics["cv_accuracy"] = cv_accuracy
    metrics["best_params"] = best_params
    metrics["cv_f1"] = cv_accuracy

    print("\n🔄 Retraining on full training dataset for better generalization...")
    pipeline.fit(df["Processed_Text"], df["Sentiment"])

    external = evaluate_external_test_set(pipeline)
    if external:
        metrics["external"] = external

    metrics["training_samples"] = len(df)
    return pipeline, metrics


def plot_confusion_matrix(y_test, y_pred, title_suffix: str = "") -> None:
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
    suffix = f" – {title_suffix}" if title_suffix else ""
    plt.title(f"Confusion Matrix – Sentiment Classifier{suffix}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n📈 Confusion matrix saved to {save_path}")
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    plt.close()


def save_model(
    pipeline: Pipeline, path: str = MODEL_PATH, metrics: dict | None = None
) -> None:
    """Persist trained model and optional training metrics to disk."""
    joblib.dump(pipeline, path)
    print(f"💾 Model saved to {path}")

    if metrics is None:
        return

    metadata = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_file": os.path.basename(path),
        "model_name": metrics.get("model_name"),
        "best_params": metrics.get("best_params"),
        "cv_accuracy": round(metrics.get("cv_accuracy", 0), 4),
        "validation_accuracy": round(metrics.get("accuracy", 0), 4),
        "external_test_accuracy": round(
            metrics.get("external", {}).get("accuracy", 0), 4
        ),
        "training_samples": metrics.get("training_samples"),
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Training results saved to {MODEL_METADATA_PATH}")


def load_model_metadata(path: str = MODEL_METADATA_PATH) -> dict | None:
    """Load saved training metrics if available."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(path: str = MODEL_PATH) -> Pipeline:
    """Load a previously trained model."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Train the model first (menu option 7)."
        )
    return joblib.load(path)


def try_load_saved_model() -> Pipeline | None:
    """Load saved model if it exists."""
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model()


def print_saved_model_info() -> None:
    """Display saved model accuracy summary."""
    metadata = load_model_metadata()
    if metadata:
        print("\n📦 Saved model ready (no retraining needed)")
        print(f"   Saved at:              {metadata.get('saved_at', 'unknown')}")
        print(f"   Validation accuracy:   {metadata['validation_accuracy']:.2%}")
        print(f"   External test accuracy: {metadata['external_test_accuracy']:.2%}")
        print(f"   CV accuracy:           {metadata['cv_accuracy']:.2%}")
        print(f"   Training samples:      {metadata.get('training_samples', 'N/A')}")
    elif os.path.exists(MODEL_PATH):
        print("\n📦 Saved model loaded from model.pkl")


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
    pipeline = try_load_saved_model()

    print("=" * 55)
    print("  🎯 Multi-Class Sentiment Analysis System")
    print("  Categories: Positive 😊 | Negative 😡 | Neutral 😐 | Mixed 🤔")
    print("=" * 55)

    if pipeline is not None:
        print_saved_model_info()
    else:
        print("\n⚠️  No saved model found. Use option 7 to train first.")

    while True:
        print("\n─── Menu ───")
        print("  1. Predict single sentence")
        print("  2. Batch prediction")
        print("  3. View saved model info")
        print("  4. Evaluate on external test set (test_samples.csv)")
        print("  5. Evaluate & show confusion matrix")
        print("  6. Reload saved model")
        print("  7. Train model (only if you want to retrain)")
        print("  8. Exit")

        choice = input("\nSelect option (1-8): ").strip()

        if choice == "1":
            if pipeline is None:
                try:
                    pipeline = load_model()
                except FileNotFoundError:
                    print("❌ No model available. Train a model first (option 7).")
                    continue
            single_prediction_menu(pipeline)

        elif choice == "2":
            if pipeline is None:
                try:
                    pipeline = load_model()
                except FileNotFoundError:
                    print("❌ No model available. Train a model first (option 7).")
                    continue
            batch_prediction_menu(pipeline)

        elif choice == "3":
            if os.path.exists(MODEL_PATH):
                print_saved_model_info()
            else:
                print("❌ No saved model found.")

        elif choice == "4":
            if pipeline is None:
                try:
                    pipeline = load_model()
                except FileNotFoundError:
                    print("❌ No model available. Train a model first (option 7).")
                    continue
            metrics = evaluate_external_test_set(pipeline)
            if metrics:
                plot_confusion_matrix(
                    metrics["y_test"], metrics["y_pred"], "External Test Set"
                )

        elif choice == "5":
            if pipeline is None:
                try:
                    pipeline = load_model()
                except FileNotFoundError:
                    print("❌ No model available. Train a model first (option 7).")
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
            plot_confusion_matrix(y_test, y_pred, "Validation Split")

        elif choice == "6":
            try:
                pipeline = load_model()
                print("✅ Saved model reloaded successfully.")
                print_saved_model_info()
            except FileNotFoundError as e:
                print(f"❌ {e}")

        elif choice == "7":
            df = load_dataset()
            pipeline, metrics = train_model(df)
            save_model(pipeline, metrics=metrics)
            if metrics.get("external"):
                plot_confusion_matrix(
                    metrics["external"]["y_test"],
                    metrics["external"]["y_pred"],
                    "External Test Set",
                )
            else:
                plot_confusion_matrix(
                    metrics["y_test"], metrics["y_pred"], "Validation Split"
                )

        elif choice == "8":
            print("\n👋 Goodbye!")
            break

        else:
            print("⚠️  Invalid option. Please choose 1-8.")


if __name__ == "__main__":
    cli_menu()
