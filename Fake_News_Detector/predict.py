"""
Fake News Detection - Prediction Script

Single-text prediction with confidence, batch CSV processing,
and an interactive CLI menu.
"""

import os
import sys
import warnings

import joblib
import pandas as pd

from train_model import MODEL_PATH, preprocess_text

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.pkl")
MIN_TEXT_LENGTH = 5

SHORT_TEXT_WARNING = "Warning: Input is very short — prediction may be less reliable."


def load_model():
    """Load trained pipeline; exit if model file is missing."""
    if not os.path.exists(MODEL_PATH):
        print("Error: model.pkl not found. Run train_model.py first.")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def predict_news(model, text: str) -> dict:
    """
    Classify a news headline or article as Fake or Real.

    Returns label, confidence, per-class probabilities, and warnings.
    """
    raw_text = text
    cleaned = preprocess_text(text)
    warnings_list = []

    if len(cleaned) < MIN_TEXT_LENGTH:
        warnings_list.append(SHORT_TEXT_WARNING)

    if not cleaned:
        return {
            "input": raw_text,
            "prediction": "Unknown",
            "label_display": "Unable to classify",
            "confidence": 0.0,
            "probabilities": {},
            "warnings": ["Error: Empty or invalid input after preprocessing."],
        }

    proba = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    prob_map = {label: float(p) for label, p in zip(classes, proba)}
    prediction = max(prob_map, key=prob_map.get)
    confidence = prob_map[prediction]

    label_display = "Fake News" if prediction == "Fake" else "Real News"

    return {
        "input": raw_text,
        "prediction": prediction,
        "label_display": label_display,
        "confidence": confidence,
        "probabilities": prob_map,
        "warnings": warnings_list,
    }


def print_prediction(result: dict) -> None:
    """Pretty-print prediction output."""
    print(f"\nInput: {result['input'][:120]}{'...' if len(result['input']) > 120 else ''}")
    print(f"Prediction: {result['label_display']}")
    print(f"Confidence: {result['confidence']:.2%}")

    if result.get("probabilities"):
        print("\nClass probabilities:")
        for label, prob in sorted(result["probabilities"].items()):
            display = "Fake News" if label == "Fake" else "Real News"
            print(f"  {display}: {prob:.2%}")

    for warning in result.get("warnings", []):
        print(f"\n{warning}")


def interactive_prediction(model) -> None:
    """Prompt user for news text and show classification."""
    print("\n--- Single News Prediction ---")
    news = input("Enter news text: ").strip()
    if not news:
        print("No input provided.")
        return
    result = predict_news(model, news)
    print_prediction(result)


def batch_prediction(model) -> None:
    """Read texts from CSV and write predictions to output CSV."""
    print("\n--- Batch Prediction ---")
    input_path = input("Enter input CSV path (Enter for batch_input.csv): ").strip()
    if not input_path:
        input_path = os.path.join(BASE_DIR, "batch_input.csv")

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        print("Create a CSV with a 'Text' column, or any single text column.")
        return

    df = pd.read_csv(input_path)
    text_col = "Text" if "Text" in df.columns else df.columns[0]

    rows = []
    for text in df[text_col].astype(str):
        pred = predict_news(model, text)
        rows.append(
            {
                "Text": text,
                "Prediction": pred["label_display"],
                "Confidence": pred["confidence"],
                "Fake_Probability": pred["probabilities"].get("Fake", 0.0),
                "Real_Probability": pred["probabilities"].get("Real", 0.0),
            }
        )

    output_path = input("Enter output CSV path (Enter for batch_output.csv): ").strip()
    if not output_path:
        output_path = os.path.join(BASE_DIR, "batch_output.csv")

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path} ({len(rows)} rows)")


def run_demo_examples(model) -> None:
    """Run sample headlines through the model."""
    print("\n--- Demo Examples ---")
    samples = [
        "Breaking Scientists confirm water found on Mars",
        "Aliens landed in New York yesterday government hiding the truth",
        "Central bank announces interest rate decision following quarterly review",
        "Shocking leaked documents prove elections were controlled by robots",
        "Federal reserve reports steady job growth in latest employment statistics",
    ]

    for text in samples:
        result = predict_news(model, text)
        print(f"\n  Text: {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"  → {result['label_display']} ({result['confidence']:.2%})")


def show_menu() -> None:
    """Main CLI menu loop."""
    model = load_model()

    if os.path.exists(METADATA_PATH):
        meta = joblib.load(METADATA_PATH)
        best = meta.get("best_model", "Unknown")
        f1 = meta.get("best_f1", 0)
        print(f"Model loaded — best trainer: {best} (F1: {f1:.2%})")
    else:
        print("Model loaded.")

    while True:
        print("\n" + "=" * 42)
        print("  Fake News Detection System")
        print("=" * 42)
        print("  1. Predict single news text")
        print("  2. Batch prediction (CSV)")
        print("  3. Run demo examples")
        print("  4. Exit")
        print("=" * 42)

        choice = input("Select option (1-4): ").strip()

        if choice == "1":
            interactive_prediction(model)
        elif choice == "2":
            batch_prediction(model)
        elif choice == "3":
            run_demo_examples(model)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please enter 1-4.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = load_model()
        text = " ".join(sys.argv[1:])
        result = predict_news(model, text)
        print_prediction(result)
    else:
        show_menu()
