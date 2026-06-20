"""
Language Detection Model - Prediction Script

Provides single-text prediction, top-2 language probabilities,
batch CSV processing, langdetect comparison, and a simple CLI menu.
"""

import os
import sys
import warnings

import joblib
import pandas as pd

from train import MODEL_PATH, preprocess_text

warnings.filterwarnings("ignore")

# Use UTF-8 on Windows so Urdu and other scripts display correctly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def safe_print(text: str) -> None:
    """Print text safely on Windows consoles that lack UTF-8 support."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        ))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.pkl")

# Minimum character length for reliable prediction
MIN_TEXT_LENGTH = 3

# Message shown when input is too short or ambiguous
SHORT_TEXT_WARNING = (
    "Warning: Input is very short — prediction may be less reliable."
)
MIXED_LANGUAGE_WARNING = (
    "Warning: Text may contain mixed languages — showing top 2 candidates."
)


def load_model():
    """Load trained pipeline; exit with helpful message if missing."""
    if not os.path.exists(MODEL_PATH):
        print("Error: model.pkl not found. Run train.py first.")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def is_likely_mixed(text: str) -> bool:
    """
    Basic heuristic: if top-2 predicted probabilities are close,
    the text may mix languages. Actual check happens after prediction.
    """
    return False  # placeholder; decided after scoring


def predict_language(model, text: str) -> dict:
    """
    Predict language for a single text input.

    Returns predicted label, top-2 languages with probabilities,
    and any warnings for edge cases.
    """
    raw_text = text
    text = preprocess_text(text)
    warnings_list = []

    if len(text) < MIN_TEXT_LENGTH:
        warnings_list.append(SHORT_TEXT_WARNING)

    if not text:
        return {
            "input": raw_text,
            "predicted_language": "Unknown",
            "confidence": 0.0,
            "top_2": [],
            "warnings": ["Error: Empty or invalid input after preprocessing."],
        }

    proba = model.predict_proba([text])[0]
    classes = model.classes_
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

    top_language, top_conf = ranked[0]
    top_2 = [(lang, float(conf)) for lang, conf in ranked[:2]]

    # Mixed-language hint: second candidate within 15% of the top score
    if len(ranked) > 1 and ranked[1][1] >= ranked[0][1] * 0.85:
        warnings_list.append(MIXED_LANGUAGE_WARNING)

    return {
        "input": raw_text,
        "predicted_language": top_language,
        "confidence": float(top_conf),
        "top_2": top_2,
        "warnings": warnings_list,
    }


def predict_with_langdetect(text: str) -> str | None:
    """Run langdetect for side-by-side comparison."""
    try:
        from langdetect import DetectorFactory, detect
        from langdetect.lang_detect_exception import LangDetectException

        DetectorFactory.seed = 0
        code_to_language = {
            "en": "English",
            "ur": "Urdu",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
        }
        code = detect(text)
        return code_to_language.get(code, code)
    except Exception:
        return None


def print_prediction(result: dict, show_langdetect: bool = True) -> None:
    """Pretty-print a prediction result."""
    safe_print(f"\nInput: {result['input']!r}")
    safe_print(f"Predicted Language: {result['predicted_language']}")
    safe_print(f"Confidence: {result['confidence']:.2%}")

    if result["top_2"]:
        safe_print("\nTop 2 probable languages:")
        for i, (lang, conf) in enumerate(result["top_2"], start=1):
            safe_print(f"  {i}. {lang} ({conf:.2%})")

    for warning in result.get("warnings", []):
        safe_print(f"\n{warning}")

    if show_langdetect:
        ld_result = predict_with_langdetect(result["input"])
        if ld_result:
            safe_print(f"\nlangdetect prediction: {ld_result}")
        else:
            safe_print("\nlangdetect: unavailable or could not detect language.")


def interactive_prediction(model) -> None:
    """Prompt user for text and display prediction."""
    print("\n--- Single Text Prediction ---")
    text = input("Enter text: ").strip()
    if not text:
        print("No input provided.")
        return
    result = predict_language(model, text)
    print_prediction(result)


def batch_prediction(model) -> None:
    """Read texts from a CSV and write predictions to an output CSV."""
    print("\n--- Batch Prediction ---")
    input_path = input("Enter input CSV path: ").strip()
    if not input_path:
        input_path = os.path.join(BASE_DIR, "batch_input.csv")

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    text_col = "Text" if "Text" in df.columns else df.columns[0]

    results = []
    for text in df[text_col].astype(str):
        pred = predict_language(model, text)
        results.append(
            {
                "Text": text,
                "Predicted_Language": pred["predicted_language"],
                "Confidence": pred["confidence"],
                "Second_Language": pred["top_2"][1][0] if len(pred["top_2"]) > 1 else "",
                "Second_Confidence": pred["top_2"][1][1] if len(pred["top_2"]) > 1 else 0.0,
            }
        )

    output_path = input("Enter output CSV path (Enter for batch_output.csv): ").strip()
    if not output_path:
        output_path = os.path.join(BASE_DIR, "batch_output.csv")

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path} ({len(results)} rows)")


def compare_sample_texts(model) -> None:
    """Run predefined examples through both our model and langdetect."""
    print("\n--- Model vs langdetect Comparison ---")
    samples = [
        ("How are you?", "English"),
        ("آپ کیسے ہیں؟", "Urdu"),
        ("¿Cómo estás?", "Spanish"),
        ("Bonjour, comment allez-vous?", "French"),
        ("Wie geht es dir?", "German"),
        ("Come stai oggi?", "Italian"),
        ("Hello bonjour", "Mixed"),
    ]

    print(f"\n{'Text':<35} {'True':<10} {'Our Model':<12} {'langdetect':<12} {'Match'}")
    print("-" * 85)

    for text, true_label in samples:
        pred = predict_language(model, text)
        ld = predict_with_langdetect(text) or "N/A"
        match = "Y" if pred["predicted_language"] == true_label or true_label == "Mixed" else "N"
        display_text = text[:32] + "..." if len(text) > 32 else text
        safe_print(
            f"{display_text:<35} {true_label:<10} "
            f"{pred['predicted_language']:<12} {ld:<12} {match}"
        )


def show_menu() -> None:
    """Main CLI menu loop."""
    model = load_model()

    if os.path.exists(METADATA_PATH):
        meta = joblib.load(METADATA_PATH)
        print(f"Model loaded (training accuracy: {meta.get('accuracy', 'N/A'):.2%})")
    else:
        print("Model loaded.")

    while True:
        print("\n" + "=" * 40)
        print("  Language Detection System")
        print("=" * 40)
        print("  1. Predict single text")
        print("  2. Batch prediction (CSV)")
        print("  3. Compare with langdetect")
        print("  4. Exit")
        print("=" * 40)

        choice = input("Select option (1-4): ").strip()

        if choice == "1":
            interactive_prediction(model)
        elif choice == "2":
            batch_prediction(model)
        elif choice == "3":
            compare_sample_texts(model)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please enter 1-4.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Quick CLI: python predict.py "some text here"
        model = load_model()
        text = " ".join(sys.argv[1:])
        result = predict_language(model, text)
        print_prediction(result)
    else:
        show_menu()
