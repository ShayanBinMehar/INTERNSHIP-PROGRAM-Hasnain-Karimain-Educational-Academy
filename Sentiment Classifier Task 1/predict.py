"""Predict sentiment using the saved model (no training required)."""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model import (
    display_prediction,
    load_model,
    load_model_metadata,
    predict_sentiment,
    print_saved_model_info,
)


def main() -> None:
    metadata = load_model_metadata()
    if metadata is None and not __import__("os").path.exists("model.pkl"):
        print("❌ No saved model found. Run 'python model.py' and train first (option 7).")
        sys.exit(1)

    print("=" * 55)
    print("  Sentiment Prediction (saved model)")
    print("=" * 55)
    print_saved_model_info()

    pipeline = load_model()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        display_prediction(predict_sentiment(pipeline, text))
        return

    print("\nEnter a sentence to classify (or 'quit' to exit):")
    while True:
        text = input("\n> ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            print("👋 Goodbye!")
            break
        if not text:
            print("⚠️  Please enter a sentence.")
            continue
        display_prediction(predict_sentiment(pipeline, text))


if __name__ == "__main__":
    main()
