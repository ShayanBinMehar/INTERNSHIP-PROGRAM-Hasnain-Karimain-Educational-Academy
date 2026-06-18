"""
Text Preprocessing Pipeline — CLI entry point.

Menu:
  1. Process single text input
  2. Batch process from CSV/text file
  3. Compare raw vs processed (with stemming)
  0. Exit
"""

import csv
import os
import sys
from pathlib import Path

from preprocessing import TextPreprocessor

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "dataset.csv"
DEFAULT_OUTPUT = BASE_DIR / "processed_output.csv"


def _create_pipeline() -> TextPreprocessor:
    """Build preprocessor with user preferences."""
    print("\n--- Pipeline Options ---")
    remove_nums = input("Remove numbers? (y/n) [y]: ").strip().lower() != "n"
    use_nltk_sw = input("Use NLTK stopwords? (y/n) [y]: ").strip().lower() != "n"
    enable_log = input("Enable step logging? (y/n) [y]: ").strip().lower() != "n"
    return TextPreprocessor(
        remove_numbers=remove_nums,
        use_nltk_stopwords=use_nltk_sw,
        enable_logging=enable_log,
    )


def process_single_text(pipeline: TextPreprocessor) -> None:
    """Option 1: preprocess a single sentence from user input."""
    print("\n--- Single Text Processing ---")
    text = input("Enter text to preprocess: ").strip()

    if not text:
        print("Error: Empty input. Please enter some text.")
        return

    result = pipeline.preprocess(text, return_stems=True)
    print("\n--- Results ---")
    print(f"Processed (lemmatized): {result['lemmas']}")
    print(f"Stemmed comparison    : {result['stems']}")


def process_batch(pipeline: TextPreprocessor) -> None:
    """Option 2: load sentences from CSV or plain text file and save output."""
    print("\n--- Batch Processing ---")
    default_hint = str(DEFAULT_DATASET)
    file_path = input(f"Enter file path [{default_hint}]: ").strip() or default_hint
    path = Path(file_path)

    if not path.exists():
        print(f"Error: File not found — {path}")
        return

    texts: list[str] = []

    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "text" not in (reader.fieldnames or []):
                print("Error: CSV must contain a 'text' column.")
                return
            for row in reader:
                texts.append(row.get("text", "").strip())
    else:
        with open(path, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    if not texts:
        print("Error: No text entries found in file.")
        return

    print(f"Loaded {len(texts)} sentence(s). Processing...")

    output_path = input(f"Save output to [{DEFAULT_OUTPUT}]: ").strip() or str(DEFAULT_OUTPUT)
    out = Path(output_path)

    rows = []
    for idx, text in enumerate(texts, start=1):
        result = pipeline.preprocess(text, return_stems=True)
        rows.append({
            "id": idx,
            "raw_text": result["raw"],
            "cleaned_text": result["cleaned"],
            "tokens": " ".join(result["tokens"]),
            "without_stopwords": " ".join(result["without_stopwords"]),
            "lemmatized": " ".join(result["lemmas"]),
            "stemmed": " ".join(result["stems"]),
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "raw_text", "cleaned_text", "tokens",
                        "without_stopwords", "lemmatized", "stemmed"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBatch complete. {len(rows)} row(s) saved to: {out}")


def compare_text(pipeline: TextPreprocessor) -> None:
    """Option 3: detailed raw vs processed comparison with stemming."""
    print("\n--- Raw vs Processed Comparison ---")
    text = input("Enter text to compare: ").strip()

    if not text:
        print("Error: Empty input.")
        return

    pipeline.compare_raw_vs_processed(text)


def run_demo(pipeline: TextPreprocessor) -> None:
    """Run the task example to verify pipeline output."""
    demo = "I am loving the new AI tools! They are running amazingly well."
    print("\n--- Demo (Task Example) ---")
    print(f"Input : {demo}")
    lemmas = pipeline.preprocess(demo)
    print(f"Output: {lemmas}")


def print_menu() -> None:
    print("\n" + "=" * 50)
    print("  TEXT PREPROCESSING PIPELINE")
    print("=" * 50)
    print("  1. Process single text")
    print("  2. Batch processing (CSV / text file)")
    print("  3. Compare raw vs processed (+ stemming)")
    print("  4. Run demo example")
    print("  0. Exit")
    print("=" * 50)


def main() -> None:
    print("Initializing NLTK resources (first run may take a moment)...")
    pipeline = _create_pipeline()

    actions = {
        "1": process_single_text,
        "2": process_batch,
        "3": compare_text,
        "4": run_demo,
    }

    while True:
        print_menu()
        choice = input("Select option: ").strip()

        if choice == "0":
            print("Goodbye!")
            sys.exit(0)

        action = actions.get(choice)
        if action:
            try:
                action(pipeline)
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
            except Exception as exc:
                print(f"Error: {exc}")
        else:
            print("Invalid option. Please choose 0–4.")


if __name__ == "__main__":
    main()
