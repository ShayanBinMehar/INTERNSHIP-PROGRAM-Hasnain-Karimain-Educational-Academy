"""Non-interactive demo run of the text preprocessing pipeline."""

import csv
from pathlib import Path

from preprocessing import TextPreprocessor

print("=" * 60)
print("  TEXT PREPROCESSING PIPELINE — DEMO RUN")
print("=" * 60)

pipeline = TextPreprocessor(enable_logging=True)

# Task example
demo = "I am loving the new AI tools! They are running amazingly well."
print("\n--- Demo Example ---")
print("Input :", demo)
lemmas = pipeline.preprocess(demo)
print("Output:", lemmas)

# Raw vs processed comparison
print("\n--- Raw vs Processed Comparison ---")
pipeline.compare_raw_vs_processed(demo)

# Edge cases
print("--- Edge Cases ---")
for label, text in [
    ("Empty input", ""),
    ("Special chars", "Hello!!! @#$ 123 test..."),
    ("Mixed case", "Running BETTER than before!!!"),
]:
    result = pipeline.preprocess(text, return_stems=False)
    print(f"  {label:15} -> {result}")

# Batch processing
print("\n--- Batch Processing (dataset.csv) ---")
texts = []
with open("dataset.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        texts.append(row["text"])

out_path = Path("processed_output.csv")
rows = []
for idx, text in enumerate(texts, 1):
    r = pipeline.preprocess(text, return_stems=True)
    rows.append({
        "id": idx,
        "raw_text": r["raw"],
        "lemmatized": " ".join(r["lemmas"]),
        "stemmed": " ".join(r["stems"]),
    })
    print(f"  [{idx}] {r['lemmas']}")

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "raw_text", "lemmatized", "stemmed"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nBatch output saved to: {out_path.resolve()}")
print("=" * 60)
