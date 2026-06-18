# Text Preprocessing Pipeline

A modular, reusable NLP preprocessing system that cleans raw text, tokenizes with custom regex logic, removes stopwords, and lemmatizes tokens — ready for machine learning workflows.

## Features

- **Text cleaning** — lowercase, punctuation/special character removal, optional number stripping, whitespace normalization
- **Custom tokenization** — regex-based word extraction (`re.findall`), not default library splitting
- **Stopword removal** — NLTK English stopwords or a built-in custom list
- **Lemmatization** — POS-aware WordNet lemmatization (`running` → `run`, `tools` → `tool`)
- **Stemming comparison** — Porter stemmer output alongside lemmas
- **Logging** — step-by-step pipeline output
- **CLI menu** — single text, batch processing, comparison mode, demo

## Project Structure

```
Text_Preprocessing_Pipeline/
├── main.py              # CLI entry point
├── preprocessing.py     # Core pipeline (class + convenience function)
├── dataset.csv          # Sample batch input
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

```bash
cd Text_Preprocessing_Pipeline
pip install -r requirements.txt
```

NLTK data (stopwords, WordNet, POS tagger) is downloaded automatically on first run.

## Usage

### CLI

```bash
python main.py
```

| Option | Description |
|--------|-------------|
| 1 | Process a single sentence via `input()` |
| 2 | Batch process `dataset.csv` (or any CSV with a `text` column / plain `.txt` file) |
| 3 | Compare raw vs processed text with lemma/stem diff |
| 4 | Run the task demo example |
| 0 | Exit |

### Programmatic

```python
from preprocessing import preprocess, TextPreprocessor

# Quick one-liner
tokens = preprocess("I am loving the new AI tools!")
# ['love', 'new', 'ai', 'tool']

# Full control with logging and comparison
pipeline = TextPreprocessor(enable_logging=True)
result = pipeline.preprocess(
    "I am loving the new AI tools! They are running amazingly well.",
    return_stems=True,
)
print(result["lemmas"])   # ['love', 'new', 'ai', 'tool', 'run', 'amazingly', 'well']
print(result["stems"])    # Porter stem equivalents
```

### Batch output

Option 2 writes `processed_output.csv` with columns:

`id`, `raw_text`, `cleaned_text`, `tokens`, `without_stopwords`, `lemmatized`, `stemmed`

## Example

**Input:**
```
I am loving the new AI tools! They are running amazingly well.
```

**Output (lemmatized):**
```
['love', 'new', 'ai', 'tool', 'run', 'amazingly', 'well']
```

## Edge Cases Handled

- Empty or whitespace-only input
- Mixed case and special characters
- Punctuation-heavy sentences
- Batch files with missing/blank rows

## Requirements

- Python 3.8+
- NLTK 3.8+
- Standard library: `re`, `string`, `logging`, `csv`
