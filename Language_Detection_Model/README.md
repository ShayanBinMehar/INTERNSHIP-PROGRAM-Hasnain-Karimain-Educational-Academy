# Language Detection Model

A machine learning-based language detection system that identifies the language of multilingual text using character-level TF-IDF features and Multinomial Naive Bayes classification.

## Supported Languages

| Language | Script |
|----------|--------|
| English  | Latin  |
| Urdu     | Arabic (Nastaliq) |
| Spanish  | Latin  |
| French   | Latin  |
| German   | Latin  |
| Italian  | Latin  |

## Project Structure

```
Language_Detection_Model/
├── dataset.csv           # Labeled training data (~180 sentences)
├── train.py              # Training, evaluation, and model export
├── predict.py            # CLI prediction system
├── model.pkl             # Saved model (generated after training)
├── model_metadata.pkl    # Training metadata (generated after training)
├── confusion_matrix.png  # Evaluation visualization (generated after training)
├── requirements.txt
└── README.md
```

## Setup

```bash
cd Language_Detection_Model
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python train.py
```

This will:

- Load and preprocess `dataset.csv`
- Split data 80/20 (stratified)
- Train a char n-gram TF-IDF + Naive Bayes pipeline
- Print accuracy, classification report, and confusion matrix
- Compare against `langdetect` on the test set
- Save `model.pkl` and `confusion_matrix.png`

### 2. Predict Language

**Interactive CLI menu:**

```bash
python predict.py
```

**Single text from command line:**

```bash
python predict.py "Bonjour, comment allez-vous?"
```

**Example output:**

```
Input: 'Bonjour, comment allez-vous?'
Predicted Language: French
Confidence: 94.32%

Top 2 probable languages:
  1. French (94.32%)
  2. Italian (3.15%)

langdetect prediction: French
```

### 3. Batch Prediction

Create a CSV with a `Text` column (or use the first column), then use menu option **2** in `predict.py`. Results are written to `batch_output.csv` by default.

## Technical Details

### Preprocessing

- Lowercase conversion
- Whitespace normalization
- Symbol removal while **preserving** Unicode letters (including Urdu/Arabic script ranges)
- Empty inputs handled gracefully

### Feature Engineering

```python
TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)
```

Character n-grams capture language-specific patterns and work well for short text and non-Latin scripts.

### Model

- **Multinomial Naive Bayes** (`alpha=0.1`) — fast, effective for text classification
- Wrapped in a scikit-learn `Pipeline` for clean train/inference

### Evaluation Metrics

- Accuracy
- Per-class precision, recall, F1 (classification report)
- Confusion matrix heatmap (`confusion_matrix.png`)

### Edge Cases

- **Short text:** warns when input has fewer than 3 characters after preprocessing
- **Mixed language:** warns when top-2 predictions are within 15% confidence of each other

## Bonus Features

- Top 2 language predictions with probability scores
- Comparison with `langdetect` library
- Model persistence via `joblib`
- Batch CSV input/output
- CLI menu system
- Confusion matrix visualization with matplotlib/seaborn

## Dataset

The included `dataset.csv` contains **180 labeled sentences** (30 per language). You can extend it with more examples to improve accuracy on specific domains.

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- langdetect (optional, for comparison)

## Author

AI Intern Task 3 — Language Detection Model
