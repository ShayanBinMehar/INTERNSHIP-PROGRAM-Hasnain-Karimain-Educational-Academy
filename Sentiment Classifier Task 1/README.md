# Multi-Class Sentiment Analysis System

A Python-based NLP system that classifies text into four sentiment categories:

| Sentiment | Emoji | Description |
|-----------|-------|-------------|
| Positive  | 😊    | Expresses favorable or happy opinions |
| Negative  | 😡    | Expresses unfavorable or angry opinions |
| Neutral   | 😐    | Neither positive nor negative |
| Mixed     | 🤔    | Contains both positive and negative opinions |

## Project Structure

```
Sentiment Classifier Task 1/
├── dataset.csv           # 800 labeled training sentences (200 per class)
├── test_samples.csv      # 80 held-out test sentences (20 per class)
├── model.py              # Training, evaluation, prediction & CLI
├── predict.py            # Quick prediction script (uses saved model)
├── utils.py              # Text preprocessing utilities
├── build_datasets.py     # Script to rebuild training/test CSV files
├── requirements.txt      # Python dependencies
├── model.pkl             # Saved trained model (ready to use)
├── model_metadata.json   # Saved accuracy results & training info
├── confusion_matrix.png  # Confusion matrix from last evaluation
└── README.md
```

## Setup

```bash
cd "Sentiment Classifier Task 1"
pip install -r requirements.txt
```

> **Note:** On first run, NLTK data (`punkt`, `stopwords`, `wordnet`, etc.) is downloaded automatically.

---

## Quick Start — Predict Without Training

A trained model is already saved in `model.pkl`. You do **not** need to retrain to make predictions.

### Option 1: Quick predict script (recommended)

Single sentence:

```bash
python predict.py "I love the design but hate the battery life"
```

Interactive mode:

```bash
python predict.py
```

### Option 2: Full CLI menu

```bash
python model.py
```

The saved model loads automatically on startup. Use **option 1** or **2** to predict.

### Example output

```
Enter your sentence: I love the design but hate the battery life

Predicted Sentiment: Mixed 🤔
Probabilities:
   Mixed      92.99%  ██████████████████
   Positive    4.29%
   Negative    2.23%
   Neutral     0.48%
```

---

## Saved Model & Results

| File | Description |
|------|-------------|
| `model.pkl` | Trained classifier pipeline (~1 MB) |
| `model_metadata.json` | Accuracy metrics and training configuration |

### Current model performance

| Metric | Score |
|--------|-------|
| Validation accuracy | **86.25%** |
| External test accuracy | **86.25%** |
| Cross-validation accuracy | **80.78%** |
| Training samples | **800** (200 per class) |
| Selected model | Logistic Regression (`C=5.0`) |

> Keep `model.pkl`, `model_metadata.json`, and `utils.py` together. The preprocessing in `utils.py` must match what was used during training.

---

## CLI Menu (`python model.py`)

| Option | Action |
|--------|--------|
| **1** | Predict single sentence |
| **2** | Batch prediction (optionally export to CSV) |
| **3** | View saved model info |
| **4** | Evaluate on external test set (`test_samples.csv`) |
| **5** | Evaluate & show confusion matrix |
| **6** | Reload saved model |
| **7** | Train model *(only if you want to retrain)* |
| **8** | Exit |

---

## Retraining (Optional)

Only retrain if you update `dataset.csv` or want to experiment with new settings.

```bash
python model.py
# Select option 7 — Train model
```

Training will:
1. Compare and tune **Logistic Regression** and **Linear SVC** via grid search
2. Evaluate on a 20% validation split
3. Retrain on the full dataset
4. Evaluate on the held-out `test_samples.csv`
5. Save `model.pkl`, `model_metadata.json`, and `confusion_matrix.png`

To rebuild the dataset from scratch:

```bash
python build_datasets.py
```

---

## Technical Details

### Data Preprocessing (`utils.py`)
- Lowercasing and punctuation removal
- Tokenization and lemmatization (WordNet)
- Minimal stopword removal — keeps sentiment words (*but*, *not*, *very*) and content words needed for **Neutral** detection

### Feature Engineering
- **Word TF-IDF** — unigrams, bigrams, and trigrams (up to 20,000 features)
- **Character TF-IDF** — character n-grams (2–5) for robust pattern matching
- Combined via `FeatureUnion`

### Model
- **Logistic Regression** (LBFGS, class-weight balanced) — selected after grid search
- **Linear SVC** (calibrated) — compared during training
- 5-fold stratified cross-validation for model selection
- Hyperparameter tuning via `GridSearchCV` (optimizes accuracy)
- 80/20 stratified train-validation split for reporting
- Final model retrained on all 800 training samples before saving

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (per class and weighted)
- Confusion matrix visualization
- Separate external evaluation on `test_samples.csv` (never seen during training)

---

## Dataset

| File | Samples | Purpose |
|------|---------|---------|
| `dataset.csv` | 800 (200 per class) | Training data |
| `test_samples.csv` | 80 (20 per class) | Held-out external test set |

Training and test sets have **no overlapping sentences**.

---

## Author

AI Intern – Task 1: Multi-Class Sentiment Analysis
