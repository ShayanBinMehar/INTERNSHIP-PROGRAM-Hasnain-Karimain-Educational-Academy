# Fake News Detection System

A machine learning-based system that classifies news headlines and articles as **Fake** or **Real** using NLP preprocessing, TF-IDF features, and scikit-learn classifiers.

## Project Structure

```
Fake_News_Detector/
├── dataset.csv              # Labeled news samples (Text, Label)
├── train_model.py           # Preprocessing, training, evaluation, model saving
├── predict.py               # Interactive CLI and batch prediction
├── model.pkl                # Saved best model (generated after training)
├── model_metadata.pkl       # Training metrics and model info
├── requirements.txt         # Python dependencies
├── confusion_matrix.png     # Evaluation plot (generated)
├── model_comparison.png     # Model comparison chart (generated)
└── feature_importance.png   # Top words for Fake vs Real (generated)
```

## Setup

```bash
cd Fake_News_Detector
pip install -r requirements.txt
python train_model.py
```

Training downloads required NLTK data automatically (stopwords, tokenizer, WordNet).

## Usage

### Interactive CLI

```bash
python predict.py
```

Menu options:
1. **Single prediction** — enter any headline or article
2. **Batch prediction** — CSV in → CSV out with confidence scores
3. **Demo examples** — run sample texts through the model

### Quick command-line prediction

```bash
python predict.py "Breaking: Scientists confirm water found on Mars"
```

Example output:

```
Prediction: Real News
Confidence: 87.42%
```

### Batch prediction

Create a CSV with a `Text` column (or any single text column), then use menu option 2. Output includes `Prediction`, `Confidence`, `Fake_Probability`, and `Real_Probability`.

## How It Works

### 1. Data Preprocessing
- Lowercase conversion
- URL and punctuation removal
- Tokenization
- Stopword removal
- Lemmatization (NLTK WordNet)

### 2. Feature Engineering
- **TfidfVectorizer** with unigrams and bigrams (max 5000 features)

### 3. Model Training
Three classifiers are trained and compared:
- Logistic Regression (primary)
- Multinomial Naive Bayes
- Random Forest

The model with the highest **F1-score** on the test set is saved to `model.pkl`.

### 4. Evaluation (80/20 split)
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Dataset

`dataset.csv` contains 200+ labeled samples with columns:

| Text | Label |
|------|-------|
| Government confirms new policy reform... | Real |
| Aliens landed in New York yesterday! | Fake |

You can replace this with the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by merging `True.csv` and `Fake.csv` into a single file with `Text` and `Label` columns.

## Example Predictions

| Input | Expected |
|-------|----------|
| Scientists confirm water found on Mars | Real News |
| Aliens landed in New York yesterday | Fake News |
| Central bank announces interest rate decision | Real News |
| Shocking miracle cure hidden by big pharma | Fake News |

## Notes

- Short headlines and noisy social-media-style text are handled via robust preprocessing.
- Very short inputs (< 5 characters after cleaning) trigger a reliability warning.
- For production use, train on a larger dataset (e.g. Kaggle) for better generalization.

## Requirements

- Python 3.10+
- pandas, scikit-learn, nltk, matplotlib, seaborn, joblib
