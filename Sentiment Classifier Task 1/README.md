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
Sentiment_Classifier/
├── dataset.csv          # 100 labeled sentences (25 per class)
├── model.py             # Training, evaluation, prediction & CLI
├── utils.py             # Text preprocessing (tokenization, lemmatization)
├── requirements.txt     # Python dependencies
├── model.pkl            # Saved model (generated after training)
├── confusion_matrix.png # Confusion matrix plot (generated after training)
└── README.md
```

## Setup

```bash
cd Sentiment_Classifier
pip install -r requirements.txt
```

## Usage

Run the interactive CLI menu:

```bash
python model.py
```

### Menu Options

1. **Train model** – Trains on `dataset.csv`, evaluates metrics, saves `model.pkl`, and shows confusion matrix
2. **Load saved model** – Loads a previously trained `model.pkl`
3. **Predict single sentence** – Enter one sentence and get sentiment + probabilities
4. **Batch prediction** – Predict multiple sentences, optionally export to CSV
5. **Evaluate & show confusion matrix** – Re-run evaluation on test split
6. **Exit**

### Example

```
Enter your sentence: I like the features but the app is very slow

Predicted Sentiment: Mixed 🤔
Probabilities:
   Mixed      72.15%  ██████████████
   Negative   15.32%  ███
   Positive   10.21%  ██
   Neutral     2.32%
```

## Technical Details

### Data Preprocessing
- Lowercasing
- Punctuation removal
- Stopword removal (NLTK)
- Tokenization & lemmatization (WordNet)

### Feature Engineering
- **TfidfVectorizer** with unigrams and bigrams (max 5000 features)

### Model
- **Logistic Regression** (multinomial, LBFGS solver)
- 80/20 stratified train-test split

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (weighted)
- Confusion matrix visualization

## Dataset

`dataset.csv` contains 100 labeled sentences with balanced distribution across all four sentiment classes.

## Author

AI Intern – Task 1: Multi-Class Sentiment Analysis
