"""Text preprocessing utilities for sentiment classification."""

import re
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data on first run
for resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(
            f"tokenizers/{resource}"
            if resource.startswith("punkt")
            else f"corpora/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

# Keep negation/contrast words; remove only high-frequency function words so
# Neutral factual sentences retain distinguishing content terms.
MINIMAL_STOP_WORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "with", "as", "by", "from", "into", "through", "during", "before",
    "after", "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "few", "most", "other", "some", "such", "only",
    "own", "same", "so", "than", "can", "will", "just", "don", "should",
    "now", "he", "she", "they", "them", "their", "we", "our", "you", "your",
    "i", "me", "my", "him", "her", "his", "hers", "us", "yourself", "yourselves",
}
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Clean and normalize text while preserving sentiment and content words."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in MINIMAL_STOP_WORDS and len(token) > 1
    ]
    return " ".join(tokens)


def preprocess_batch(texts: list[str]) -> list[str]:
    """Preprocess a list of text samples."""
    return [preprocess_text(text) for text in texts]
