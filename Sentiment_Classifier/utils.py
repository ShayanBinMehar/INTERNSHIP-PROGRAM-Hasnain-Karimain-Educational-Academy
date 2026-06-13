"""Text preprocessing utilities for sentiment classification."""

import re
import string

import nltk
from nltk.corpus import stopwords
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

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Clean and normalize text for model input."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]
    return " ".join(tokens)


def preprocess_batch(texts: list[str]) -> list[str]:
    """Preprocess a list of text samples."""
    return [preprocess_text(text) for text in texts]
