"""
Text Preprocessing Pipeline
Modular pipeline for cleaning, tokenizing, stopword removal, and lemmatization.
"""

import logging
import re
import string
from typing import List, Optional, Union

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag

# ---------------------------------------------------------------------------
# NLTK resource bootstrap (downloads only if missing)
# ---------------------------------------------------------------------------
_REQUIRED_NLTK = (
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "punkt",
    "punkt_tab",
)


def _ensure_nltk_data() -> None:
    """Download required NLTK corpora and models on first use."""
    for resource in _REQUIRED_NLTK:
        try:
            if resource == "stopwords":
                stopwords.words("english")
            elif resource == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif resource == "averaged_perceptron_tagger":
                nltk.data.find("taggers/averaged_perceptron_tagger")
            elif resource == "averaged_perceptron_tagger_eng":
                nltk.data.find("taggers/averaged_perceptron_tagger_eng")
            elif resource in ("punkt", "punkt_tab"):
                nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# WordNet POS tag mapping for accurate lemmatization
_POS_MAP = {
    "J": "a",  # adjective
    "V": "v",  # verb
    "N": "n",  # noun
    "R": "r",  # adverb
}


def _wordnet_pos(treebank_tag: str) -> str:
    """Map Penn Treebank POS tag to WordNet POS letter."""
    if not treebank_tag:
        return "n"
    prefix = treebank_tag[0].upper()
    return _POS_MAP.get(prefix, "n")


class TextPreprocessor:
    """
    Reusable text preprocessing pipeline.

    Steps: clean -> tokenize -> remove stopwords -> lemmatize (optional stemming).
    """

    DEFAULT_STOPWORDS = {
        "is", "the", "and", "a", "in", "am", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall", "can",
        "to", "of", "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
        "them", "my", "your", "his", "its", "our", "their", "this", "that",
        "these", "those", "what", "which", "who", "whom", "an", "or", "but",
        "if", "because", "while", "although", "so", "than", "too", "very",
    }

    def __init__(
        self,
        remove_numbers: bool = True,
        use_nltk_stopwords: bool = True,
        enable_logging: bool = True,
        log_level: int = logging.INFO,
    ) -> None:
        _ensure_nltk_data()

        self.remove_numbers = remove_numbers
        self.enable_logging = enable_logging
        self._lemmatizer = WordNetLemmatizer()
        self._stemmer = PorterStemmer()

        if use_nltk_stopwords:
            self._stopwords = set(stopwords.words("english"))
        else:
            self._stopwords = self.DEFAULT_STOPWORDS.copy()

        self._logger = logging.getLogger("TextPreprocessor")
        self._logger.setLevel(log_level)
        if enable_logging and not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
            )
            self._logger.addHandler(handler)

    def _log(self, step: str, message: str) -> None:
        if self.enable_logging:
            self._logger.info("[%s] %s", step, message)

    # ------------------------------------------------------------------
    # Step 1: Text cleaning
    # ------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """
        Normalize raw text: lowercase, strip punctuation/special chars,
        optionally remove numbers, collapse whitespace.
        """
        if not text or not str(text).strip():
            self._log("CLEAN", "Empty input received — returning empty string.")
            return ""

        cleaned = str(text).lower()

        # Remove numbers when configured
        if self.remove_numbers:
            cleaned = re.sub(r"\d+", " ", cleaned)

        # Remove punctuation and special characters (keep letters and spaces)
        cleaned = re.sub(r"[^a-z\s]", " ", cleaned)

        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        self._log("CLEAN", f"'{text[:60]}...' -> '{cleaned[:60]}...'" if len(text) > 60 else f"'{text}' -> '{cleaned}'")
        return cleaned

    # ------------------------------------------------------------------
    # Step 2: Custom tokenization (regex-based, not default library split)
    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        """
        Custom tokenizer using regex word-boundary matching.
        Returns empty list for empty/whitespace-only input.
        """
        if not text or not text.strip():
            self._log("TOKENIZE", "No tokens — empty text.")
            return []

        tokens = re.findall(r"\b[a-z]+\b", text)
        self._log("TOKENIZE", f"{len(tokens)} token(s): {tokens}")
        return tokens

    # ------------------------------------------------------------------
    # Step 3: Stopword removal
    # ------------------------------------------------------------------
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter out common stopwords from token list."""
        if not tokens:
            self._log("STOPWORDS", "No tokens to filter.")
            return []

        filtered = [t for t in tokens if t not in self._stopwords]
        removed = len(tokens) - len(filtered)
        self._log("STOPWORDS", f"Removed {removed} stopword(s). Remaining: {filtered}")
        return filtered

    # ------------------------------------------------------------------
    # Step 4: Lemmatization (POS-aware)
    # ------------------------------------------------------------------
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Convert tokens to base form using WordNet lemmatizer with POS tags."""
        if not tokens:
            self._log("LEMMATIZE", "No tokens to lemmatize.")
            return []

        tagged = pos_tag(tokens)
        lemmas = [
            self._lemmatizer.lemmatize(word, _wordnet_pos(tag))
            for word, tag in tagged
        ]
        self._log("LEMMATIZE", f"Lemmas: {lemmas}")
        return lemmas

    # ------------------------------------------------------------------
    # Bonus: Stemming for comparison
    # ------------------------------------------------------------------
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens (for comparison with lemmatization)."""
        if not tokens:
            return []
        stems = [self._stemmer.stem(t) for t in tokens]
        self._log("STEM", f"Stems: {stems}")
        return stems

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def preprocess(self, text: str, return_stems: bool = False) -> Union[List[str], dict]:
        """
        Run the full preprocessing pipeline on a single text string.

        Parameters
        ----------
        text : str
            Raw input text.
        return_stems : bool
            If True, return dict with tokens, lemmas, and stems for comparison.

        Returns
        -------
        list or dict
            Processed lemma list, or comparison dict when return_stems=True.
        """
        self._log("PIPELINE", "Starting preprocessing pipeline.")
        raw = text

        cleaned = self.clean_text(raw)
        tokens = self.tokenize(cleaned)
        no_stop = self.remove_stopwords(tokens)
        lemmas = self.lemmatize_tokens(no_stop)

        self._log("PIPELINE", f"Done. Raw: {len(raw)} chars -> {len(lemmas)} lemma(s).")

        if return_stems:
            stems = self.stem_tokens(no_stop)
            return {
                "raw": raw,
                "cleaned": cleaned,
                "tokens": tokens,
                "without_stopwords": no_stop,
                "lemmas": lemmas,
                "stems": stems,
            }

        return lemmas

    def preprocess_batch(self, texts: List[str], return_stems: bool = False) -> List[Union[List[str], dict]]:
        """Apply preprocessing to a list of text strings."""
        return [self.preprocess(t, return_stems=return_stems) for t in texts]

    def compare_raw_vs_processed(self, text: str) -> None:
        """Print side-by-side comparison of raw text and each pipeline stage."""
        result = self.preprocess(text, return_stems=True)

        print("\n" + "=" * 60)
        print("RAW vs PROCESSED COMPARISON")
        print("=" * 60)
        print(f"  Raw text       : {result['raw']}")
        print(f"  Cleaned        : {result['cleaned']}")
        print(f"  Tokens         : {result['tokens']}")
        print(f"  No stopwords   : {result['without_stopwords']}")
        print(f"  Lemmatized     : {result['lemmas']}")
        print(f"  Stemmed        : {result['stems']}")
        print("=" * 60 + "\n")

        # Highlight lemma vs stem differences
        diffs = [
            (l, s)
            for l, s in zip(result["lemmas"], result["stems"])
            if l != s
        ]
        if diffs:
            print("Lemma vs Stem differences:")
            for lemma, stem in diffs:
                print(f"  {lemma!r} (lemma) vs {stem!r} (stem)")
        else:
            print("Lemma and stem outputs are identical for all tokens.")
        print()


def preprocess(text: str, remove_numbers: bool = True) -> List[str]:
    """
    Convenience function — single-call preprocessing.

    Example
    -------
    >>> preprocess("I am loving the new AI tools!")
    ['love', 'new', 'ai', 'tool']
    """
    pipeline = TextPreprocessor(remove_numbers=remove_numbers, enable_logging=False)
    return pipeline.preprocess(text)
