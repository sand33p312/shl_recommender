"""
Retriever: TF-IDF cosine similarity search over the SHL catalog.

Design choices:
  - Pure numpy — no sentence-transformers, FAISS, or heavy dependencies.
    Keeps the Docker image small and cold-start fast.
  - L2-normalised TF-IDF with smooth IDF to avoid zero-division on rare terms.
  - Optional test_type_filter applies a score boost (not hard filter) so
    relevant cross-type results can still surface.
  - get_by_name supports exact and case-insensitive partial matching for
    hallucination recovery in the agent layer.
"""

import json
import math
import re

import numpy as np
from functools import lru_cache
from pathlib import Path


# ─── Text normalisation ────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "and", "or", "for", "to", "of",
    "in", "on", "with", "at", "by", "from", "as", "be", "was", "were",
    "that", "this", "it", "its", "not", "no", "can", "has", "have",
    "will", "do", "does", "did", "but", "so", "if", "we", "you",
})


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t not in _STOPWORDS and len(t) > 1]


# ─── Retriever ─────────────────────────────────────────────────────────────────

class CatalogRetriever:
    """TF-IDF retriever over the SHL assessment catalog."""

    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
        self._load()

    def _load(self) -> None:
        with open(self.catalog_path, "r") as fh:
            data = json.load(fh)

        self.assessments: list[dict] = data["assessments"]
        self.test_type_legend: dict = data.get("test_type_legend", {})

        # Build document corpus: name (3×) + description + type labels
        # Tripling the name gives it strong TF weight vs generic description words
        self.corpus = []
        for a in self.assessments:
            name_tokens = " ".join([a["name"]] * 3)
            type_labels = " ".join(a.get("test_type_labels", []))
            text = f"{name_tokens} {a.get('description', '')} {type_labels}"
            self.corpus.append(text)

        self._build_tfidf()

    def _build_tfidf(self) -> None:
        """Build and L2-normalise the TF-IDF matrix."""
        N = len(self.corpus)

        # Document frequency
        df: dict[str, int] = {}
        tokenized_docs: list[list[str]] = []
        for doc in self.corpus:
            tokens = _tokenize(doc)
            tokenized_docs.append(tokens)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        self.vocab = sorted(df.keys())
        vocab_idx = {t: i for i, t in enumerate(self.vocab)}

        # Smooth IDF: log((N+1)/(df+1)) + 1
        idf = np.array(
            [math.log((N + 1) / (df.get(t, 0) + 1)) + 1.0 for t in self.vocab],
            dtype=np.float32,
        )

        # TF-IDF matrix
        V = len(self.vocab)
        self.tfidf_matrix = np.zeros((N, V), dtype=np.float32)
        for doc_idx, tokens in enumerate(tokenized_docs):
            tf_raw: dict[str, int] = {}
            for t in tokens:
                if t in vocab_idx:
                    tf_raw[t] = tf_raw.get(t, 0) + 1
            n_tokens = max(len(tokens), 1)
            for t, count in tf_raw.items():
                self.tfidf_matrix[doc_idx, vocab_idx[t]] = count / n_tokens

        self.tfidf_matrix *= idf

        # L2 normalise rows (unit vectors → dot product == cosine similarity)
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self.tfidf_matrix /= norms

        # Store IDF and vocab index for query vectorisation
        self._idf = idf
        self._vocab_idx = vocab_idx

    def _vectorise_query(self, query: str) -> np.ndarray:
        """Convert a query string to a normalised TF-IDF vector."""
        tokens = _tokenize(query)
        V = len(self.vocab)
        vec = np.zeros(V, dtype=np.float32)
        tf_raw: dict[str, int] = {}
        for t in tokens:
            if t in self._vocab_idx:
                tf_raw[t] = tf_raw.get(t, 0) + 1
        n_tokens = max(len(tokens), 1)
        for t, count in tf_raw.items():
            idx = self._vocab_idx[t]
            vec[idx] = (count / n_tokens) * self._idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(
        self,
        query: str,
        k: int = 10,
        test_type_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Return up to k assessments ranked by relevance to query.

        test_type_filter: if provided, apply a 1.5× score boost to assessments
                          whose test_types overlap with the filter list.
                          This is a soft boost, not a hard filter, so highly
                          relevant cross-type results can still surface.
        """
        q_vec = self._vectorise_query(query)
        scores: np.ndarray = self.tfidf_matrix @ q_vec  # (N,)

        if test_type_filter:
            for i, a in enumerate(self.assessments):
                if set(a.get("test_types", [])) & set(test_type_filter):
                    scores[i] *= 1.5

        top_indices = np.argsort(scores)[::-1][:k]
        return [self.assessments[i] for i in top_indices if scores[i] > 0]

    def get_by_name(self, name: str) -> dict | None:
        """
        Look up an assessment by name.
        1. Exact match (case-insensitive)
        2. One name is a substring of the other
        Returns the first match or None.
        """
        name_lower = name.strip().lower()
        # Pass 1: exact
        for a in self.assessments:
            if a["name"].lower() == name_lower:
                return a
        # Pass 2: substring
        for a in self.assessments:
            aname = a["name"].lower()
            if name_lower in aname or aname in name_lower:
                return a
        return None

    def get_all(self) -> list[dict]:
        """Return the full catalog (used by hallucination guard)."""
        return self.assessments


@lru_cache(maxsize=1)
def get_retriever(catalog_path: str) -> CatalogRetriever:
    """Return a cached CatalogRetriever (singleton per path)."""
    return CatalogRetriever(catalog_path)
