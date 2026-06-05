from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")
EXTRA_SPACE_PATTERN = re.compile(r"\s+")

CUSTOM_STOP_WORDS = {
    "hi",
    "hello",
    "please",
    "thanks",
    "thank",
    "regards",
    "support",
    "customer",
}

STOP_WORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS)


def combine_ticket_text(subject: str, description: str) -> str:
    return f"{subject or ''} {description or ''}".strip()


def simple_lemmatize(token: str) -> str:
    irregular_words = {
        "was": "be",
        "were": "be",
        "is": "be",
        "are": "be",
        "has": "have",
        "had": "have",
        "failed": "fail",
        "charged": "charge",
        "cancelled": "cancel",
        "changed": "change",
        "expired": "expire",
        "attached": "attach",
        "affected": "affect",
        "crashes": "crash",
        "permissions": "permission",
    }

    if token in irregular_words:
        return irregular_words[token]

    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"

    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]

    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]

    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = NON_LETTER_PATTERN.sub(" ", text)
    text = EXTRA_SPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return [token for token in text.split() if token]


def remove_stop_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def preprocess_text(text: str) -> str:
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    meaningful_tokens = remove_stop_words(tokens)
    lemmatized_tokens = [simple_lemmatize(token) for token in meaningful_tokens]
    return " ".join(lemmatized_tokens)


def preprocess_ticket(subject: str, description: str) -> str:
    combined_text = combine_ticket_text(subject, description)
    return preprocess_text(combined_text)
