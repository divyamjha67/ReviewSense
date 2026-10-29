"""Keep preprocessing small and preserve negation such as 'not good'."""
import re
import unicodedata

MAX_CHARACTERS = 10_000

def clean_review(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()

def validate_review(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Review must be text.")
    if len(value) > MAX_CHARACTERS:
        raise ValueError("Each review must be at most 10,000 characters.")
    normalized = clean_review(value)
    if not normalized or not any(character.isalpha() for character in normalized):
        raise ValueError("Enter a review containing words, not only whitespace, numbers, or punctuation.")
    if len(normalized) > MAX_CHARACTERS:
        raise ValueError("Each normalized review must be at most 10,000 characters.")
    return normalized
