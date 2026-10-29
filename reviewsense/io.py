"""CSV validation and spreadsheet-safe exports; no uploads are saved."""
import csv
import io
import pandas as pd
from reviewsense.text import validate_review

MAX_BYTES = 1_000_000
MAX_ROWS = 1000

def parse_reviews_csv(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_BYTES:
        raise ValueError("CSV must be at most 1,000,000 bytes.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Use a UTF-8 CSV file.") from error
    reader = csv.DictReader(io.StringIO(text), strict=True)
    try:
        names = reader.fieldnames or []
        if len(set(names)) != len(names):
            raise ValueError("Duplicate CSV column names are not allowed.")
        if "review" not in names or set(names) - {"review", "review_id"}:
            raise ValueError("CSV needs a review column and optionally review_id; no other columns are accepted.")
        rows, ids, problems = [], set(), []
        for index, row in enumerate(reader, 2):
            if len(rows) + len(problems) >= MAX_ROWS:
                raise ValueError("Maximum 1,000 review rows per upload.")
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Row {index} has the wrong number of columns.")
            identifier = row.get("review_id", "").strip() or f"row-{index - 1}"
            if len(identifier) > 80:
                raise ValueError(f"Row {index}: review_id must be at most 80 characters.")
            if identifier in ids:
                raise ValueError(f"Row {index}: duplicate review_id.")
            ids.add(identifier)
            try:
                validate_review(row["review"])
                rows.append({"review_id": identifier, "review": row["review"]})
            except ValueError as error:
                problems.append(f"Row {index}: {error}")
        if problems:
            raise ValueError("No rows scored. Correct these reviews and upload again:\n" + "\n".join(problems[:10]))
        if not rows:
            raise ValueError("CSV contains no review rows.")
        return pd.DataFrame(rows, columns=["review_id", "review"])
    except csv.Error as error:
        raise ValueError("Malformed CSV quoting. Check the header and review cells.") from error

def _safe_cell(value):
    if not isinstance(value, str):
        return value
    stripped = value.lstrip()
    if stripped.startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value

def scored_csv(input_df: pd.DataFrame, predictions: list[dict]) -> bytes:
    if len(input_df) != len(predictions):
        raise ValueError("Prediction count must match input rows.")
    result = input_df[["review_id", "review"]].reset_index(drop=True).copy()
    for column in ["sentiment", "positive_probability", "vocabulary_matches"]:
        result[column] = [prediction[column] for prediction in predictions]
    for column in ["review_id", "review", "sentiment"]:
        result[column] = result[column].map(_safe_cell)
    return result.to_csv(index=False, quoting=csv.QUOTE_ALL, lineterminator="\r\n").encode("utf-8-sig")
