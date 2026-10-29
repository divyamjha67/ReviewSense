"""Behavioral checks for text validation, CSV safety and trained inference."""
import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from reviewsense.io import parse_reviews_csv, scored_csv
from reviewsense.service import Scorer, load_scorer
from reviewsense.text import clean_review, validate_review

ROOT = Path(__file__).resolve().parents[1]


def make_csv(rows, fields=("review_id", "review")):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


@pytest.fixture(scope="module")
def scorer():
    return load_scorer()


def test_normalization_preserves_negation():
    assert clean_review("  ＮＯＴ\tGood\n at all  ") == "not good at all"
    assert validate_review("Not a good product.") == "not a good product."


@pytest.mark.parametrize("value", [None, 1, True, "", " \t\n ", "!!!123", "😀", "a"*10001])
def test_invalid_review_rejected(value):
    with pytest.raises(ValueError):
        validate_review(value)


def test_csv_bom_and_generated_identifiers():
    raw = b"\xef\xbb\xbfreview\r\nExcellent service!\r\nTerrible quality.\r\n"
    result = parse_reviews_csv(raw)
    assert list(result.columns) == ["review_id", "review"]
    assert len(result) == 2
    assert result.review_id.is_unique
    assert result.review.str.len().gt(0).all()


def test_csv_quoted_commas_and_multiline_cells():
    raw = make_csv([{"review_id": "A", "review": 'Great food, "lovely" staff.\nVery helpful.'}])
    result = parse_reviews_csv(raw)
    assert result.review_id.tolist() == ["A"]
    assert "lovely" in result.review.iloc[0]


@pytest.mark.parametrize("raw", [
    b"", b"wrong\nhello", b"review,review\ngood,bad",
    b"review,label\ngood,1", b"review\n", b"review\n\"unterminated",
    b"\xff\xfe", b"review_id,review\nA,good,extra",
    b"review_id,review\nA", b"review_id,review\nA,Good\nA,Bad",
])
def test_invalid_csv_rejected(raw):
    with pytest.raises(ValueError):
        parse_reviews_csv(raw)


def test_invalid_row_rejects_whole_csv():
    raw = make_csv([
        {"review_id": "valid", "review": "Excellent product."},
        {"review_id": "invalid", "review": "12345!!!"},
    ])
    with pytest.raises(ValueError):
        parse_reviews_csv(raw)


def test_csv_limits():
    with pytest.raises(ValueError):
        parse_reviews_csv(b"x"*1_000_001)
    raw = make_csv([{"review_id": str(index), "review": "A nice product."} for index in range(1001)])
    with pytest.raises(ValueError):
        parse_reviews_csv(raw)
    with pytest.raises(ValueError):
        parse_reviews_csv(make_csv([{"review_id": "x"*81, "review": "Good service."}]))


def test_prediction_matches_fitted_pipeline(scorer):
    reviews = ["I love this excellent product!", "The food was awful and the service was terrible."]
    expected = scorer.pipeline.predict_proba([clean_review(value) for value in reviews])[:, 1]
    results = scorer.predict(reviews)
    np.testing.assert_allclose([row["positive_probability"] for row in results], expected)
    assert [row["sentiment"] for row in results] == ["Positive" if p >= .5 else "Negative" for p in expected]
    assert all(row["vocabulary_matches"] > 0 for row in results)


def test_unknown_vocabulary_has_no_supported_score(scorer):
    result = scorer.predict(["zxqvzzzx qxzzvvqqz"])[0]
    assert result["sentiment"] == "Not scored"
    assert result["positive_probability"] is None
    assert result["vocabulary_matches"] == 0


def test_batch_and_individual_prediction_parity(scorer):
    raw = make_csv([
        {"review_id": "P", "review": "Good food and friendly staff."},
        {"review_id": "N", "review": "Poor quality, very disappointing."},
    ])
    reviews = parse_reviews_csv(raw)
    batch = scorer.predict(reviews.review.tolist())
    separate = [scorer.predict([review])[0] for review in reviews.review]
    assert batch == separate


def test_csv_export_preserves_prediction_order(scorer):
    reviews = parse_reviews_csv(make_csv([
        {"review_id": "first", "review": "Excellent product."},
        {"review_id": "second", "review": "Terrible service."},
    ]))
    results = scorer.predict(reviews.review.tolist())
    output = scored_csv(reviews, results)
    exported = list(csv.DictReader(io.StringIO(output.decode("utf-8-sig"))))
    assert [row["review_id"] for row in exported] == ["first", "second"]
    assert [row["sentiment"] for row in exported] == [row["sentiment"] for row in results]
    np.testing.assert_allclose([float(row["positive_probability"]) for row in exported], [row["positive_probability"] for row in results])


def test_export_sanitizes_formula_like_text():
    reviews = pd.DataFrame({"review_id": ["=A1", "+formula"], "review": ["=Excellent product", "@Good service"]})
    predictions = [{"sentiment": "Positive", "positive_probability": .8, "vocabulary_matches": 2}]*2
    exported = list(csv.DictReader(io.StringIO(scored_csv(reviews, predictions).decode("utf-8-sig"))))
    assert all(row["review_id"].startswith("'") and row["review"].startswith("'") for row in exported)


def test_artifact_mismatch_rejected_before_loading(tmp_path):
    report = json.loads((ROOT / "artifacts/report.json").read_text())
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "pipeline.joblib").write_bytes(b"not the trusted artifact")
    with pytest.raises(ValueError):
        Scorer(artifact_dir=tmp_path)


def test_cleaned_dataset_and_partition_integrity():
    data = pd.read_csv(ROOT / "data/reviews.csv")
    splits = json.loads((ROOT / "artifacts/splits.json").read_text())
    train, test = set(splits["train"]), set(splits["test"])
    assert not train.intersection(test)
    assert train | test == set(data.review_id)
    assert data.review_id.is_unique
    assert data.review.map(clean_review).is_unique
    assert set(data.label) == {0, 1}
    assert set(data.source) == {"amazon", "yelp", "imdb"}


def test_vectorizer_is_fitted_on_training_statistics(scorer):
    data = pd.read_csv(ROOT / "data/reviews.csv")
    splits = json.loads((ROOT / "artifacts/splits.json").read_text())
    train = data[data.review_id.isin(splits["train"])]
    vectorizer = scorer.pipeline.named_steps["tfidf"]
    analyzer = vectorizer.build_analyzer()
    possible_terms = {term for text in train.review for term in analyzer(clean_review(text))}
    assert set(vectorizer.vocabulary_).issubset(possible_terms)
    reference = clone(vectorizer).fit(train.review.map(clean_review).tolist())
    assert vectorizer.vocabulary_ == reference.vocabulary_
    np.testing.assert_allclose(vectorizer.idf_, reference.idf_)


def test_report_matches_saved_experiment(scorer):
    report = scorer.report
    assert report["train_rows"] + report["test_rows"] == report["rows_clean"]
    assert sum(map(sum, report["metrics"]["confusion_matrix"])) == report["test_rows"]
    assert report["model_name"] == max(report["comparison"], key=lambda row: row["cv_f1_mean"])["name"]
    assert all(0 <= report["metrics"][name] <= 1 for name in ["accuracy", "precision", "recall", "f1", "roc_auc"])


def test_streamlit_initial_page_has_no_exception():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
