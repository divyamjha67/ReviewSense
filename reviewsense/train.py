"""Train a small text classifier without fitting vocabulary on held-out data."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from reviewsense.text import validate_review

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

def make_pipeline(classifier):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
        max_features=20_000, strip_accents="unicode")), ("classifier", classifier)])

def metrics(labels, probabilities):
    predicted = probabilities >= .5
    return {"accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist()}

def prepare_data(raw):
    data = raw.copy()
    valid = data["review"].map(lambda text: any(c.isalpha() for c in text))
    invalid_removed = int((~valid).sum())
    data = data.loc[valid].copy()
    data["review"] = data["review"].map(validate_review)
    counts = data.groupby("review")["label"].nunique()
    conflicts = set(counts[counts > 1].index)
    conflicting_removed = int(data["review"].isin(conflicts).sum())
    data = data.loc[~data["review"].isin(conflicts)]
    duplicates_removed = int(data.duplicated("review").sum())
    data = data.drop_duplicates("review").copy()
    data["review_id"] = data.review.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    return data[["review_id", "review", "label", "source"]].reset_index(drop=True), duplicates_removed, conflicting_removed, invalid_removed

def main():
    raw_path = ROOT / "data/raw/reviews.csv"
    if not raw_path.exists():
        raise SystemExit("Run python scripts/download_data.py first.")
    source = json.loads((ROOT / "data/source.json").read_text())
    dataset_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if dataset_sha != source["raw_csv_sha256"]:
        raise ValueError("Source data hash mismatch; download the original archive again.")
    raw = pd.read_csv(raw_path, keep_default_na=False)
    if len(raw) != 3000 or set(raw.label) != {0, 1} or set(raw.source) != {"amazon", "imdb", "yelp"}:
        raise ValueError("Unexpected dataset structure.")
    data, duplicates_removed, conflicting_removed, invalid_removed = prepare_data(raw)
    data.to_csv(ROOT / "data/reviews.csv", index=False)
    strata = data.source + ":" + data.label.astype(str)
    train_ids, test_ids = train_test_split(np.arange(len(data)), test_size=.2, stratify=strata, random_state=SEED)
    train, test = data.iloc[train_ids], data.iloc[test_ids]
    cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(train.review, strata.iloc[train_ids]))
    candidates = {"Prior baseline": DummyClassifier(strategy="prior"),
        **{f"Logistic regression (C={value:g})": LogisticRegression(C=value, max_iter=1500, random_state=SEED)
           for value in [.1, 1., 4.]}}
    comparison = []
    for name, classifier in candidates.items():
        scores = cross_val_score(make_pipeline(classifier), train.review, train.label, cv=cv, scoring="f1", n_jobs=1)
        comparison.append({"name": name, "cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std()), "fold_f1": scores.tolist()})
        print(f"{name}: CV positive-class F1 {scores.mean():.4f}", flush=True)
    winner = max(comparison, key=lambda row: row["cv_f1_mean"])["name"]
    model = make_pipeline(candidates[winner]).fit(train.review, train.label)
    probabilities = model.predict_proba(test.review)[:, 1]
    predicted = probabilities >= .5
    matches = np.asarray(model.named_steps["tfidf"].transform(test.review).getnnz(axis=1))
    actual = test.label.to_numpy()
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump(model, artifact_dir / "pipeline.joblib")
    domain_metrics = []
    for domain in sorted(test.source.unique()):
        mask = test.source.to_numpy() == domain
        domain_metrics.append({"source": domain, "rows": int(mask.sum()), **metrics(actual[mask], probabilities[mask])})
    terms = {"positive": [], "negative": []}
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        features = model.named_steps["tfidf"].get_feature_names_out()
        weights = classifier.coef_[0]
        for key, indices in [("negative", np.argsort(weights)[:12]), ("positive", np.argsort(weights)[-12:][::-1])]:
            terms[key] = [{"term": str(features[i]), "weight": float(weights[i])} for i in indices]
    error_indices = np.flatnonzero(predicted != actual)
    # Show the most confident mistakes to make limitations inspectable.
    error_indices = sorted(error_indices, key=lambda i: -abs(probabilities[i]-.5))[:12]
    report = {"model_version": "1.0.0", "model_name": winner,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "dataset_sha256": dataset_sha,
        "model_sha256": hashlib.sha256((artifact_dir / "pipeline.joblib").read_bytes()).hexdigest(),
        "rows_raw": len(raw), "rows_clean": len(data), "duplicates_removed": duplicates_removed,
        "conflicting_removed": conflicting_removed, "invalid_removed": invalid_removed, "train_rows": len(train), "test_rows": len(test),
        "positive_rate": float(data.label.mean()), "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
        "threshold": .5, "comparison": comparison, "metrics": metrics(actual, probabilities),
        "domain_metrics": domain_metrics, "top_terms": terms,
        "test_zero_vocabulary": int((matches == 0).sum()),
        "metric_scope": "All held-out rows, using the underlying classifier at threshold 0.5. The app abstains on rows with no vocabulary matches.",
        "error_examples": [{"review": test.review.iloc[i], "actual": "Positive" if actual[i] else "Negative",
            "predicted": "Positive" if predicted[i] else "Negative", "positive_probability": float(probabilities[i])}
            for i in error_indices],
        "limitations": ["Curated balanced short English sentences, not representative production feedback",
            "No neutral sentiment class; sarcasm and mixed opinions are not specifically modeled",
            "Random stratified sentence split, not a future-time or author-disjoint evaluation",
            "Probabilities are not explicitly calibrated; vocabulary matches do not establish domain validity"]}
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "splits.json").write_text(json.dumps({"train": train.review_id.tolist(), "test": test.review_id.tolist()}, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["model_name", "rows_raw", "rows_clean", "duplicates_removed", "conflicting_removed", "train_rows", "test_rows", "metrics", "test_zero_vocabulary"]}, indent=2))

if __name__ == "__main__":
    main()
