"""Single and batch inference use the saved training pipeline."""
from pathlib import Path
import hashlib
import json
import joblib
from reviewsense.text import validate_review

ROOT = Path(__file__).resolve().parents[1]

class Scorer:
    def __init__(self, artifact_dir=ROOT / "artifacts"):
        artifact_dir = Path(artifact_dir)
        self.report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
        path = artifact_dir / "pipeline.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.report["model_sha256"]:
            raise ValueError("Model checksum mismatch. Retrain your local artifact and restart the app.")
        # Hashes detect mismatch, not authenticity. Never load an untrusted joblib.
        self.pipeline = joblib.load(path)

    def predict(self, reviews: list[str]) -> list[dict]:
        if not reviews:
            raise ValueError("Provide at least one review.")
        cleaned = [validate_review(review) for review in reviews]
        matrix = self.pipeline.named_steps["tfidf"].transform(cleaned)
        matches = matrix.getnnz(axis=1)
        probabilities = self.pipeline.named_steps["classifier"].predict_proba(matrix)[:, 1]
        return [{"sentiment": "Not scored" if count == 0 else ("Positive" if probability >= .5 else "Negative"),
                 "positive_probability": None if count == 0 else float(probability),
                 "vocabulary_matches": int(count)} for probability, count in zip(probabilities, matches)]

def load_scorer():
    return Scorer()
