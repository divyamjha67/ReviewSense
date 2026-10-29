"""Download a small attributed public sentiment benchmark, without archive extraction."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import csv
import hashlib
import io
import json

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
FILES = {"amazon": "amazon_cells_labelled.txt", "imdb": "imdb_labelled.txt", "yelp": "yelp_labelled.txt"}

def main():
    with urlopen(URL, timeout=90) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    manifest_path = ROOT / "data/source.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["archive_sha256"] != digest:
        raise ValueError("Upstream archive checksum changed. Investigate the source before replacing the manifest.")
    records = []
    with ZipFile(io.BytesIO(content)) as archive:
        for source, suffix in FILES.items():
            member = next(name for name in archive.namelist() if name.endswith("/" + suffix))
            # splitlines() also splits Unicode NEL characters inside two IMDb
            # sentences; only physical LF record boundaries belong here.
            lines = archive.read(member).decode("utf-8").rstrip("\r\n").split("\n")
            for line in lines:
                review, label = line.rstrip("\r").rsplit("\t", 1)
                if label not in {"0", "1"}:
                    raise ValueError("Unexpected source label")
                records.append({"review": review, "label": int(label), "source": source})
            if len(lines) != 1000:
                raise ValueError(f"Unexpected row count for {source}")
    raw = ROOT / "data/raw/reviews.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["review", "label", "source"])
        writer.writeheader(); writer.writerows(records)
    manifest = {"dataset": "Sentiment Labelled Sentences", "creator": "Dimitrios Kotzias",
        "year": 2015, "doi": "https://doi.org/10.24432/C57604",
        "dataset_page": "https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences",
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "download_url": URL, "archive_sha256": digest,
        "raw_csv_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "rows": len(records), "sources": list(FILES),
        "changes": "Archive parsed to CSV; normalization, duplicate removal and splitting happen during training."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded and verified {len(records)} labeled review sentences. Archive SHA256: {digest}")

if __name__ == "__main__":
    main()
