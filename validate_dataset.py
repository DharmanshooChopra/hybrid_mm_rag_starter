import gzip
import json
from pathlib import Path

folder = Path("data/multimodalqa/dataset")

files = [
    "MMQA_train.jsonl.gz",
    "MMQA_dev.jsonl.gz",
    "MMQA_test.jsonl.gz",
    "MMQA_texts.jsonl.gz",
    "MMQA_tables.jsonl.gz",
    "MMQA_images.jsonl.gz",
]

for filename in files:
    path = folder / filename
    print(f"\nChecking: {filename}")

    with gzip.open(path, "rt", encoding="utf-8") as f:
        first_line = f.readline()
        record = json.loads(first_line)

    print("  Valid gzip: YES")
    print("  Valid JSON: YES")
    print("  First record fields:", list(record.keys()))
