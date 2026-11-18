import json
from pathlib import Path

files = [
    "results/baseline_summary.json",
    "results/baseline_query_metrics.jsonl",
    "results/candidates.jsonl",
    "results/failed_images.json",
]

for name in files:
    path = Path(name)
    print(f"\n{name}")
    print("Exists:", path.exists())
    if path.exists():
        print("Size:", path.stat().st_size, "bytes")

with open("results/baseline_summary.json", encoding="utf-8") as f:
    summary = json.load(f)

n_queries = (
    summary.get("dataset", {}).get("queries_evaluated")
    or summary.get("dataset", {}).get("dev_queries_total")
    or summary.get("n_queries")
)
n_images = (
    summary.get("cache", {}).get("image_embeddings_available")
    or summary.get("n_images_encoded")
)
n_failures = (
    summary.get("image_failures_this_run")
    if "image_failures_this_run" in summary
    else summary.get("n_failed_images", len(summary.get("failed_images", [])))
)

print("\nSummary validation:")
print("Queries:", n_queries)
print("Images encoded:", n_images)
print("Failed images:", n_failures)

with open("results/candidates.jsonl", encoding="utf-8") as f:
    first_candidate = json.loads(f.readline())

print("\nFirst candidate record:")
print(json.dumps(first_candidate, indent=2)[:2000])

with open("results/failed_images.json", encoding="utf-8") as f:
    failures = json.load(f)

print("\nFailure file type:", type(failures).__name__)
print("Failure count:", len(failures))
