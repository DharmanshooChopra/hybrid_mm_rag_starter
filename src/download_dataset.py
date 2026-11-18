import argparse
from pathlib import Path
import urllib.request

BASE_URL = "https://github.com/allenai/multimodalqa/raw/master/dataset/"
FILES = [
    "MMQA_train.jsonl.gz",
    "MMQA_dev.jsonl.gz",
    "MMQA_texts.jsonl.gz",
    "MMQA_tables.jsonl.gz",
    "MMQA_images.jsonl.gz",
    "images.zip",
]


def download_file(url, target):
    print(f"[download] {url} -> {target}")
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as exc:
        if target.exists():
            target.unlink()
        print(f"[failed] {target.name}: {exc}")
        print("Check the upstream README for the current official download link:")
        print("https://github.com/allenai/multimodalqa")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download AllenAI MultiModalQA official dataset release.")
    parser.add_argument("--output-dir", default="data/multimodalqa/dataset",
                        help="Directory to save downloaded files")
    args = parser.parse_args()

    target_dir = Path(args.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        target = target_dir / name
        # If images.zip, it may be placed in data/ or target_dir
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {name} already exists in {target_dir}")
            continue
        # Also check root data/ fallback
        alt_target = Path("data") / name
        if alt_target.exists() and alt_target.stat().st_size > 0:
            print(f"[skip] {name} already exists in data/")
            continue

        url = BASE_URL + name
        download_file(url, target)

    print(f"\nDownload stage finished. Saved to {target_dir.resolve()}.")
    print("If images.zip was downloaded, extract it into data/multimodalqa/images/final_dataset_images/ before image indexing.")


if __name__ == "__main__":
    main()
