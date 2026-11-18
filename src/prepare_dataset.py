
import argparse
import gzip
import json
from pathlib import Path
from tqdm import tqdm


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def table_to_text(record):
    table = record.get("table", {}) or {}
    parts = [record.get("title", "")]

    for row in table.get("table_rows", []) or []:
        cells = []
        for cell in row:
            if isinstance(cell, dict):
                cells.append(str(cell.get("text", "")))
            else:
                cells.append(str(cell))
        parts.append(" | ".join(cells))

    return "\n".join(x for x in parts if x.strip())


def main():
    parser = argparse.ArgumentParser(description="Prepare MultimodalQA corpus and dev queries.")
    parser.add_argument(
        "--data-dir",
        default="data/multimodalqa/dataset",
        help="Directory containing MMQA .jsonl.gz files"
    )
    parser.add_argument(
        "--image-dir",
        default="data/multimodalqa/images/final_dataset_images",
        help="Directory containing extracted images"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save corpus.jsonl and dev_queries.jsonl"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=3000,
        help="Maximum development queries to process"
    )
    parser.add_argument(
        "--max-docs-per-modality",
        type=int,
        default=0,
        help="Maximum documents per modality (0 = unlimited)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    # Fallback to alternate standard subdirectories if not found directly
    if not (data_dir / "MMQA_texts.jsonl.gz").exists():
        for candidate_sub in ("multimodalqa/dataset", "dataset"):
            candidate_dir = data_dir / candidate_sub
            if (candidate_dir / "MMQA_texts.jsonl.gz").exists():
                print(f"Data files not found in {data_dir}; using {candidate_dir}")
                data_dir = candidate_dir
                break

    if not image_dir.exists():
        for candidate_sub in (
            "data/multimodalqa/images/final_dataset_images",
            "multimodalqa/images/final_dataset_images",
            "images/final_dataset_images",
            "images",
        ):
            candidate_path = Path(candidate_sub)
            if candidate_path.is_dir():
                print(f"Image directory {image_dir} not found; using {candidate_path}")
                image_dir = candidate_path
                break

    files = {
        "text": data_dir / "MMQA_texts.jsonl.gz",
        "table": data_dir / "MMQA_tables.jsonl.gz",
        "image": data_dir / "MMQA_images.jsonl.gz",
    }

    docs = []
    doc_ids = {"text": set(), "table": set(), "image": set()}
    missing_images = []

    for modality, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)

        count = 0

        for record in tqdm(
            read_jsonl(path),
            desc=f"Loading {modality}"
        ):
            doc_id = str(record.get("id", "")).strip()

            if not doc_id:
                continue

            if args.max_docs_per_modality and (
                count >= args.max_docs_per_modality
            ):
                break

            if modality == "text":
                content = record.get("text", "")

            elif modality == "table":
                content = table_to_text(record)

            else:
                image_path = image_dir / Path(
                    record.get("path", "")
                ).name

                if not image_path.is_file():
                    missing_images.append(doc_id)

                content = record.get("title", "")

            docs.append({
                "doc_id": doc_id,
                "modality": modality,
                "title": record.get("title", ""),
                "text": content,
                "image_path": (
                    str(image_path.resolve())
                    if modality == "image"
                    and image_path.is_file()
                    else ""
                ),
                "url": record.get("url", "")
            })

            doc_ids[modality].add(doc_id)
            count += 1

    query_path = data_dir / "MMQA_dev.jsonl.gz"

    if not query_path.exists():
        raise FileNotFoundError(query_path)

    queries = []
    evidence_missing = []

    for record in tqdm(
        read_jsonl(query_path),
        desc="Loading development queries"
    ):
        supports = record.get("supporting_context", []) or []

        gold = [
            {
                "doc_id": str(item["doc_id"]),
                "modality": item.get("doc_part", "")
            }
            for item in supports
            if item.get("doc_id")
        ]

        for item in gold:
            modality = item["modality"]

            if item["doc_id"] not in doc_ids.get(modality, set()):
                evidence_missing.append({
                    "qid": record.get("qid"),
                    **item
                })

        queries.append({
            "qid": record.get("qid"),
            "question": record.get("question", ""),
            "answers": record.get("answers", []),
            "gold_evidence": gold,
            "gold_doc_ids": [
                item["doc_id"] for item in gold
            ],
            "required_modalities": (
                record.get("metadata", {}) or {}
            ).get("modalities", [])
        })

        if (
            args.max_questions > 0
            and len(queries) >= args.max_questions
        ):
            break

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "corpus.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for document in docs:
            f.write(
                json.dumps(document, ensure_ascii=False) + "\n"
            )

    with (output_dir / "dev_queries.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for query in queries:
            f.write(
                json.dumps(query, ensure_ascii=False) + "\n"
            )

    report = {
        "document_counts": {
            key: len(value) for key, value in doc_ids.items()
        },
        "total_documents": len(docs),
        "development_queries": len(queries),
        "missing_images": len(missing_images),
        "missing_gold_evidence": len(evidence_missing)
    }

    with (output_dir / "preparation_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nPreparation report:")
    print(json.dumps(report, indent=2))

    if missing_images or evidence_missing:
        raise RuntimeError(
            "Data validation failed. Inspect preparation_report "
            "and correct missing references before experiments."
        )


if __name__ == "__main__":
    main()
