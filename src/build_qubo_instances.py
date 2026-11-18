import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_CORPUS = "data/processed/corpus.jsonl"
DEFAULT_CANDIDATES = "results/candidates.jsonl"
DEFAULT_OUTPUT = "results/qubo_instances.jsonl"
DEFAULT_REPORT = "results/feature_report.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_CANDIDATES = 12
DEFAULT_TOKEN_BUDGET = 1200


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Build QUBO instances from retrieval candidates.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="Path to corpus.jsonl")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES, help="Path to candidates.jsonl")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save qubo_instances.jsonl")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Path to save feature_report.json")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES, help="Max candidates per query")
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET, help="Token budget constraint")
    parser.add_argument("--max-queries", type=int, default=0, help="Max queries to process (0 = all)")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at: {corpus_path}")

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        fallback = Path("results/baseline_full_2441q/candidates.jsonl")
        if fallback.exists():
            print(f"Candidates file {candidates_path} not found; falling back to {fallback}")
            candidates_path = fallback
        else:
            raise FileNotFoundError(f"Candidates file not found at: {candidates_path}")

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from: {corpus_path}")
    corpus_rows = load_jsonl(corpus_path)
    corpus = {row["doc_id"]: row for row in corpus_rows}

    print(f"Loading candidates from: {candidates_path}")
    queries = load_jsonl(candidates_path)
    if args.max_queries > 0:
        queries = queries[:args.max_queries]

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer '{args.model_name}' on {device}...")
    model = SentenceTransformer(args.model_name, device=device)
    tokenizer = model.tokenizer

    total = 0
    missing = 0
    modality_counts = {}
    token_costs = []

    with output_path.open("w", encoding="utf-8") as output:
        for row in queries:
            candidates = row.get("candidates", [])[:args.max_candidates]
            docs = []

            for candidate in candidates:
                doc = corpus.get(candidate["doc_id"])
                if doc is None:
                    missing += 1
                    continue
                docs.append((candidate, doc))

            if not docs:
                continue

            # Rank-derived relevance, not calibrated relevance probability.
            relevance = [
                1.0 / (60.0 + rank)
                for rank in range(1, len(docs) + 1)
            ]

            texts = []
            costs = []
            modalities = []

            for candidate, doc in docs:
                modality = doc.get("modality", "unknown")
                modalities.append(modality)

                title = doc.get("title", "")
                text = doc.get("text", "")

                if modality in ("text", "table"):
                    content = (title + "\n" + text).strip()
                    token_count = len(
                        tokenizer.encode(
                            content,
                            add_special_tokens=True,
                            truncation=True,
                            max_length=256
                        )
                    )
                    cost = max(1, token_count)
                elif modality == "image":
                    # Explicit pilot approximation: fixed image evidence cost.
                    # This is a modelling assumption, not measured image tokens.
                    content = title.strip() or "image evidence"
                    cost = 256
                else:
                    content = (title + "\n" + text).strip()
                    cost = max(1, len(tokenizer.encode(
                        content, add_special_tokens=True,
                        truncation=True, max_length=256
                    )))

                texts.append(content if content else "evidence")
                costs.append(cost)
                token_costs.append(cost)
                modality_counts[modality] = modality_counts.get(modality, 0) + 1

            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            redundancy = np.matmul(embeddings, embeddings.T)
            redundancy = np.clip(redundancy, 0.0, 1.0)
            np.fill_diagonal(redundancy, 0.0)

            quadratic_dict = {
                f"{i},{j}": -float(redundancy[i, j])
                for i in range(len(docs))
                for j in range(i + 1, len(docs))
            }

            item = {
                "qid": row["qid"],
                "question": row.get("question", ""),
                "doc_ids": [doc["doc_id"] for _, doc in docs],
                "modalities": modalities,
                "linear_relevance": relevance,
                "linear": relevance,
                "redundancy": redundancy.tolist(),
                "quadratic": quadratic_dict,
                "costs": costs,
                "budget": args.token_budget,
                "candidate_count": len(docs),
                "relevance_definition": "1/(60+candidate_rank)",
                "image_cost_assumption": 256
            }

            output.write(json.dumps(item) + "\n")
            total += 1

    report = {
        "queries_written": total,
        "missing_candidate_documents": missing,
        "max_candidates": args.max_candidates,
        "token_budget": args.token_budget,
        "modality_counts": modality_counts,
        "mean_cost": float(np.mean(token_costs)) if token_costs else None,
        "model": args.model_name,
        "image_cost_policy": "Fixed 256-unit pilot proxy; not measured visual token count."
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print("Created:", output_path)


if __name__ == "__main__":
    main()
