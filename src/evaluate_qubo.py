
import argparse
import json
import math
from pathlib import Path


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_metrics(retrieved, gold):
    retrieved = list(retrieved)
    gold = set(gold)

    if not gold:
        return None

    hits = [1 if doc_id in gold else 0 for doc_id in retrieved]
    hit_count = sum(hits)

    recall = hit_count / len(gold)
    precision = hit_count / len(retrieved) if retrieved else 0.0

    reciprocal_rank = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        hit / math.log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
    )
    ideal_hits = min(len(gold), len(retrieved))
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
        "hits": hit_count,
        "retrieved": len(retrieved),
        "gold": len(gold),
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(rows):
    return {
        "recall": mean([x["recall"] for x in rows]),
        "precision": mean([x["precision"] for x in rows]),
        "mrr": mean([x["mrr"] for x in rows]),
        "ndcg": mean([x["ndcg"] for x in rows]),
        "evaluated_queries": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate QUBO evidence selections vs baseline retrieval.")
    parser.add_argument("--queries", default="data/processed/dev_queries.jsonl",
                        help="Path to dev_queries.jsonl")
    parser.add_argument("--selections", default="results/qubo_experiment/exact_selections.jsonl",
                        help="Path to exact_selections.jsonl")
    parser.add_argument("--candidates", default="results/candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--output", default="results/qubo_experiment/evaluation.json",
                        help="Path to save evaluation report JSON")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K cutoff for baseline comparison")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found at: {queries_path}")

    selections_path = Path(args.selections)
    if not selections_path.exists():
        fallback_sel = Path("results/exact_selections.jsonl")
        if fallback_sel.exists():
            print(f"Selections file {selections_path} not found; falling back to {fallback_sel}")
            selections_path = fallback_sel
        else:
            raise FileNotFoundError(f"Selections file not found at: {selections_path}")

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        fallback_cand = Path("results/baseline_full_2441q/candidates.jsonl")
        if fallback_cand.exists():
            print(f"Candidates file {candidates_path} not found; falling back to {fallback_cand}")
            candidates_path = fallback_cand
        else:
            raise FileNotFoundError(f"Candidates file not found at: {candidates_path}")

    queries = {x["qid"]: x for x in load_jsonl(queries_path)}
    selections = {x["qid"]: x for x in load_jsonl(selections_path)}
    candidates = {x["qid"]: x for x in load_jsonl(candidates_path)}

    eval_qids = [qid for qid in selections if qid in queries and qid in candidates]
    if not eval_qids:
        raise RuntimeError(
            f"No matching query IDs found between selections ({len(selections)}), "
            f"candidates ({len(candidates)}), and queries ({len(queries)})."
        )

    if len(eval_qids) < len(selections):
        print(f"Warning: {len(selections) - len(eval_qids)} selections missing from queries/candidates. "
              f"Evaluating {len(eval_qids)} common queries.")

    qubo_rows = []
    baseline_rows = []
    selected_counts = []
    selected_costs = []
    objective_values = []
    empty_selections = 0
    missing_gold = 0

    for qid in eval_qids:
        q = queries[qid]
        gold = set(q.get("gold_doc_ids", []))
        if not gold:
            missing_gold += 1
            continue

        selection = selections[qid]
        candidate_row = candidates[qid]
        pool = candidate_row.get("candidates", [])

        if "selected_doc_ids" in selection:
            selected_ids = selection["selected_doc_ids"]
        else:
            selected_indices = selection.get("selected_indices", [])
            selected_ids = [
                pool[i]["doc_id"]
                for i in selected_indices
                if i < len(pool)
            ]

        baseline_ids = [
            item["doc_id"]
            for item in pool[:args.top_k]
        ]

        qm = compute_metrics(selected_ids, gold)
        bm = compute_metrics(baseline_ids, gold)

        if qm is not None:
            qubo_rows.append(qm)
            baseline_rows.append(bm)

        selected_counts.append(len(selected_ids))
        selected_costs.append(selection.get("token_cost", 0.0))
        objective_values.append(selection.get("objective", 0.0))

        if not selected_ids:
            empty_selections += 1

    report = {
        "evaluation_note": (
            "Metrics use the same fused candidate pool and gold document IDs. "
            "QUBO selections are variable-size and constrained by the configured "
            f"cost budget. Baseline is the first {args.top_k} fused candidates."
        ),
        "qubo": summarize(qubo_rows),
        f"baseline_top{args.top_k}": summarize(baseline_rows),
        "selection_statistics": {
            "mean_selected_documents": mean(selected_counts),
            "mean_selected_cost": mean(selected_costs),
            "mean_objective": mean(objective_values),
            "empty_selections": empty_selections,
        },
        "queries_without_gold_evidence": missing_gold,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
