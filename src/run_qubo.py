import argparse
import json
from itertools import product
from pathlib import Path
import numpy as np


def objective(x, rel, red, lam):
    value = float(np.dot(rel, x))
    value -= float(lam * sum(
        red[i, j] * x[i] * x[j]
        for i in range(len(x))
        for j in range(i + 1, len(x))
    ))
    return value


def solve_exact(rel, red, costs, budget, lam):
    n = len(rel)
    best = None

    for bits in product((0, 1), repeat=n):
        x = np.asarray(bits, dtype=int)
        cost = float(np.dot(costs, x))

        if cost > budget:
            continue

        value = objective(x, rel, red, lam)

        if best is None or value > best["objective"]:
            best = {
                "x": x.tolist(),
                "selected_indices": np.where(x == 1)[0].tolist(),
                "objective": value,
                "token_cost": cost,
                "feasible": True
            }

    if best is None:
        best = {
            "x": [0] * n,
            "selected_indices": [],
            "objective": 0.0,
            "token_cost": 0.0,
            "feasible": True
        }

    return best


def main():
    parser = argparse.ArgumentParser(description="Exact classical solver for QUBO evidence selection.")
    parser.add_argument("--input", default="results/qubo_instances.jsonl",
                        help="Path to qubo_instances.jsonl (or candidates.jsonl)")
    parser.add_argument("--output", default="results/exact_selections.jsonl",
                        help="Path to output exact_selections.jsonl")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.35,
                        help="Redundancy penalty weight lambda")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Max queries to solve (0 = all)")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        fallback = Path("results/qubo_experiment/qubo_instances.jsonl")
        if fallback.exists():
            print(f"Input file {src} not found; falling back to {fallback}")
            src = fallback
        else:
            fallback_cands = Path("results/candidates.jsonl")
            if fallback_cands.exists():
                print(f"Input file {src} not found; falling back to candidates file {fallback_cands}")
                src = fallback_cands
            else:
                raise FileNotFoundError(f"Input file not found at: {src}")

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    feasible = 0

    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)

            if args.max_queries and total >= args.max_queries:
                break

            doc_ids = item.get("doc_ids", [])

            if "linear_relevance" in item and "redundancy" in item:
                rel = np.asarray(item["linear_relevance"], dtype=float)
                red = np.asarray(item["redundancy"], dtype=float)
                costs = np.asarray(item.get("costs", [1.0] * len(rel)), dtype=float)
                budget = float(item.get("budget", 1200.0))
            elif "linear" in item:
                rel = np.asarray(item["linear"], dtype=float)
                n = len(rel)
                red = np.zeros((n, n), dtype=float)
                for k, v in item.get("quadratic", {}).items():
                    try:
                        i, j = map(int, k.split(","))
                        red[i, j] = red[j, i] = -float(v)
                    except ValueError:
                        pass
                costs = np.asarray(item.get("costs", [1.0] * n), dtype=float)
                budget = float(item.get("budget", 1200.0))
            elif "candidates" in item:
                cands = item.get("candidates", [])[:12]
                n = len(cands)
                rel = np.asarray([float(c.get("rrf_score", 1.0 / (60 + i + 1))) for i, c in enumerate(cands)], dtype=float)
                red = np.zeros((n, n), dtype=float)
                costs = np.ones(n, dtype=float)
                budget = 10.0
                doc_ids = [c["doc_id"] for c in cands]
            else:
                print(f"Skipping unparseable item with keys: {list(item.keys())}")
                continue

            result = solve_exact(rel, red, costs, budget, args.lam)

            selected_doc_ids = [doc_ids[i] for i in result["selected_indices"] if i < len(doc_ids)]

            record = {
                "qid": item["qid"],
                **result,
                "selected_doc_ids": selected_doc_ids,
                "lambda": args.lam
            }
            fout.write(json.dumps(record) + "\n")

            total += 1
            feasible += int(result["feasible"])

    summary = {
        "queries_solved": total,
        "feasible_solutions": feasible,
        "lambda": args.lam,
        "output": str(dst)
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
