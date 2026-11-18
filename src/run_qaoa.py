"""Optional QAOA integration. Requires a validated QUBO and qiskit-optimization."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Optional QAOA simulation for QUBO evidence selection.")
    ap.add_argument("--input", default="results/qubo_instances.jsonl",
                    help="Path to qubo_instances.jsonl")
    ap.add_argument("--output", default="results/qaoa_selections.jsonl",
                    help="Path to save qaoa_selections.jsonl")
    ap.add_argument("--reps", type=int, default=1,
                    help="QAOA circuit depth (p / reps)")
    ap.add_argument("--maxiter", type=int, default=100,
                    help="Maximum optimizer iterations")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="Max queries to simulate (0 = all)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.35,
                    help="Redundancy penalty weight lambda")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        fallback = Path("results/qubo_experiment/qubo_instances.jsonl")
        if fallback.exists():
            print(f"Input {src} not found; falling back to {fallback}")
            src = fallback
        else:
            raise SystemExit(
                f"{src} not found. First create validated QUBO instances with relevance, "
                "pairwise redundancy, token costs, and a budget using src/build_qubo_instances.py."
            )

    try:
        from qiskit.primitives import StatevectorSampler
        from qiskit_algorithms.optimizers import SPSA
        from qiskit_algorithms import QAOA
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
    except Exception as exc:
        raise SystemExit(
            "QAOA dependencies are not installed or API versions differ. "
            "Install compatible qiskit, qiskit-algorithms and qiskit-optimization packages. "
            f"Original error: {exc}"
        )

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    sampler = StatevectorSampler()
    qaoa = QAOA(sampler=sampler, optimizer=SPSA(maxiter=args.maxiter), reps=args.reps)
    optimizer = MinimumEigenOptimizer(qaoa)

    processed = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            if args.max_queries and processed >= args.max_queries:
                break

            item = json.loads(line)
            linear = item.get("linear") or item.get("linear_relevance", [])
            n = len(linear)
            if n == 0:
                continue

            qp = QuadraticProgram()
            for i in range(n):
                qp.binary_var(name=f"x{i}")

            quadratic = {}
            if "quadratic" in item and item["quadratic"]:
                for k, v in item["quadratic"].items():
                    try:
                        i_str, j_str = k.split(",")
                        quadratic[(f"x{i_str}", f"x{j_str}")] = float(v)
                    except ValueError:
                        pass
            elif "redundancy" in item:
                red = item["redundancy"]
                for i in range(n):
                    for j in range(i + 1, n):
                        val = -float(args.lam) * float(red[i][j])
                        if val:
                            quadratic[(f"x{i}", f"x{j}")] = val

            qp.maximize(linear={f"x{i}": float(linear[i]) for i in range(n)},
                        quadratic=quadratic)

            if "costs" in item and "budget" in item:
                costs = item["costs"]
                budget = float(item["budget"])
                qp.linear_constraint(
                    linear={f"x{i}": float(costs[i]) for i in range(n)},
                    sense="<=",
                    rhs=budget,
                    name="token_budget"
                )

            result = optimizer.solve(qp)
            x = [int(round(result.variables_dict.get(f"x{i}", 0))) for i in range(n)]
            doc_ids = item.get("doc_ids", [])
            selected_indices = [i for i, val in enumerate(x) if val == 1]
            selected_doc_ids = [doc_ids[i] for i in selected_indices if i < len(doc_ids)]

            fout.write(json.dumps({
                "qid": item.get("qid"),
                "x": x,
                "selected_indices": selected_indices,
                "selected_doc_ids": selected_doc_ids,
                "objective_value": float(result.fval) if hasattr(result, "fval") else None,
                "status": str(result.status),
                "backend": "Qiskit StatevectorSampler (classical statevector simulation)",
                "reps": args.reps,
                "maxiter": args.maxiter
            }) + "\n")
            processed += 1

    print(f"Saved QAOA simulation results for {processed} queries to {dst}")


if __name__ == "__main__":
    main()
