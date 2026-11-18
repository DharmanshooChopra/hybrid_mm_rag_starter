# What happens next (student checklist)

## You must do locally
1. Confirm the NVIDIA GPU model and CUDA availability using the command in README.
2. Create the Python environment and install dependencies.
3. Download the official MultiModalQA data.
4. Extract the image archive.
5. Run corpus preparation.
6. Run the baseline script and send me:
   - `results/baseline_summary.json`
   - first 5 lines of `results/baseline_query_metrics.jsonl`
   - terminal output if there are errors
7. Do not submit or quote any results in the paper until the run is complete and checked.

## I can help with
- resolving installation/runtime errors from the exact traceback
- improving feature extraction and calibrating modality-specific scores
- implementing redundancy and token-cost features
- building a validated QUBO instance generator
- comparing exact classical, greedy, MMR and QAOA selectors
- producing tables/plots from your actual result files
- revising the paper after results exist

## Research integrity
The supplied QUBO selector file is an initial exact classical scaffold and uses neutral redundancy/unit costs until proper features are implemented. It is useful for testing the pipeline shape only. It is not publishable evidence. The QAOA script refuses to run without a separately validated QUBO instance file. This is intentional.
