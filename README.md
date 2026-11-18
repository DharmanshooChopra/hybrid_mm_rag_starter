# Hybrid Multimodal RAG + QUBO/QAOA — implementation starter

This repository is an implementation scaffold for the project. It does **not** contain measured results.
Run the classical retrieval baseline first, validate its output, and only then run evidence selection and QAOA.

## Dataset

The starter targets the official AllenAI MultiModalQA release (text, tables, images).
The official repository describes 29,918 QA examples and separate question/context files.
Dataset documentation: https://github.com/allenai/multimodalqa

The official repository's data format includes:
- question records with `qid`, `question`, `answers`, `metadata`, `supporting_context`
- text records with `id`, `title`, `text`
- table records with `id`, `title`, `table`
- image metadata with `id`, `title`, `path`

**Access/licensing note:** public download availability does not by itself establish permission for every reuse or redistribution. Check the dataset owner's current terms and cite the dataset paper. Do not redistribute downloaded image assets in your paper package.

## Step 0 — environment

Use a clean Python 3.10 or 3.11 environment. Install a PyTorch build matching your NVIDIA CUDA driver from https://pytorch.org/get-started/locally/ .

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Check GPU:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is not available, the encoders will run on CPU (slower).

## Step 1 — download dataset

Run:
```bash
python src/download_dataset.py
```

This downloads the official train/dev questions, text/table/image metadata and image archive.
Then extract `data/images.zip` into `data/multimodalqa/images/final_dataset_images/`:
```bash
python -c "import zipfile; zipfile.ZipFile('data/images.zip').extractall('data/multimodalqa/images/final_dataset_images')"
```

If an official download URL changes, follow the download links in the upstream repository README rather than substituting an unofficial mirror.

## Step 2 — prepare corpus

```bash
python src/prepare_dataset.py --data-dir data/multimodalqa/dataset --output-dir data/processed --max-questions 3000
```

The first run is intentionally limited to 3,000 dev questions for a manageable pilot. The corpus includes all loaded context records unless `--max-docs-per-modality` is set. The script creates:
- `data/processed/corpus.jsonl`
- `data/processed/dev_queries.jsonl`

Inspect these files before proceeding. The test split is not used for tuning.

## Step 3 — build and evaluate the classical multimodal retrieval baseline

```bash
python src/run_baseline.py --data-dir data/processed --output-dir results --top-k 10 --max-queries 500
```

The baseline:
- embeds text and serialized tables with Sentence-BERT
- embeds image files with CLIP when image files are present
- retrieves independently within each modality
- merges modality rankings using reciprocal-rank fusion (RRF)
- evaluates retrieval of annotated supporting context IDs on the labelled dev split

Validate the baseline output files:
```bash
python validate_baseline_outputs.py
```

## Step 4 — build QUBO instances from candidates

Construct QUBO instances containing linear relevance, pairwise semantic redundancy, token costs, and budget constraints:
```bash
python src/build_qubo_instances.py --corpus data/processed/corpus.jsonl --candidates results/candidates.jsonl --output results/qubo_instances.jsonl
```

## Step 5 — evidence selection (QUBO objective and exact reference solver)

```bash
python src/run_qubo.py --input results/qubo_instances.jsonl --output results/exact_selections.jsonl --lambda 0.35
```

The selector uses:
`maximize sum_i relevance_i*x_i - lambda*sum_(i<j) redundancy_ij*x_i*x_j`
subject to `sum_i token_cost_i*x_i <= budget`.

For small candidate sets, exhaustive enumeration gives an exact reference solution.

## Step 6 — evaluate QUBO evidence selection vs baseline retrieval

```bash
python src/evaluate_qubo.py --queries data/processed/dev_queries.jsonl --selections results/exact_selections.jsonl --candidates results/candidates.jsonl --output results/qubo_experiment/evaluation.json
```

## Step 7 — QAOA simulation (optional, after QUBO validation)

```bash
pip install qiskit qiskit-algorithms qiskit-optimization
python src/run_qaoa.py --input results/qubo_instances.jsonl --output results/qaoa_selections.jsonl
```

The QAOA script is deliberately limited to small candidate sets. It reports backend, depth, shots and feasibility. QAOA simulator execution is classical simulation; it is not evidence of quantum advantage.

## Step 6 — results and paper update

Only populate paper tables with values written to `results/` by completed runs. Report:
- corpus/query counts and modality distribution
- Recall@K / Precision@K / MRR / NDCG@K for retrieval
- feasible-sample rate and original QUBO objective
- runtime broken down by embedding, retrieval, optimization and generation
- repeated seeds and confidence intervals for stochastic QAOA runs

## Reproducibility

Keep a copy of:
- package versions (`pip freeze > results/requirements-lock.txt`)
- GPU model and CUDA version
- model checkpoint names
- random seeds
- dataset split and query IDs
- all command-line arguments
- raw run outputs

Do not use the test split for parameter tuning. Never invent missing values.
