import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
CACHE_VERSION = 2
RRF_K = 60
CANDIDATE_MULTIPLIER = 5


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fingerprint(items):
    h = hashlib.sha256()
    for item in items:
        h.update(json.dumps(item, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def file_signature(path):
    p = Path(path).resolve()
    s = p.stat()
    return {"path": str(p), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def versions():
    import PIL
    import sentence_transformers
    import torch
    import transformers
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "numpy": np.__version__, "pillow": PIL.__version__
    }


def st_revision(model):
    try:
        return getattr(model[0].auto_model.config, "_commit_hash", None)
    except Exception:
        return None


def clip_revision(model):
    return getattr(model.config, "_commit_hash", None)


def rrf(rankings):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            scores[doc["doc_id"]] += 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def metrics(pred, gold, k):
    gold = set(gold)
    if not gold:
        return None
    top = pred[:k]
    hits = sum(x in gold for x in top)
    rr = next((1.0 / rank for rank, x in enumerate(pred, 1) if x in gold), 0.0)
    dcg = sum(1.0 / np.log2(rank + 1)
              for rank, x in enumerate(top, 1) if x in gold)
    ideal = min(len(gold), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal + 1))
    return {
        "recall_at_k": hits / len(gold),
        "precision_at_k": hits / k,
        "mrr": rr,
        "ndcg_at_k": dcg / idcg if idcg else 0.0
    }


def aggregate(rows):
    names = ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k")
    out = {}
    for name in names:
        vals = [r[name] for r in rows if r is not None and r.get(name) is not None]
        out[name + "_mean"] = float(np.mean(vals)) if vals else None
        out[name + "_n"] = len(vals)
    return out


def text_cache(docs, model, cache_dir, batch_size, software):
    emb_path = cache_dir / "text_table_embeddings.npy"
    meta_path = cache_dir / "text_table_metadata.json"
    fp = fingerprint([{"doc_id": d["doc_id"], "text": d["text"]} for d in docs])
    dim = int(model.get_sentence_embedding_dimension())
    revision = st_revision(model)
    matrix = None

    # Preserve legacy v1 cache if model, document count, content and shape match.
    if emb_path.exists() and meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if (old.get("model") == TEXT_MODEL and
                old.get("document_count") == len(docs) and
                old.get("content_fingerprint") == fp):
                arr = np.load(emb_path, mmap_mode="r")
                if arr.shape == (len(docs), dim) and np.issubdtype(arr.dtype, np.floating):
                    if np.isfinite(arr).all():
                        matrix = np.asarray(arr, dtype=np.float32)
                del arr
        except Exception as e:
            print("Text cache validation issue:", repr(e))

    meta = {
        "cache_version": CACHE_VERSION, "model": TEXT_MODEL,
        "model_revision": revision, "document_count": len(docs),
        "embedding_dimension": dim, "content_fingerprint": fp,
        "software_versions": software,
        "validation": "model/count/content fingerprint/shape/dtype/finite values"
    }
    if matrix is not None:
        write_json(meta_path, meta)
        print("Text/table cache HIT:", matrix.shape, "(legacy cache validated and metadata upgraded)")
        return matrix, 0.0, True, meta

    print("Text/table cache MISS: encoding corpus.")
    start = time.time()
    matrix = model.encode([d["text"] for d in docs], batch_size=batch_size,
                          convert_to_numpy=True, normalize_embeddings=True,
                          show_progress_bar=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (len(docs), dim) or not np.isfinite(matrix).all():
        raise RuntimeError("Invalid text embedding matrix: " + str(matrix.shape))
    elapsed = time.time() - start
    np.save(emb_path, matrix)
    write_json(meta_path, meta)
    return matrix, elapsed, False, meta


def encode_images(docs, processor, model, device, batch_size):
    import torch
    from PIL import Image
    vectors, valid, failures = [], [], []
    for start in tqdm(range(0, len(docs), batch_size), desc="Encoding images"):
        batch = docs[start:start + batch_size]
        images, usable = [], []
        for doc in batch:
            try:
                with Image.open(doc["image_path"]) as im:
                    images.append(im.convert("RGB"))
                usable.append(doc)
            except Exception as e:
                failures.append({"doc_id": doc["doc_id"], "path": doc.get("image_path"),
                                 "error": str(e)[:300]})
        if not images:
            continue
        try:
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.vision_model(pixel_values=inputs["pixel_values"])
                feat = model.visual_projection(out.pooler_output)
                feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
            vectors.extend(feat.cpu().numpy().astype(np.float32))
            valid.extend(usable)
        except Exception as e:
            for doc in usable:
                failures.append({"doc_id": doc["doc_id"], "path": doc.get("image_path"),
                                 "error": ("Batch encoding: " + str(e))[:300]})
        finally:
            for im in images:
                im.close()
    matrix = np.asarray(vectors, dtype=np.float32) if vectors else np.empty((0, 512), np.float32)
    return matrix, valid, failures


def image_cache(docs, processor, model, device, cache_dir, batch_size, software):
    path = cache_dir / "image_embeddings.npz"
    cached = {}
    cache_read = False
    revision = clip_revision(model)

    # Old v1 image cache is intentionally rejected: it has no file signatures.
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                required = {"doc_ids", "signatures", "embeddings", "model"}
                if required.issubset(z.files) and str(z["model"].item()) == CLIP_MODEL:
                    ids, sigs, vecs = z["doc_ids"].tolist(), z["signatures"].tolist(), z["embeddings"]
                    if vecs.ndim == 2 and vecs.shape[1] == 512 and len(ids) == len(sigs) == len(vecs):
                        for i, doc_id in enumerate(ids):
                            cached[doc_id] = (json.loads(sigs[i]), vecs[i].astype(np.float32))
                        cache_read = True
        except Exception as e:
            print("Image cache will be rebuilt:", repr(e))

    signatures, changed, signature_failures = {}, [], []
    for doc in docs:
        try:
            sig = file_signature(doc["image_path"])
            signatures[doc["doc_id"]] = sig
            if doc["doc_id"] not in cached or cached[doc["doc_id"]][0] != sig:
                changed.append(doc)
        except Exception as e:
            signature_failures.append({"doc_id": doc["doc_id"],
                                       "path": doc.get("image_path"), "error": str(e)[:300]})

    print("Images requested:", len(docs))
    print("Unchanged cached images:", len(docs) - len(changed) - len(signature_failures))
    print("Images to encode:", len(changed))
    start = time.time()
    new_matrix, new_docs, failures = encode_images(changed, processor, model, device, batch_size)
    elapsed = time.time() - start
    for doc, vec in zip(new_docs, new_matrix):
        cached[doc["doc_id"]] = (signatures[doc["doc_id"]], vec)

    out_docs, out_vectors = [], []
    for doc in docs:
        did = doc["doc_id"]
        if did in signatures and did in cached and cached[did][0] == signatures[did]:
            out_docs.append(doc)
            out_vectors.append(cached[did][1])
    matrix = np.stack(out_vectors).astype(np.float32) if out_vectors else np.empty((0, 512), np.float32)
    sig_json = [json.dumps(signatures[d["doc_id"]], sort_keys=True) for d in out_docs]
    np.savez(path, doc_ids=np.asarray([d["doc_id"] for d in out_docs], dtype=str),
             signatures=np.asarray(sig_json, dtype=str), embeddings=matrix,
             model=np.asarray(CLIP_MODEL), model_revision=np.asarray(revision or ""))
    write_json(cache_dir / "image_cache_metadata.json", {
        "cache_version": CACHE_VERSION, "model": CLIP_MODEL, "model_revision": revision,
        "embedding_dimension": 512, "cached_image_count": len(out_docs),
        "signature_type": "resolved path + file size + mtime_ns (not content hash)",
        "software_versions": software
    })
    return matrix, out_docs, signature_failures + failures, elapsed, cache_read


def clip_text(question, processor, model, device):
    import torch
    inputs = processor(text=[question], return_tensors="pt",
                       padding=True, truncation=True).to(device)
    with torch.inference_mode():
        out = model.text_model(input_ids=inputs["input_ids"],
                               attention_mask=inputs["attention_mask"])
        feat = model.text_projection(out.pooler_output)
        feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
    return feat[0].cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--image-root", default="data/multimodalqa/images/final_dataset_images",
                   help="Compatibility argument; image_path in corpus.jsonl is used.")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--text-batch-size", type=int, default=16)
    p.add_argument("--image-batch-size", type=int, default=4)
    p.add_argument("--max-images", type=int, default=0)
    a = p.parse_args()
    if min(a.top_k, a.text_batch_size, a.image_batch_size) <= 0 or min(a.max_queries, a.max_images) < 0:
        raise ValueError("K and batch sizes must be positive; limits cannot be negative.")

    data_dir, out_dir, cache_dir = Path(a.data_dir), Path(a.output_dir), Path(a.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    corpus_path, query_path = data_dir / "corpus.jsonl", data_dir / "dev_queries.jsonl"
    if (not corpus_path.is_file() or not query_path.is_file()) and (data_dir / "processed" / "corpus.jsonl").is_file():
        data_dir = data_dir / "processed"
        corpus_path, query_path = data_dir / "corpus.jsonl", data_dir / "dev_queries.jsonl"
    if not corpus_path.is_file() or not query_path.is_file():
        raise FileNotFoundError("Expected corpus.jsonl and dev_queries.jsonl under " + str(data_dir))

    import torch
    from PIL import ImageFile
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    print("Loading corpus and queries...")
    docs, all_queries = read_jsonl(corpus_path), read_jsonl(query_path)
    queries = all_queries[:a.max_queries] if a.max_queries else all_queries
    text_docs = [d for d in docs if d.get("modality") in ("text", "table") and d.get("text", "").strip()]
    all_image_docs = [d for d in docs if d.get("modality") == "image"]
    image_docs = all_image_docs[:a.max_images] if a.max_images else all_image_docs
    modalities = {d["doc_id"]: d["modality"] for d in docs}
    if len(modalities) != len(docs):
        raise ValueError("Duplicate doc_id values in corpus.")

    software = versions()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Text/table docs:", len(text_docs), "| selected images:", len(image_docs), "| queries:", len(queries))
    print("Device:", device, "| GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    corpus_fp = fingerprint([{"doc_id": d.get("doc_id"), "modality": d.get("modality"),
                               "text": d.get("text"), "image_path": d.get("image_path")} for d in docs])
    query_fp = fingerprint(all_queries)

    print("\nLoading Sentence-BERT...")
    st_model = SentenceTransformer(TEXT_MODEL, device=device)
    text_matrix, text_time, text_hit, text_meta = text_cache(
        text_docs, st_model, cache_dir, a.text_batch_size, software)

    print("\nLoading CLIP...")
    cm = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    cp = CLIPProcessor.from_pretrained(CLIP_MODEL)
    image_matrix, valid_images, failures, image_time, image_cache_read = image_cache(
        image_docs, cp, cm, device, cache_dir, a.image_batch_size, software)
    print("Usable images:", len(valid_images), "| image failures:", len(failures))

    metric_rows, candidate_rows = [], []
    retrieval_start = time.time()
    limit = a.top_k * CANDIDATE_MULTIPLIER
    for q in tqdm(queries, desc="Retrieval and evaluation"):
        question, gold = q["question"], set(q.get("gold_doc_ids", []))
        gold_tt = {x for x in gold if modalities.get(x) in ("text", "table")}
        gold_img = {x for x in gold if modalities.get(x) == "image"}

        qv = st_model.encode([question], convert_to_numpy=True,
                             normalize_embeddings=True)[0]
        if text_docs:
            order = np.argsort(-(text_matrix @ qv), kind="stable")[:limit]
            trank = [text_docs[i] for i in order]
        else:
            trank = []

        if valid_images:
            iv = clip_text(question, cp, cm, device)
            order = np.argsort(-(image_matrix @ iv), kind="stable")[:limit]
            irank = [valid_images[i] for i in order]
        else:
            irank = []

        fused = rrf([trank, irank])
        ranks = {
            "text_table": [d["doc_id"] for d in trank],
            "image": [d["doc_id"] for d in irank],
            "rrf": [x[0] for x in fused]
        }
        row = {"qid": q["qid"], "n_gold_evidence": len(gold),
               "n_gold_text_table": len(gold_tt), "n_gold_image": len(gold_img),
               "methods": {}}
        for method, pred in ranks.items():
            row["methods"][method] = {
                "all_evidence": metrics(pred, gold, a.top_k),
                "text_table_conditioned": metrics(pred, gold_tt, a.top_k),
                "image_conditioned": metrics(pred, gold_img, a.top_k)
            }
        metric_rows.append(row)
        candidate_rows.append({
            "qid": q["qid"], "question": question,
            "text_table_candidates": [{"doc_id": d["doc_id"], "rank": i}
                                      for i, d in enumerate(trank, 1)],
            "image_candidates": [{"doc_id": d["doc_id"], "rank": i}
                                 for i, d in enumerate(irank, 1)],
            "candidates": [{"doc_id": did, "rrf_score": float(score)}
                           for did, score in fused[:limit]]
        })
    retrieval_time = time.time() - retrieval_start

    write_jsonl(out_dir / "baseline_query_metrics.jsonl", metric_rows)
    write_jsonl(out_dir / "candidates.jsonl", candidate_rows)
    write_json(out_dir / "failed_images.json", failures)

    metric_summary = {}
    for method in ("text_table", "image", "rrf"):
        metric_summary[method] = {}
        for scope in ("all_evidence", "text_table_conditioned", "image_conditioned"):
            metric_summary[method][scope] = aggregate([
                row["methods"][method][scope] for row in metric_rows
            ])

    manifest = {
        "experiment_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "software_versions": software,
        "models": {
            "text_table": {"name": TEXT_MODEL, "revision": st_revision(st_model)},
            "image": {"name": CLIP_MODEL, "revision": clip_revision(cm)}
        },
        "dataset": {
            "corpus_fingerprint_sha256": corpus_fp,
            "full_dev_query_fingerprint_sha256": query_fp,
            "corpus_documents": len(docs), "text_table_documents": len(text_docs),
            "image_documents_in_corpus": len(all_image_docs),
            "image_documents_selected": len(image_docs),
            "dev_queries_total": len(all_queries), "queries_evaluated": len(queries)
        },
        "evaluation": {
            "k": a.top_k, "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "candidate_limit_per_modality": limit, "rrf_k": RRF_K,
            "relevance": "binary doc_id match against gold_doc_ids",
            "precision_denominator": "K",
            "text_table_conditioned": "queries with at least one gold text/table item; gold restricted to text/table",
            "image_conditioned": "queries with at least one gold image item; gold restricted to image"
        },
        "cache": {
            "text_table_cache_hit": text_hit, "text_table_cache_metadata": text_meta,
            "image_cache_file_read": image_cache_read,
            "image_embeddings_available": len(valid_images),
            "image_signature_type": "resolved path + size + mtime_ns; not content hash"
        },
        "timings_seconds": {"text_table_embedding": text_time,
                            "image_embedding": image_time,
                            "retrieval_evaluation": retrieval_time},
        "image_failures_this_run": len(failures),
        "metrics": metric_summary,
        "arguments": vars(a)
    }
    write_json(out_dir / "baseline_summary.json", manifest)
    write_json(out_dir / "experiment_manifest.json", manifest)
    print("\nSummary:")
    print(json.dumps(manifest, indent=2))
    print("\nSaved results in:", out_dir.resolve())
    print("Cache directory:", cache_dir.resolve())


if __name__ == "__main__":
    main()
