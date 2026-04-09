from __future__ import annotations
import os
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA

# =========================================================
# Utilities
# =========================================================

def l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def _choices_obj_to_mapping(choices_obj) -> Dict[str, str]:
    if isinstance(choices_obj, dict):
        labels = choices_obj.get("label", None)
        texts = choices_obj.get("text", None)
        if isinstance(labels, list) and isinstance(texts, list):
            return {str(l): str(t) for l, t in zip(labels, texts)}
        out = {}
        for k, v in choices_obj.items():
            if str(k) in ["A", "B", "C", "D"]:
                out[str(k)] = str(v)
        return out
    if isinstance(choices_obj, list):
        out = {}
        for c in choices_obj:
            if isinstance(c, dict):
                if "label" in c and "text" in c:
                    out[str(c["label"])] = str(c["text"])
                elif "key" in c and "value" in c:
                    out[str(c["key"])] = str(c["value"])
        return out
    return {}

def _safe_str(x) -> str:
    if x is None: return ""
    return str(x).strip()


# =========================================================
# Dataset loader (Unified Task List)
# =========================================================

def load_task_split(dataset_name: str, split: str) -> Dataset:
    if dataset_name == "wgs": return load_dataset("winogrande", "winogrande_s", split=split)
    elif dataset_name == "wgm": return load_dataset("winogrande", "winogrande_m", split=split)
    elif dataset_name == "arc-c": return load_dataset("ai2_arc", "ARC-Challenge", split=split)
    elif dataset_name == "arc-e": return load_dataset("ai2_arc", "ARC-Easy", split=split)
    elif dataset_name == "obqa": return load_dataset("openbookqa", "main", split=split)
    elif dataset_name == "boolq": return load_dataset("super_glue", "boolq", split=split)
    elif dataset_name == "sciq": return load_dataset("sciq", split=split)
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


# =========================================================
# Formatting for Embedding
# =========================================================

def format_wg(ex) -> Optional[str]:
    sent = _safe_str(ex.get("sentence", ""))
    o1 = _safe_str(ex.get("option1", ""))
    o2 = _safe_str(ex.get("option2", ""))
    if not sent or not o1 or not o2: return None
    return f"{sent} (A) {o1} (B) {o2}".strip()

def format_boolq(ex) -> Optional[str]:
    q = _safe_str(ex.get("question", ""))
    p = _safe_str(ex.get("passage", ""))
    if not q or not p: return None
    return f"{p} Question: {q} (A) False (B) True".strip()

def format_obqa(ex) -> Optional[str]:
    q = _safe_str(ex.get("question_stem", ""))
    mapping = _choices_obj_to_mapping(ex.get("choices", {}))
    if not all(k in mapping for k in ["A", "B", "C", "D"]): return None
    formatted_choices = " ".join([f"({k}) {mapping[k]}" for k in ["A", "B", "C", "D"]])
    return f"{q} {formatted_choices}".strip()

def format_arc(ex) -> Optional[str]:
    if "question" in ex and isinstance(ex["question"], dict):
        q = _safe_str(ex["question"].get("stem", ex["question"].get("text", "")))
        mapping = _choices_obj_to_mapping(ex["question"].get("choices", {}))
    else:
        q = _safe_str(ex.get("question", ex.get("question_stem", "")))
        mapping = _choices_obj_to_mapping(ex.get("choices", {}))
    if not all(k in mapping for k in ["A", "B", "C", "D"]): return None
    formatted_choices = " ".join([f"({k}) {mapping[k]}" for k in ["A", "B", "C", "D"]])
    return f"{q} {formatted_choices}".strip()

def format_sciq(ex, deterministic_shuffle: bool = False) -> Optional[str]:
    q, correct = _safe_str(ex.get("question", "")), _safe_str(ex.get("correct_answer", ""))
    d1, d2, d3 = _safe_str(ex.get("distractor1", "")), _safe_str(ex.get("distractor2", "")), _safe_str(ex.get("distractor3", ""))
    opts = [d1, d2, d3, correct]
    if any(x == "" for x in opts): return None

    if deterministic_shuffle:
        rng = random.Random(abs(hash(q)) % (10**8))
        rng.shuffle(opts)
    mapping = {"A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3]}
    formatted_choices = " ".join([f"({k}) {mapping[k]}" for k in ["A", "B", "C", "D"]])
    return f"{q} {formatted_choices}".strip()

def format_example_for_embed(ex, dataset_name: str, sciq_shuffle: bool = False) -> Optional[str]:
    if dataset_name in ["wgs", "wgm"]: return format_wg(ex)
    elif dataset_name == "boolq": return format_boolq(ex)
    elif dataset_name == "obqa": return format_obqa(ex)
    elif dataset_name in ["arc-e", "arc-c"]: return format_arc(ex)
    elif dataset_name == "sciq": return format_sciq(ex, deterministic_shuffle=sciq_shuffle)
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


# =========================================================
# K selection & Ordering
# =========================================================

def pick_k_by_silhouette(embeddings: np.ndarray, k_min: int, k_max: int, seed: int, min_samples_per_cluster: int, sil_sample: int = 2000) -> int:
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    emb_s = embeddings[rng.choice(n, size=sil_sample, replace=False)] if n > sil_sample else embeddings

    best_k, best_score = -1, -1e9
    print(f"[K-search] silhouette on sample size={emb_s.shape[0]} over K=[{k_min}..{k_max}]")

    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb_s)
        min_c = int(np.min(np.bincount(labels)))
        if min_c < min_samples_per_cluster:
            print(f"  K={k}: skip (min cluster {min_c} < {min_samples_per_cluster})")
            continue
        score = silhouette_score(emb_s, labels, metric="cosine")
        print(f"  K={k}: silhouette={score:.4f}  (min_size={min_c})")
        if score > best_score: best_score, best_k = score, k

    if best_k < 0: raise RuntimeError("No K satisfies min cluster size constraint. Lower --min_samples_per_cluster.")
    print(f"[K-search] Winner K={best_k} silhouette={best_score:.4f}")
    return best_k

def order_clusters(centers: np.ndarray, mode: str = "pca", seed: int = 42) -> List[int]:
    K = centers.shape[0]
    if K <= 1: return [0]

    if mode == "max_drift":
        dist = cosine_distances(centers)
        visited, cur = [0], 0
        for _ in range(K - 1):
            d = dist[cur].copy()
            d[visited] = -1.0
            cur = int(np.argmax(d))
            visited.append(cur)
        return visited

    if mode == "pca":
        z = PCA(n_components=1, random_state=seed).fit_transform(centers).reshape(-1)
        return list(np.argsort(z).astype(int))

    if mode == "mst":
        dist = cosine_distances(centers)
        in_mst, parent, key = np.zeros(K, dtype=bool), -np.ones(K, dtype=int), np.full(K, np.inf)
        key[0] = 0.0
        for _ in range(K):
            u = int(np.argmin(np.where(in_mst, np.inf, key)))
            in_mst[u] = True
            for v in range(K):
                if not in_mst[v] and dist[u, v] < key[v]:
                    key[v], parent[v] = dist[u, v], u
        adj = [[] for _ in range(K)]
        for v in range(1, K):
            u = parent[v]
            if u >= 0: adj[u].append(v); adj[v].append(u)
        order, stack, seen = [], [0], np.zeros(K, dtype=bool)
        while stack:
            u = stack.pop()
            if seen[u]: continue
            seen[u] = True
            order.append(u)
            for v in sorted(adj[u], reverse=True):
                if not seen[v]: stack.append(v)
        return order

    raise ValueError(f"Unknown order mode: {mode}")

def print_centroid_distance_matrix(centers: np.ndarray, names: Optional[List[str]] = None) -> None:
    dist = cosine_distances(centers)
    K = dist.shape[0]
    names = names or [f"s{i}" for i in range(K)]
    print("\n[Drift Check] Cosine distance between cluster centroids (after ordering):")
    print(" " * 14 + " ".join([f"{n:>12s}" for n in names]))
    for i in range(K):
        print(f"{names[i]:>12s}  " + " ".join([f"{dist[i, j]:12.3f}" for j in range(K)]))


# =========================================================
# Preprocess + save helpers
# =========================================================

def build_embedding_texts(ds: Dataset, dataset_name: str, sciq_shuffle: bool = False) -> Tuple[List[str], List[int]]:
    texts, valid_indices, dropped = [], [], 0
    for i in range(len(ds)):
        txt = format_example_for_embed(ds[i], dataset_name=dataset_name, sciq_shuffle=sciq_shuffle)
        if not txt: dropped += 1; continue
        texts.append(txt)
        valid_indices.append(i)
    print(f"[Prep] kept={len(texts)} dropped={dropped}")
    if not texts: raise RuntimeError("No valid examples after formatting.")
    return texts, valid_indices


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, required=True, choices=["wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq", "sciq"])
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--k_min", type=int, default=6)
    ap.add_argument("--k_max", type=int, default=20)
    ap.add_argument("--min_samples_per_cluster", type=int, default=128)
    ap.add_argument("--order", type=str, default="pca", choices=["pca", "mst", "max_drift"])

    ap.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--embed_batch_size", type=int, default=128)
    ap.add_argument("--sil_sample", type=int, default=6000)
    ap.add_argument("--sciq_shuffle_choices", action="store_true")

    ap.add_argument("--save_full_train", action="store_true", default=True)
    ap.add_argument("--save_kfac_balanced", action="store_true", default=True)
    ap.add_argument("--kfac_per_slice", type=int, default=128)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Load] dataset_name={args.dataset_name} split={args.split}")
    ds_raw = load_task_split(args.dataset_name, args.split)
    print(f"[Load] raw size={len(ds_raw)}")

    print("[Prep] formatting texts for SBERT...")
    texts, valid_indices = build_embedding_texts(ds_raw, dataset_name=args.dataset_name, sciq_shuffle=args.sciq_shuffle_choices)
    ds = ds_raw.select(valid_indices)
    n = len(ds)

    print(f"[Embed] SentenceTransformer={args.sbert_model}")
    st = SentenceTransformer(args.sbert_model)
    embs = l2_normalize(st.encode(texts, batch_size=args.embed_batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32))

    K = pick_k_by_silhouette(embs, k_min=args.k_min, k_max=args.k_max, seed=args.seed, min_samples_per_cluster=args.min_samples_per_cluster, sil_sample=args.sil_sample)

    print(f"[Cluster] fitting KMeans K={K} on full data...")
    labels = KMeans(n_clusters=K, random_state=args.seed, n_init=10).fit_predict(embs)
    centers = l2_normalize(KMeans(n_clusters=K, random_state=args.seed, n_init=10).fit(embs).cluster_centers_.astype(np.float32))
    
    counts = np.bincount(labels, minlength=K).tolist()
    print(f"[Cluster] raw cluster sizes: {counts}")
    if min(counts) < args.min_samples_per_cluster:
        print("[Warn] full-data clustering violates min_samples_per_cluster. Sampling-based K selection may differ from full-data result.")

    order = order_clusters(centers, mode=args.order, seed=args.seed)
    cluster_map = {old: new for new, old in enumerate(order)}
    print(f"[Order] mode={args.order} order(old_cluster_ids)={order}")
    print_centroid_distance_matrix(centers[order], names=[f"slice{i}" for i in range(K)])

    slice_indices = {sid: [] for sid in range(K)}
    for idx, old_c in enumerate(labels): slice_indices[cluster_map[int(old_c)]].append(idx)
    for sid in range(K): slice_indices[sid] = sorted(slice_indices[sid])

    if args.save_full_train:
        slice_id_col, raw_cluster_col = np.empty(n, dtype=np.int32), labels.astype(np.int32)
        for sid in range(K): slice_id_col[slice_indices[sid]] = sid
        ds_full = ds.add_column("slice_id", slice_id_col.tolist()).add_column("raw_cluster_id", raw_cluster_col.tolist())
        out_full = os.path.join(args.out_dir, "full_train")
        DatasetDict({"train": ds_full.select(np.argsort(slice_id_col).tolist())}).save_to_disk(out_full)
        print(f"[Save] full_train -> {out_full}")

    if args.save_kfac_balanced:
        per, kept = int(args.kfac_per_slice), []
        for sid in range(K):
            idxs = slice_indices[sid]
            if len(idxs) < per: raise RuntimeError(f"Slice {sid} has only {len(idxs)} samples, but kfac_per_slice={per}. Lower it!")
            kept.extend([idxs[i] for i in rng.choice(len(idxs), size=per, replace=False)])
        
        kept = sorted(kept)
        ds_kfac = ds.select(kept).add_column("slice_id", [cluster_map[int(labels[idx])] for idx in kept]).add_column("raw_cluster_id", [int(labels[idx]) for idx in kept])
        out_kfac = os.path.join(args.out_dir, "kfac_balanced")
        DatasetDict({"train": ds_kfac.select(np.argsort(np.array(ds_kfac["slice_id"])).tolist())}).save_to_disk(out_kfac)
        print(f"[Save] kfac_balanced -> {out_kfac}")

    print("\n[Done]")

if __name__ == "__main__":
    main()