from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from datasets import Dataset, DatasetDict

from obqa_interpretable_stats import _load_obqa_split, classify_obqa


def _balanced_targets(total: int, n: int) -> List[int]:
    base = total // n
    rem = total % n
    return [base + 1 if i < rem else base for i in range(n)]


def _stable_int(seed: int, text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


def _shuffle_copy(values: Sequence[int], seed: int, tag: str) -> List[int]:
    out = list(values)
    rng = random.Random(_stable_int(seed, tag))
    rng.shuffle(out)
    return out


def _make_parts(category_to_indices: Dict[str, List[int]], chunk_target: int, seed: int) -> List[Dict]:
    parts: List[Dict] = []
    for category, indices in sorted(category_to_indices.items()):
        shuffled = _shuffle_copy(indices, seed, category)
        n_parts = max(1, math.ceil(len(shuffled) / max(1, chunk_target)))
        sizes = _balanced_targets(len(shuffled), n_parts)
        cursor = 0
        for part_id, size in enumerate(sizes, start=1):
            part_indices = shuffled[cursor : cursor + size]
            cursor += size
            parts.append(
                {
                    "category": category,
                    "label": f"{category}__p{part_id}" if n_parts > 1 else category,
                    "indices": part_indices,
                }
            )
    parts.sort(key=lambda x: (-len(x["indices"]), x["label"]))
    return parts


def _make_empty_slices(num_slices: int, known_targets: Sequence[int], unknown_targets: Sequence[int]) -> List[Dict]:
    out = []
    for i in range(num_slices):
        out.append(
            {
                "slice_id": i,
                "known_target": int(known_targets[i]),
                "unknown_target": int(unknown_targets[i]),
                "known_indices": [],
                "unknown_indices": [],
                "source_counts": Counter(),
                "part_labels": [],
            }
        )
    return out


def _remaining_known_capacity(slice_info: Dict) -> int:
    return int(slice_info["known_target"]) - len(slice_info["known_indices"])


def _assign_part_to_slices(part: Dict, slices: List[Dict]) -> None:
    size = len(part["indices"])
    category = str(part["category"])
    candidates = []
    for sl in slices:
        remain = _remaining_known_capacity(sl)
        if remain >= size:
            # Prefer a tight fit, then keeping source categories compact.
            candidates.append(
                (
                    remain - size,
                    0 if category in sl["source_counts"] else 1,
                    len(sl["source_counts"]),
                    len(sl["known_indices"]),
                    int(sl["slice_id"]),
                )
            )

    if candidates:
        best_key = min(candidates)
        best_slice = slices[int(best_key[-1])]
        best_slice["known_indices"].extend(part["indices"])
        best_slice["source_counts"][category] += size
        best_slice["part_labels"].append(f"{part['label']}:{size}")
        return

    if size <= 1:
        raise RuntimeError("Failed to fit a 1-sample known part into any slice. Try fewer slices.")

    mid = size // 2
    left = {"category": category, "label": f"{part['label']}a", "indices": part["indices"][:mid]}
    right = {"category": category, "label": f"{part['label']}b", "indices": part["indices"][mid:]}
    _assign_part_to_slices(left, slices)
    _assign_part_to_slices(right, slices)


def _format_slice_name(source_counts: Counter) -> str:
    if not source_counts:
        return "unknown_only"
    ordered = sorted(source_counts.items(), key=lambda x: (-x[1], x[0]))
    top = [name for name, _ in ordered[:3]]
    return "+".join(top)


def _write_summary(summary_rows: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "slice_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "slice_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "slice_id",
            "slice_name",
            "total_size",
            "known_size",
            "unknown_size",
            "top_categories",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "slice_id": row["slice_id"],
                    "slice_name": row["slice_name"],
                    "total_size": row["total_size"],
                    "known_size": row["known_size"],
                    "unknown_size": row["unknown_size"],
                    "top_categories": "; ".join(
                        f"{name}:{count}" for name, count in row["source_counts"]
                    ),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build interpretable balanced OBQA slices using the full training split.")
    ap.add_argument("--num_slices", type=int, default=12, help="Final number of slices. Default: 12.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=str, default="train", choices=["train"], help="Only train is supported for slicing.")
    ap.add_argument("--out_dir", type=str, default="./slice_data/obqa_interpretable_full")
    ap.add_argument("--save_labels_csv", type=str, default="")
    args = ap.parse_args()

    if int(args.num_slices) < 11:
        raise ValueError("--num_slices must be at least 11 to satisfy the requested 10+ slice setup.")

    ds = _load_obqa_split(args.split)
    n = len(ds)
    print(f"[Load] split={args.split} size={n}")

    idx_to_category: Dict[int, str] = {}
    category_to_indices: Dict[str, List[int]] = defaultdict(list)
    label_rows: List[Dict[str, str]] = []
    for idx, ex in enumerate(ds):
        category, hits = classify_obqa(ex)
        idx_to_category[idx] = category
        category_to_indices[category].append(idx)
        if args.save_labels_csv:
            label_rows.append(
                {
                    "id": str(ex.get("id", "")),
                    "primary_category": category,
                    "matched_terms": "|".join(sorted(set(hits.get(category, [])))),
                }
            )

    unknown_indices = _shuffle_copy(category_to_indices.pop("other_unknown", []), int(args.seed), "other_unknown")
    known_total = sum(len(v) for v in category_to_indices.values())
    unknown_total = len(unknown_indices)
    print(f"[Categories] known={known_total} unknown={unknown_total} known_groups={len(category_to_indices)}")

    slice_targets = _balanced_targets(n, int(args.num_slices))
    unknown_targets = _balanced_targets(unknown_total, int(args.num_slices))
    known_targets = [slice_targets[i] - unknown_targets[i] for i in range(int(args.num_slices))]
    chunk_target = max(64, math.ceil(known_total / (2 * int(args.num_slices))))
    print(f"[Plan] num_slices={args.num_slices} final_targets={slice_targets[0]}..{slice_targets[-1]} "
          f"unknown_per_slice={unknown_targets[0]}..{unknown_targets[-1]} known_per_slice={known_targets[0]}..{known_targets[-1]} "
          f"chunk_target={chunk_target}")

    parts = _make_parts(category_to_indices, chunk_target=chunk_target, seed=int(args.seed))
    print(f"[Plan] initial_known_parts={len(parts)}")

    slices = _make_empty_slices(int(args.num_slices), known_targets, unknown_targets)
    for part in parts:
        _assign_part_to_slices(part, slices)

    known_assigned = sum(len(sl["known_indices"]) for sl in slices)
    if known_assigned != known_total:
        raise RuntimeError(f"Known assignment mismatch: expected {known_total}, got {known_assigned}")
    for sl in slices:
        if len(sl["known_indices"]) != int(sl["known_target"]):
            raise RuntimeError(
                f"Slice {sl['slice_id']} known count mismatch: "
                f"{len(sl['known_indices'])} vs target {sl['known_target']}"
            )

    cursor = 0
    for sl in slices:
        take = int(sl["unknown_target"])
        sl["unknown_indices"] = unknown_indices[cursor : cursor + take]
        cursor += take
        sl["source_counts"]["other_unknown"] += take
    if cursor != unknown_total:
        raise RuntimeError(f"Unknown assignment mismatch: expected {unknown_total}, got {cursor}")

    ordered_indices: List[int] = []
    summary_rows: List[Dict] = []

    for sl in slices:
        combined = list(sl["known_indices"]) + list(sl["unknown_indices"])
        combined.sort(key=lambda idx: (idx_to_category[idx], str(ds[int(idx)].get("id", ""))))
        slice_name = _format_slice_name(sl["source_counts"])
        unknown_size = len(sl["unknown_indices"])
        known_size = len(sl["known_indices"])

        for idx in combined:
            ordered_indices.append(idx)

        summary_rows.append(
            {
                "slice_id": int(sl["slice_id"]),
                "slice_name": slice_name,
                "total_size": len(combined),
                "known_size": known_size,
                "unknown_size": unknown_size,
                "source_counts": sorted(sl["source_counts"].items(), key=lambda x: (-x[1], x[0])),
                "part_labels": list(sl["part_labels"]),
            }
        )

    ordered_rows: List[Dict] = []
    for row_pos, idx in enumerate(ordered_indices):
        row = dict(ds[int(idx)])
        row["slice_id"] = int(summary_rows[0]["slice_id"])  # placeholder overwritten below
        ordered_rows.append(row)

    cursor = 0
    for row in summary_rows:
        take = int(row["total_size"])
        for pos in range(cursor, cursor + take):
            ordered_rows[pos]["slice_id"] = int(row["slice_id"])
            ordered_rows[pos]["slice_name"] = str(row["slice_name"])
            ordered_rows[pos]["primary_category"] = idx_to_category[ordered_indices[pos]]
        cursor += take

    ordered_ds: Dataset = Dataset.from_list(ordered_rows)

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)
    out_kfac = os.path.join(out_root, "kfac_balanced")
    DatasetDict({"train": ordered_ds}).save_to_disk(out_kfac)
    _write_summary(summary_rows, out_root)

    if args.save_labels_csv:
        with open(args.save_labels_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "primary_category", "matched_terms"])
            writer.writeheader()
            writer.writerows(label_rows)

    print(f"[Save] kfac_balanced -> {out_kfac}")
    print("[Slice Summary]")
    for row in summary_rows:
        top = ", ".join(f"{name}:{count}" for name, count in row["source_counts"][:4])
        print(
            f"slice={row['slice_id']:02d}  size={row['total_size']:4d}  "
            f"known={row['known_size']:4d}  unknown={row['unknown_size']:3d}  "
            f"name={row['slice_name']}  top={top}"
        )


if __name__ == "__main__":
    main()
