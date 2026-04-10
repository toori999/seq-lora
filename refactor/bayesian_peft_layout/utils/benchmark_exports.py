from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import csv
import json
import statistics

from .benchmark_config import EXCLUDED_STATUS_PREFIXES
from .benchmark_parsing import augment_result_row_with_infer_peaks


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_or_none(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _stdev_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def summarize_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["report_task"])), []).append(row)

    out: List[Dict[str, object]] = []
    for (method, report_task), group_rows in sorted(grouped.items()):
        infer_vals = [float(r["infer_time_sec"]) for r in group_rows if r["infer_time_sec"] is not None]
        infer_peak_alloc_vals = [
            float(r["infer_peak_alloc_gb"]) for r in group_rows if r.get("infer_peak_alloc_gb") is not None
        ]
        infer_peak_reserved_vals = [
            float(r["infer_peak_reserved_gb"]) for r in group_rows if r.get("infer_peak_reserved_gb") is not None
        ]
        train_vals = [float(r["train_time_sec"]) for r in group_rows if r["train_time_sec"] is not None]
        out.append(
            {
                "method": method,
                "report_task": report_task,
                "n_seeds": len(group_rows),
                "seed_list": ",".join(str(int(r["seed"])) for r in group_rows),
                "nll_mean": statistics.mean(float(r["nll"]) for r in group_rows),
                "nll_sd": statistics.stdev(float(r["nll"]) for r in group_rows) if len(group_rows) > 1 else 0.0,
                "acc_pct_mean": statistics.mean(float(r["acc_pct"]) for r in group_rows),
                "acc_pct_sd": statistics.stdev(float(r["acc_pct"]) for r in group_rows) if len(group_rows) > 1 else 0.0,
                "ece_pct_mean": statistics.mean(float(r["ece_pct"]) for r in group_rows),
                "ece_pct_sd": statistics.stdev(float(r["ece_pct"]) for r in group_rows) if len(group_rows) > 1 else 0.0,
                "brier_mean": statistics.mean(float(r["brier"]) for r in group_rows),
                "brier_sd": statistics.stdev(float(r["brier"]) for r in group_rows) if len(group_rows) > 1 else 0.0,
                "infer_time_sec_mean": _mean_or_none(infer_vals),
                "infer_time_sec_sd": _stdev_or_none(infer_vals),
                "infer_peak_alloc_gb_mean": _mean_or_none(infer_peak_alloc_vals),
                "infer_peak_alloc_gb_sd": _stdev_or_none(infer_peak_alloc_vals),
                "infer_peak_reserved_gb_mean": _mean_or_none(infer_peak_reserved_vals),
                "infer_peak_reserved_gb_sd": _stdev_or_none(infer_peak_reserved_vals),
                "train_time_sec_mean": _mean_or_none(train_vals),
                "train_time_sec_sd": _stdev_or_none(train_vals),
            }
        )

    macro_rows: List[Dict[str, object]] = []
    by_method_seed: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        if str(row["report_task"]) not in {"mmlu_science_high", "mmlu_science_college"}:
            continue
        key = (str(row["method"]), int(row["seed"]))
        by_method_seed.setdefault(key, []).append(row)
    for (method, seed), subset in sorted(by_method_seed.items()):
        if len(subset) != 2:
            continue
        macro_rows.append(
            {
                "method": method,
                "seed": seed,
                "report_task": "mmlu_macro",
                "nll": statistics.mean(float(r["nll"]) for r in subset),
                "acc_pct": statistics.mean(float(r["acc_pct"]) for r in subset),
                "ece_pct": statistics.mean(float(r["ece_pct"]) for r in subset),
                "brier": statistics.mean(float(r["brier"]) for r in subset),
                "infer_time_sec": statistics.mean(
                    float(r["infer_time_sec"]) for r in subset if r["infer_time_sec"] is not None
                ),
                "infer_peak_alloc_gb": (
                    statistics.mean(
                        float(r["infer_peak_alloc_gb"]) for r in subset if r.get("infer_peak_alloc_gb") is not None
                    )
                    if any(r.get("infer_peak_alloc_gb") is not None for r in subset)
                    else None
                ),
                "infer_peak_reserved_gb": (
                    statistics.mean(
                        float(r["infer_peak_reserved_gb"]) for r in subset if r.get("infer_peak_reserved_gb") is not None
                    )
                    if any(r.get("infer_peak_reserved_gb") is not None for r in subset)
                    else None
                ),
                "train_time_sec": (
                    statistics.mean(float(r["train_time_sec"]) for r in subset if r["train_time_sec"] is not None)
                    if any(r["train_time_sec"] is not None for r in subset)
                    else None
                ),
            }
        )
    if macro_rows:
        grouped_macro: Dict[str, List[Dict[str, object]]] = {}
        for row in macro_rows:
            grouped_macro.setdefault(str(row["method"]), []).append(row)
        for method, subset in sorted(grouped_macro.items()):
            infer_vals = [float(r["infer_time_sec"]) for r in subset if r["infer_time_sec"] is not None]
            infer_peak_alloc_vals = [
                float(r["infer_peak_alloc_gb"]) for r in subset if r.get("infer_peak_alloc_gb") is not None
            ]
            infer_peak_reserved_vals = [
                float(r["infer_peak_reserved_gb"]) for r in subset if r.get("infer_peak_reserved_gb") is not None
            ]
            train_vals = [float(r["train_time_sec"]) for r in subset if r["train_time_sec"] is not None]
            out.append(
                {
                    "method": method,
                    "report_task": "mmlu_macro",
                    "n_seeds": len(subset),
                    "seed_list": ",".join(str(int(r["seed"])) for r in subset),
                    "nll_mean": statistics.mean(float(r["nll"]) for r in subset),
                    "nll_sd": statistics.stdev(float(r["nll"]) for r in subset) if len(subset) > 1 else 0.0,
                    "acc_pct_mean": statistics.mean(float(r["acc_pct"]) for r in subset),
                    "acc_pct_sd": statistics.stdev(float(r["acc_pct"]) for r in subset) if len(subset) > 1 else 0.0,
                    "ece_pct_mean": statistics.mean(float(r["ece_pct"]) for r in subset),
                    "ece_pct_sd": statistics.stdev(float(r["ece_pct"]) for r in subset) if len(subset) > 1 else 0.0,
                    "brier_mean": statistics.mean(float(r["brier"]) for r in subset),
                    "brier_sd": statistics.stdev(float(r["brier"]) for r in subset) if len(subset) > 1 else 0.0,
                    "infer_time_sec_mean": _mean_or_none(infer_vals),
                    "infer_time_sec_sd": _stdev_or_none(infer_vals),
                    "infer_peak_alloc_gb_mean": _mean_or_none(infer_peak_alloc_vals),
                    "infer_peak_alloc_gb_sd": _stdev_or_none(infer_peak_alloc_vals),
                    "infer_peak_reserved_gb_mean": _mean_or_none(infer_peak_reserved_vals),
                    "infer_peak_reserved_gb_sd": _stdev_or_none(infer_peak_reserved_vals),
                    "train_time_sec_mean": _mean_or_none(train_vals),
                    "train_time_sec_sd": _stdev_or_none(train_vals),
                }
            )
    return out


def summarize_command_rows(command_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in sorted(command_rows, key=lambda x: str(x.get("name", ""))):
        out.append(
            {
                "name": row.get("name"),
                "seed": row.get("seed"),
                "source_order": row.get("source_order"),
                "returncode": row.get("returncode"),
                "wall_time_sec": row.get("wall_time_sec"),
                "train_time_sec": row.get("train_time_sec"),
                "train_peak_alloc_gb": row.get("train_peak_alloc_gb"),
                "train_peak_reserved_gb": row.get("train_peak_reserved_gb"),
                "log_path": row.get("log_path"),
            }
        )
    return out


def summarize_training_resources(command_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    families = {
        "seq": "seq_constantq_order",
        "laplace": "laplace_order",
        "blob": "blob_order",
    }
    out: List[Dict[str, object]] = []
    for family, prefix in families.items():
        subset = [row for row in command_rows if str(row.get("name", "")).startswith(prefix)]
        if not subset:
            continue
        train_times = [float(row["train_time_sec"]) for row in subset if row.get("train_time_sec") is not None]
        peak_allocs = [float(row["train_peak_alloc_gb"]) for row in subset if row.get("train_peak_alloc_gb") is not None]
        peak_reserved = [float(row["train_peak_reserved_gb"]) for row in subset if row.get("train_peak_reserved_gb") is not None]
        out.append(
            {
                "method_family": family,
                "n_runs": len(subset),
                "seed_list": ",".join(str(row.get("seed")) for row in subset),
                "train_time_sec_mean": _mean_or_none(train_times),
                "train_time_sec_sd": _stdev_or_none(train_times),
                "train_peak_alloc_gb_max": max(peak_allocs) if peak_allocs else None,
                "train_peak_reserved_gb_max": max(peak_reserved) if peak_reserved else None,
            }
        )
    return out


def collect_status_rows(status_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    metrics_rows: List[Dict[str, object]] = []
    command_rows: List[Dict[str, object]] = []
    for path in sorted(status_dir.glob("*.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        command_name = str(payload.get("name", ""))
        if any(command_name.startswith(prefix) for prefix in EXCLUDED_STATUS_PREFIXES):
            continue
        command_rows.append(payload)
        stage_peaks = payload.get("stage_peaks_gb", {})
        if not isinstance(stage_peaks, dict):
            stage_peaks = {}
        for row in payload.get("results", []):
            if isinstance(row, dict):
                metrics_rows.append(
                    augment_result_row_with_infer_peaks(command_name, row, stage_peaks)
                )
    return metrics_rows, command_rows


def refresh_exports(result_root: Path) -> None:
    status_dir = result_root / "status"
    metrics_rows, command_rows = collect_status_rows(status_dir)
    write_json(result_root / "all_commands.json", command_rows)
    write_json(result_root / "all_metrics.json", metrics_rows)
    write_csv(result_root / "command_times.csv", summarize_command_rows(command_rows))
    write_csv(
        result_root / "training_resource_summary.csv",
        summarize_training_resources(command_rows),
    )
    write_csv(result_root / "all_metrics.csv", metrics_rows)
    write_csv(result_root / "summary_mean_sd.csv", summarize_rows(metrics_rows))


__all__ = [
    "collect_status_rows",
    "read_json",
    "refresh_exports",
    "summarize_command_rows",
    "summarize_rows",
    "summarize_training_resources",
    "write_csv",
    "write_json",
]
