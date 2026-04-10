from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import re

from .benchmark_config import SOURCE_TASK

TIME_RE = re.compile(r"^\[TIME\] (?P<tag>.+?): (?P<sec>[0-9.]+) sec")
PEAK_RE = re.compile(
    r"^\[PEAK\] (?P<tag>.+?): alloc=(?P<alloc>[0-9.]+) GB\s+reserved=(?P<reserved>[0-9.]+) GB"
)
MAP_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\]\[MAP\]$")
MAP_METRICS_RE = re.compile(
    r"NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+ECE=(?P<ece>[0-9.]+)%\s+Brier=(?P<brier>[0-9.]+)"
)
MCDROP_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\]\[MCDROP\]$")
MCDROP_METRICS_RE = re.compile(
    r"NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+ECE=(?P<ece>[0-9.]+)%\s+"
    r"Brier=(?P<brier>[0-9.]+)\s+std=(?P<std>[0-9.]+)\s+mc=(?P<mc>[0-9.]+)"
)
SEQ_BLOCK_RE = re.compile(
    r"\[(?P<tag>[^\]]+)\]\s*\n\s*===== Bayesian \(Seq-LoRA\) Only =====\s*\n"
    r"\s*nll_bayes:\s*(?P<nll>[0-9.]+)\s*\n"
    r"\s*brier_bayes:\s*(?P<brier>[0-9.]+)\s*\n"
    r"\s*ece_bayes:\s*(?P<ece>[0-9.]+)%\s*\n"
    r"\s*acc_bayes:\s*(?P<acc>[0-9.]+)%",
    re.MULTILINE,
)
TAG_WITH_SPLIT_RE = re.compile(r"^(?P<task>.+)\((?P<split>[^()]+)\)$")
SEQ_TAG_RE = re.compile(r"^(?P<task>.+)_(?P<split>iid|ood)$")
LAP_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\]\s+n=(?P<n>[0-9]+)$")
LAP_METRICS_RE = re.compile(
    r"^(?P<method>MAP|LAP):\s+NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+"
    r"ECE=(?P<ece>[0-9.]+)%\s+Brier=(?P<brier>[0-9.]+)$"
)
BLOB_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\s+Results\]$")
BLOB_METRICS_RE = re.compile(
    r"^\s*(?P<method>MAP|BLoB mean|BLoB samp)\s*: NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+"
    r"ECE=(?P<ece>[0-9.]+)%\s+Brier=(?P<brier>[0-9.]+)(?:\s+\(N=(?P<mc>[0-9.]+)\))?$"
)
ENSEMBLE_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\]\[ENSEMBLE\]$")
ENSEMBLE_METRICS_RE = re.compile(
    r"NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+ECE=(?P<ece>[0-9.]+)%\s+"
    r"Brier=(?P<brier>[0-9.]+)(?:\s+members=(?P<members>[0-9.]+))?$"
)


def parse_stage_times(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        match = TIME_RE.match(line.strip())
        if match:
            out[match.group("tag")] = float(match.group("sec"))
    return out


def parse_stage_peaks(text: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        match = PEAK_RE.match(line.strip())
        if match:
            out[match.group("tag")] = {
                "alloc_gb": float(match.group("alloc")),
                "reserved_gb": float(match.group("reserved")),
            }
    return out


def split_parenthetical_tag(tag: str) -> Tuple[str, str]:
    match = TAG_WITH_SPLIT_RE.match(tag.strip())
    if not match:
        raise ValueError(f"Could not parse tag with split: {tag}")
    return match.group("task"), match.group("split")


def split_seq_tag(tag: str) -> Tuple[str, str]:
    match = SEQ_TAG_RE.match(tag.strip())
    if not match:
        raise ValueError(f"Could not parse seq tag: {tag}")
    return match.group("task"), match.group("split")


def normalize_report_task(eval_task: str) -> str:
    if eval_task == SOURCE_TASK:
        return "iid"
    if eval_task == "scienceqa_closedchoice_grade12":
        return "grade12"
    return eval_task


def make_result_row(
    *,
    method: str,
    seed: int,
    source_order: str,
    eval_task: str,
    split: str,
    nll: float,
    acc_pct: float,
    ece_pct: float,
    brier: float,
    infer_time_sec: Optional[float],
    infer_peak_alloc_gb: Optional[float],
    infer_peak_reserved_gb: Optional[float],
    train_time_sec: Optional[float],
    wall_time_sec: float,
    extras: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "method": method,
        "seed": int(seed),
        "source_order": source_order,
        "eval_task": eval_task,
        "report_task": normalize_report_task(eval_task),
        "split": split,
        "nll": float(nll),
        "acc": float(acc_pct) / 100.0,
        "acc_pct": float(acc_pct),
        "ece": float(ece_pct) / 100.0,
        "ece_pct": float(ece_pct),
        "brier": float(brier),
        "infer_time_sec": None if infer_time_sec is None else float(infer_time_sec),
        "infer_peak_alloc_gb": (
            None if infer_peak_alloc_gb is None else float(infer_peak_alloc_gb)
        ),
        "infer_peak_reserved_gb": (
            None if infer_peak_reserved_gb is None else float(infer_peak_reserved_gb)
        ),
        "train_time_sec": None if train_time_sec is None else float(train_time_sec),
        "command_wall_time_sec": float(wall_time_sec),
    }
    if extras:
        row.update(extras)
    return row


def _get_stage_peak(
    stage_peaks: Dict[str, Dict[str, float]],
    tag: str,
) -> Tuple[Optional[float], Optional[float]]:
    payload = stage_peaks.get(tag)
    if not payload:
        return None, None
    return payload.get("alloc_gb"), payload.get("reserved_gb")


def parse_map_eval_output(
    text: str,
    seed: int,
    source_order: str,
    wall_time_sec: float,
) -> List[Dict[str, object]]:
    stage_times = parse_stage_times(text)
    stage_peaks = parse_stage_peaks(text)
    rows: List[Dict[str, object]] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        m_header = MAP_HEADER_RE.match(line.strip())
        if not m_header or idx + 1 >= len(lines):
            continue
        tag = m_header.group("tag")
        m_metrics = MAP_METRICS_RE.search(lines[idx + 1].strip())
        if not m_metrics:
            continue
        eval_task, split = split_parenthetical_tag(tag)
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(
            stage_peaks, f"INFER MAP on {tag}"
        )
        rows.append(
            make_result_row(
                method=f"map_{source_order}",
                seed=seed,
                source_order=source_order,
                eval_task=eval_task,
                split=split,
                nll=float(m_metrics.group("nll")),
                acc_pct=float(m_metrics.group("acc")),
                ece_pct=float(m_metrics.group("ece")),
                brier=float(m_metrics.group("brier")),
                infer_time_sec=stage_times.get(f"INFER MAP on {tag}"),
                infer_peak_alloc_gb=infer_peak_alloc_gb,
                infer_peak_reserved_gb=infer_peak_reserved_gb,
                train_time_sec=None,
                wall_time_sec=wall_time_sec,
            )
        )
    return rows


def parse_mcdrop_output(
    text: str,
    seed: int,
    source_order: str,
    wall_time_sec: float,
) -> List[Dict[str, object]]:
    del source_order
    stage_times = parse_stage_times(text)
    stage_peaks = parse_stage_peaks(text)
    rows: List[Dict[str, object]] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        m_header = MCDROP_HEADER_RE.match(line.strip())
        if not m_header or idx + 1 >= len(lines):
            continue
        tag = m_header.group("tag")
        m_metrics = MCDROP_METRICS_RE.search(lines[idx + 1].strip())
        if not m_metrics:
            continue
        eval_task, split = split_parenthetical_tag(tag)
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(
            stage_peaks, f"INFER MCDrop on {tag}"
        )
        rows.append(
            make_result_row(
                method="mcdrop_order",
                seed=seed,
                source_order="order",
                eval_task=eval_task,
                split=split,
                nll=float(m_metrics.group("nll")),
                acc_pct=float(m_metrics.group("acc")),
                ece_pct=float(m_metrics.group("ece")),
                brier=float(m_metrics.group("brier")),
                infer_time_sec=stage_times.get(f"INFER MCDrop on {tag}"),
                infer_peak_alloc_gb=infer_peak_alloc_gb,
                infer_peak_reserved_gb=infer_peak_reserved_gb,
                train_time_sec=None,
                wall_time_sec=wall_time_sec,
                extras={
                    "mc_std": float(m_metrics.group("std")),
                    "mc_samples": float(m_metrics.group("mc")),
                },
            )
        )
    return rows


def parse_seq_output(
    text: str,
    seed: int,
    source_order: str,
    wall_time_sec: float,
) -> Tuple[List[Dict[str, object]], Optional[float]]:
    del source_order
    stage_times = parse_stage_times(text)
    stage_peaks = parse_stage_peaks(text)
    train_time_sec = None
    for tag, sec in stage_times.items():
        if tag.startswith("TRAIN-STAGE Seq-LoRA posterior build on "):
            train_time_sec = sec
            break

    rows: List[Dict[str, object]] = []
    for match in SEQ_BLOCK_RE.finditer(text):
        tag = match.group("tag")
        eval_task, split = split_seq_tag(tag)
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(
            stage_peaks, f"INFER Seq-LoRA on {tag}"
        )
        rows.append(
            make_result_row(
                method="seq_constantq_order",
                seed=seed,
                source_order="order",
                eval_task=eval_task,
                split=("test" if split == "iid" else split),
                nll=float(match.group("nll")),
                acc_pct=float(match.group("acc")),
                ece_pct=float(match.group("ece")),
                brier=float(match.group("brier")),
                infer_time_sec=stage_times.get(f"INFER Seq-LoRA on {tag}"),
                infer_peak_alloc_gb=infer_peak_alloc_gb,
                infer_peak_reserved_gb=infer_peak_reserved_gb,
                train_time_sec=train_time_sec,
                wall_time_sec=wall_time_sec,
            )
        )
    return rows, train_time_sec


def parse_laplace_output(
    text: str,
    seed: int,
    source_order: str,
    wall_time_sec: float,
) -> Tuple[List[Dict[str, object]], Optional[float]]:
    del source_order
    stage_times = parse_stage_times(text)
    stage_peaks = parse_stage_peaks(text)
    train_time_sec = None
    for tag, sec in stage_times.items():
        if tag.startswith("OFFICIAL SOURCE Laplace fit on "):
            train_time_sec = sec
            break

    rows: List[Dict[str, object]] = []
    current_tag: Optional[str] = None
    for line in text.splitlines():
        line_s = line.strip()
        m_header = LAP_HEADER_RE.match(line_s)
        if m_header:
            current_tag = m_header.group("tag")
            continue
        m_metrics = LAP_METRICS_RE.match(line_s)
        if not current_tag or not m_metrics:
            continue
        if m_metrics.group("method") != "LAP":
            continue
        eval_task, split = split_parenthetical_tag(current_tag)
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(
            stage_peaks, f"INFER Official-Source-Laplace on {current_tag}"
        )
        rows.append(
            make_result_row(
                method="laplace_order",
                seed=seed,
                source_order="order",
                eval_task=eval_task,
                split=split,
                nll=float(m_metrics.group("nll")),
                acc_pct=float(m_metrics.group("acc")),
                ece_pct=float(m_metrics.group("ece")),
                brier=float(m_metrics.group("brier")),
                infer_time_sec=stage_times.get(
                    f"INFER Official-Source-Laplace on {current_tag}"
                ),
                infer_peak_alloc_gb=infer_peak_alloc_gb,
                infer_peak_reserved_gb=infer_peak_reserved_gb,
                train_time_sec=train_time_sec,
                wall_time_sec=wall_time_sec,
            )
        )
    return rows, train_time_sec


def parse_blob_output(
    text: str,
    seed: int,
    source_order: str,
    wall_time_sec: float,
) -> Tuple[List[Dict[str, object]], Optional[float]]:
    del source_order
    stage_times = parse_stage_times(text)
    stage_peaks = parse_stage_peaks(text)
    train_time_sec = None
    for tag, sec in stage_times.items():
        if tag.startswith("BLoB TRAIN ("):
            train_time_sec = sec
            break

    rows: List[Dict[str, object]] = []
    current_tag: Optional[str] = None
    for line in text.splitlines():
        line_s = line.strip()
        m_header = BLOB_HEADER_RE.match(line_s)
        if m_header:
            current_tag = m_header.group("tag")
            continue
        m_metrics = BLOB_METRICS_RE.match(line_s)
        if not current_tag or not m_metrics:
            continue
        method_name = m_metrics.group("method")
        if method_name == "MAP":
            continue
        eval_task, split = split_parenthetical_tag(current_tag)
        if method_name == "BLoB mean":
            method = "blob_mean_order"
            infer_tag = f"EVAL blob_mean on {eval_task}"
            infer_time_sec = stage_times.get(infer_tag)
        else:
            method = "blob_samp_order"
            infer_tag = (
                f"EVAL blob_sample(N={int(float(m_metrics.group('mc') or '0'))}) "
                f"on {eval_task}"
            )
            infer_time_sec = stage_times.get(infer_tag)
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(
            stage_peaks,
            infer_tag,
        )
        extras: Dict[str, object] = {}
        if m_metrics.group("mc") is not None:
            extras["mc_samples"] = float(m_metrics.group("mc"))
        rows.append(
            make_result_row(
                method=method,
                seed=seed,
                source_order="order",
                eval_task=eval_task,
                split=split,
                nll=float(m_metrics.group("nll")),
                acc_pct=float(m_metrics.group("acc")),
                ece_pct=float(m_metrics.group("ece")),
                brier=float(m_metrics.group("brier")),
                infer_time_sec=infer_time_sec,
                infer_peak_alloc_gb=infer_peak_alloc_gb,
                infer_peak_reserved_gb=infer_peak_reserved_gb,
                train_time_sec=train_time_sec,
                wall_time_sec=wall_time_sec,
                extras=extras,
            )
        )
    return rows, train_time_sec


def parse_ensemble_output(
    text: str,
    seed: int,
    source_order: str,
    wall_time_sec: float,
) -> List[Dict[str, object]]:
    del source_order
    stage_times = parse_stage_times(text)
    stage_peaks = parse_stage_peaks(text)
    rows: List[Dict[str, object]] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        m_header = ENSEMBLE_HEADER_RE.match(line.strip())
        if not m_header or idx + 1 >= len(lines):
            continue
        tag = m_header.group("tag")
        m_metrics = ENSEMBLE_METRICS_RE.search(lines[idx + 1].strip())
        if not m_metrics:
            continue
        eval_task, split = split_parenthetical_tag(tag)
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(
            stage_peaks, f"INFER Ensemble on {tag}"
        )
        extras: Dict[str, object] = {}
        if m_metrics.group("members") is not None:
            extras["ensemble_members"] = float(m_metrics.group("members"))
        rows.append(
            make_result_row(
                method="ensemble_order",
                seed=seed,
                source_order="order",
                eval_task=eval_task,
                split=split,
                nll=float(m_metrics.group("nll")),
                acc_pct=float(m_metrics.group("acc")),
                ece_pct=float(m_metrics.group("ece")),
                brier=float(m_metrics.group("brier")),
                infer_time_sec=stage_times.get(f"INFER Ensemble on {tag}"),
                infer_peak_alloc_gb=infer_peak_alloc_gb,
                infer_peak_reserved_gb=infer_peak_reserved_gb,
                train_time_sec=None,
                wall_time_sec=wall_time_sec,
                extras=extras,
            )
        )
    return rows


def parse_no_metrics(
    text: str,
    seed: int,
    source_order: str,
    wall_time_sec: float,
) -> List[Dict[str, object]]:
    del text, seed, source_order, wall_time_sec
    return []


def parse_train_wall_only(
    text: str,
    seed: int,
    source_order: str,
    wall_time_sec: float,
) -> Tuple[List[Dict[str, object]], float]:
    del text, seed, source_order
    return [], float(wall_time_sec)


def infer_stage_tag_from_row(command_name: str, row: Dict[str, object]) -> Optional[str]:
    method = str(row.get("method", ""))
    eval_task = str(row.get("eval_task", ""))
    split = str(row.get("split", ""))
    if method.startswith("map_"):
        return f"INFER MAP on {eval_task}({split})"
    if method == "mcdrop_order":
        return f"INFER MCDrop on {eval_task}({split})"
    if method == "laplace_order":
        return f"INFER Official-Source-Laplace on {eval_task}({split})"
    if method == "seq_constantq_order":
        seq_split = "iid" if eval_task == SOURCE_TASK else "ood"
        return f"INFER Seq-LoRA on {eval_task}_{seq_split}"
    if method == "blob_mean_order":
        return f"EVAL blob_mean on {eval_task}"
    if method == "blob_samp_order":
        mc_samples = int(float(row.get("mc_samples", 0)))
        return f"EVAL blob_sample(N={mc_samples}) on {eval_task}"
    if method == "ensemble_order":
        return f"INFER Ensemble on {eval_task}({split})"
    return None


def augment_result_row_with_infer_peaks(
    command_name: str,
    row: Dict[str, object],
    stage_peaks: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    if row.get("infer_peak_alloc_gb") is not None or row.get("infer_peak_reserved_gb") is not None:
        return row
    stage_tag = infer_stage_tag_from_row(command_name, row)
    if not stage_tag:
        return row
    peak = stage_peaks.get(stage_tag)
    if not peak:
        return row
    out = dict(row)
    out["infer_peak_alloc_gb"] = peak.get("alloc_gb")
    out["infer_peak_reserved_gb"] = peak.get("reserved_gb")
    return out


__all__ = [
    "augment_result_row_with_infer_peaks",
    "parse_blob_output",
    "parse_ensemble_output",
    "parse_laplace_output",
    "parse_map_eval_output",
    "parse_mcdrop_output",
    "parse_no_metrics",
    "parse_seq_output",
    "parse_stage_peaks",
    "parse_stage_times",
    "parse_train_wall_only",
]
