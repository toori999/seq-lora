from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SOURCE_TASK = "scienceqa_closedchoice_grade2_11"
DEFAULT_EVAL_TASKS = [
    "iid",
    "scienceqa_closedchoice_grade12",
    "obqa",
    "arc-c",
    "mmlu_science_high",
    "mmlu_science_college",
    "gpqa_main",
]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
POSTHOC_INTERNAL_SEED = 0
DEFAULT_ENSEMBLE_TOTAL_SEEDS = 20
DEFAULT_ENSEMBLE_GROUPS = 5
EXCLUDED_STATUS_PREFIXES: Tuple[str, ...] = ()

MAP_VARIANT_MODULES = {
    "order": "train_scienceqa_qwen35_9b_lora_map_leftpad",
    "reverse": "train_scienceqa_qwen35_9b_lora_map_leftpad_grade_desc",
    "random": "train_scienceqa_qwen35_9b_lora_map_leftpad_random",
}

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
TFB_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\]\[TFB\]$")
TFB_METRICS_RE = re.compile(
    r"NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+ECE=(?P<ece>[0-9.]+)%\s+"
    r"Brier=(?P<brier>[0-9.]+)\s+mc=(?P<mc>[0-9.]+)\s+beta=(?P<beta>[0-9.]+)"
)
TFB_SAVE_RE = re.compile(r"^\[Save\] TFB fit info -> (?P<path>.+)$")
TFB_LOG_SPECS: Tuple[Tuple[re.Pattern[str], str, bool], ...] = (
    (re.compile(r"^tfb_order_seed(?P<seed>\d+)$"), "tfb_order", False),
    (re.compile(r"^official_tfblora_fit_seed(?P<seed>\d+)$"), "official_tfblora_order", True),
    (re.compile(r"^official_tfblora_seed(?P<seed>\d+)_(?P<eval_task>.+)$"), "official_tfblora_order", False),
    (
        re.compile(r"^official_tfblora_bench_lora_seed(?P<seed>\d+)_(?P<eval_task>.+)$"),
        "official_tfblora_bench_lora_order",
        False,
    ),
    (
        re.compile(r"^official_tfblora_src_hparam_seed(?P<seed>\d+)_(?P<eval_task>.+)$"),
        "official_tfblora_src_hparam_order",
        False,
    ),
)


@dataclass(frozen=True)
class MapVariantConfig:
    order_key: str
    module_name: str
    output_dir: Path
    run_tag: str
    slice_dir: Path

    @property
    def map_root(self) -> Path:
        return self.output_dir / self.run_tag

    def run_dir(self, seed: int) -> Path:
        return self.map_root / f"seed_{seed}"

    def map_dir(self, seed: int) -> Path:
        return self.run_dir(seed) / "map_step_2000"

    def init_lora_path(self, seed: int) -> Path:
        return self.run_dir(seed) / "init_lora.pt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_list(spec: str) -> List[int]:
    vals: List[int] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        vals.append(int(raw))
    if not vals:
        raise ValueError("Expected at least one seed")
    return vals


def expand_eval_tasks(spec_tasks: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for raw in spec_tasks:
        task = str(raw).strip().lower()
        if not task:
            continue
        if task == "mmlu":
            expanded.extend(["mmlu_science_high", "mmlu_science_college"])
        else:
            expanded.append(task)
    out: List[str] = []
    seen = set()
    for task in expanded:
        if task not in seen:
            seen.add(task)
            out.append(task)
    return out


def load_map_variant_configs() -> Dict[str, MapVariantConfig]:
    out: Dict[str, MapVariantConfig] = {}
    for order_key, module_name in MAP_VARIANT_MODULES.items():
        mod = importlib.import_module(module_name)
        runner_mod = getattr(mod, "base", mod)
        out[order_key] = MapVariantConfig(
            order_key=order_key,
            module_name=module_name,
            output_dir=Path(str(runner_mod.OUTPUT_DIR)).resolve(),
            run_tag=str(runner_mod.RUN_TAG),
            slice_dir=Path(str(runner_mod.SLICE_OUT_DIR)).resolve(),
        )
    return out


def stream_subprocess(cmd: Sequence[str], log_path: Path, cwd: Path) -> Tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"$ {' '.join(cmd)}\n\n")
        log_f.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()
    dt = time.perf_counter() - t0
    return int(proc.returncode), float(dt)


def parse_stage_times(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        m = TIME_RE.match(line.strip())
        if m:
            out[m.group("tag")] = float(m.group("sec"))
    return out


def parse_stage_peaks(text: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        m = PEAK_RE.match(line.strip())
        if m:
            out[m.group("tag")] = {
                "alloc_gb": float(m.group("alloc")),
                "reserved_gb": float(m.group("reserved")),
            }
    return out


def split_parenthetical_tag(tag: str) -> Tuple[str, str]:
    m = TAG_WITH_SPLIT_RE.match(tag.strip())
    if not m:
        raise ValueError(f"Could not parse tag with split: {tag}")
    return m.group("task"), m.group("split")


def split_seq_tag(tag: str) -> Tuple[str, str]:
    m = SEQ_TAG_RE.match(tag.strip())
    if not m:
        raise ValueError(f"Could not parse seq tag: {tag}")
    return m.group("task"), m.group("split")


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
    extras: Optional[Dict[str, float]] = None,
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
        "infer_time_sec": (None if infer_time_sec is None else float(infer_time_sec)),
        "infer_peak_alloc_gb": (None if infer_peak_alloc_gb is None else float(infer_peak_alloc_gb)),
        "infer_peak_reserved_gb": (None if infer_peak_reserved_gb is None else float(infer_peak_reserved_gb)),
        "train_time_sec": (None if train_time_sec is None else float(train_time_sec)),
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
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(stage_peaks, f"INFER MAP on {tag}")
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


def parse_map_train_output(
    text: str,
    seed: int,
    source_order: str,
    wall_time_sec: float,
) -> Tuple[List[Dict[str, object]], Optional[float]]:
    del seed, source_order
    stage_times = parse_stage_times(text)
    train_time_sec = None
    for tag, sec in stage_times.items():
        if tag.startswith("TRAIN MAP on "):
            train_time_sec = sec
            break
    if train_time_sec is None:
        train_time_sec = float(wall_time_sec)
    return [], train_time_sec


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
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(stage_peaks, f"INFER MCDrop on {tag}")
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
    for m in SEQ_BLOCK_RE.finditer(text):
        tag = m.group("tag")
        eval_task, split = split_seq_tag(tag)
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(stage_peaks, f"INFER Seq-LoRA on {tag}")
        rows.append(
            make_result_row(
                method="seq_constantq_order",
                seed=seed,
                source_order="order",
                eval_task=eval_task,
                split=("test" if split == "iid" else split),
                nll=float(m.group("nll")),
                acc_pct=float(m.group("acc")),
                ece_pct=float(m.group("ece")),
                brier=float(m.group("brier")),
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
                infer_time_sec=stage_times.get(f"INFER Official-Source-Laplace on {current_tag}"),
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
            infer_tag = f"EVAL blob_sample(N={int(float(m_metrics.group('mc') or '0'))}) on {eval_task}"
            infer_time_sec = stage_times.get(infer_tag)
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(stage_peaks, infer_tag)
        extras: Dict[str, float] = {}
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
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(stage_peaks, f"INFER Ensemble on {tag}")
        extras: Dict[str, float] = {}
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


def parse_tfb_output(
    text: str,
    seed: int,
    source_order: str,
    wall_time_sec: float,
    method: str = "tfb_order",
) -> Tuple[List[Dict[str, object]], Optional[float]]:
    del source_order
    stage_times = parse_stage_times(text)
    stage_peaks = parse_stage_peaks(text)
    train_time_sec = None
    for tag, sec in stage_times.items():
        if tag.startswith("FIT TFB on "):
            train_time_sec = sec
            break

    rows: List[Dict[str, object]] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        m_header = TFB_HEADER_RE.match(line.strip())
        if not m_header or idx + 1 >= len(lines):
            continue
        tag = m_header.group("tag")
        m_metrics = TFB_METRICS_RE.search(lines[idx + 1].strip())
        if not m_metrics:
            continue
        eval_task, split = split_parenthetical_tag(tag)
        infer_tag = f"INFER TFB on {tag}"
        infer_peak_alloc_gb, infer_peak_reserved_gb = _get_stage_peak(stage_peaks, infer_tag)
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
                infer_time_sec=stage_times.get(infer_tag),
                infer_peak_alloc_gb=infer_peak_alloc_gb,
                infer_peak_reserved_gb=infer_peak_reserved_gb,
                train_time_sec=train_time_sec,
                wall_time_sec=wall_time_sec,
                extras={
                    "mc_samples": float(m_metrics.group("mc")),
                    "tfb_beta": float(m_metrics.group("beta")),
                },
            )
        )
    return rows, train_time_sec


def _classify_tfb_log(log_stem: str) -> Optional[Tuple[int, str, bool]]:
    for pattern, method, fit_only in TFB_LOG_SPECS:
        m = pattern.match(log_stem)
        if m:
            return int(m.group("seed")), method, fit_only
    return None


def build_tfb_payload_from_log(log_path: Path) -> Optional[Dict[str, object]]:
    classified = _classify_tfb_log(log_path.stem)
    if not classified:
        return None

    seed, method, fit_only = classified
    text = log_path.read_text(encoding="utf-8", errors="replace")
    stage_times = parse_stage_times(text)
    stage_peaks = parse_stage_peaks(text)
    wall_time_sec = float(sum(stage_times.values())) if stage_times else 0.0
    results, train_time_sec = parse_tfb_output(
        text,
        seed=seed,
        source_order="order",
        wall_time_sec=wall_time_sec,
        method=method,
    )
    if not results and not fit_only:
        return None

    fit_tag = next((tag for tag in stage_times if tag.startswith("FIT TFB on ")), None)
    train_peak_alloc_gb = None
    train_peak_reserved_gb = None
    if fit_tag:
        peak = stage_peaks.get(fit_tag)
        if peak:
            train_peak_alloc_gb = peak.get("alloc_gb")
            train_peak_reserved_gb = peak.get("reserved_gb")

    artifacts: Dict[str, str] = {}
    for line in text.splitlines():
        m_save = TFB_SAVE_RE.match(line.strip())
        if m_save:
            artifacts["tfb_fit_json"] = m_save.group("path")
            break

    return {
        "name": log_path.stem,
        "seed": seed,
        "source_order": "order",
        "command": [],
        "started_at": None,
        "finished_at": None,
        "returncode": 0,
        "wall_time_sec": wall_time_sec,
        "log_path": str(log_path),
        "artifacts": artifacts,
        "stage_times_sec": stage_times,
        "stage_peaks_gb": stage_peaks,
        "train_time_sec": train_time_sec,
        "train_peak_alloc_gb": train_peak_alloc_gb,
        "train_peak_reserved_gb": train_peak_reserved_gb,
        "results": ([] if fit_only else results),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_or_none(values: List[float]) -> Optional[float]:
    return (statistics.mean(values) if values else None)


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
                "nll_sd": (statistics.stdev(float(r["nll"]) for r in group_rows) if len(group_rows) > 1 else 0.0),
                "acc_pct_mean": statistics.mean(float(r["acc_pct"]) for r in group_rows),
                "acc_pct_sd": (statistics.stdev(float(r["acc_pct"]) for r in group_rows) if len(group_rows) > 1 else 0.0),
                "ece_pct_mean": statistics.mean(float(r["ece_pct"]) for r in group_rows),
                "ece_pct_sd": (statistics.stdev(float(r["ece_pct"]) for r in group_rows) if len(group_rows) > 1 else 0.0),
                "brier_mean": statistics.mean(float(r["brier"]) for r in group_rows),
                "brier_sd": (statistics.stdev(float(r["brier"]) for r in group_rows) if len(group_rows) > 1 else 0.0),
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

    # Extra macro summary over the two current MMLU science splits.
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
                "infer_time_sec": statistics.mean(float(r["infer_time_sec"]) for r in subset if r["infer_time_sec"] is not None),
                "infer_peak_alloc_gb": (
                    statistics.mean(float(r["infer_peak_alloc_gb"]) for r in subset if r.get("infer_peak_alloc_gb") is not None)
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
                    "nll_sd": (statistics.stdev(float(r["nll"]) for r in subset) if len(subset) > 1 else 0.0),
                    "acc_pct_mean": statistics.mean(float(r["acc_pct"]) for r in subset),
                    "acc_pct_sd": (statistics.stdev(float(r["acc_pct"]) for r in subset) if len(subset) > 1 else 0.0),
                    "ece_pct_mean": statistics.mean(float(r["ece_pct"]) for r in subset),
                    "ece_pct_sd": (statistics.stdev(float(r["ece_pct"]) for r in subset) if len(subset) > 1 else 0.0),
                    "brier_mean": statistics.mean(float(r["brier"]) for r in subset),
                    "brier_sd": (statistics.stdev(float(r["brier"]) for r in subset) if len(subset) > 1 else 0.0),
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
        "map_order": "train_map_order",
        "map_reverse": "train_map_reverse",
        "map_random": "train_map_random",
        "seq": "seq_constantq_order",
        "laplace": "laplace_order",
        "blob": "blob_order",
        "tfb": "tfb_order",
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
                "train_peak_alloc_gb_max": (max(peak_allocs) if peak_allocs else None),
                "train_peak_reserved_gb_max": (max(peak_reserved) if peak_reserved else None),
            }
        )
    return out


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _infer_stage_tag_from_row(command_name: str, row: Dict[str, object]) -> Optional[str]:
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
    if method == "tfb_order":
        return f"INFER TFB on {eval_task}({split})"
    if method in {"official_tfblora_order", "official_tfblora_bench_lora_order", "official_tfblora_src_hparam_order"}:
        return f"INFER TFB on {eval_task}({split})"
    return None


def _augment_result_row_with_infer_peaks(
    command_name: str,
    row: Dict[str, object],
    stage_peaks: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    if row.get("infer_peak_alloc_gb") is not None or row.get("infer_peak_reserved_gb") is not None:
        return row
    stage_tag = _infer_stage_tag_from_row(command_name, row)
    if not stage_tag:
        return row
    peak = stage_peaks.get(stage_tag)
    if not peak:
        return row
    out = dict(row)
    out["infer_peak_alloc_gb"] = peak.get("alloc_gb")
    out["infer_peak_reserved_gb"] = peak.get("reserved_gb")
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
                metrics_rows.append(_augment_result_row_with_infer_peaks(command_name, row, stage_peaks))

    existing_command_names = {str(row.get("name", "")) for row in command_rows}
    logs_dir = status_dir.parent / "logs"
    for log_path in sorted(logs_dir.glob("*.log")):
        if log_path.stem in existing_command_names:
            continue
        payload = build_tfb_payload_from_log(log_path)
        if not payload:
            continue
        command_rows.append(payload)
        existing_command_names.add(str(payload.get("name", "")))
        stage_peaks = payload.get("stage_peaks_gb", {})
        if not isinstance(stage_peaks, dict):
            stage_peaks = {}
        for row in payload.get("results", []):
            if isinstance(row, dict):
                metrics_rows.append(_augment_result_row_with_infer_peaks(str(payload.get("name", "")), row, stage_peaks))

    map_train_by_key: Dict[Tuple[str, int], Dict[str, object]] = {}
    official_tfb_fit_by_seed: Dict[int, Dict[str, object]] = {}
    for payload in command_rows:
        name = str(payload.get("name", ""))
        m = re.match(r"^train_map_(order|reverse|random)_seed(?P<seed>\d+)$", name)
        if m:
            order_key = m.group(1)
            seed = int(m.group("seed"))
            map_train_by_key[(order_key, seed)] = payload
        m_tfb_fit = re.match(r"^official_tfblora_fit_seed(?P<seed>\d+)$", name)
        if m_tfb_fit:
            official_tfb_fit_by_seed[int(m_tfb_fit.group("seed"))] = payload

    for row in metrics_rows:
        method = str(row.get("method", ""))
        seed = int(row.get("seed"))
        if method in {"map_order", "map_reverse", "map_random"}:
            order_key = method.replace("map_", "", 1)
            payload = map_train_by_key.get((order_key, seed))
            if payload and row.get("train_time_sec") in (None, ""):
                row["train_time_sec"] = payload.get("train_time_sec")
        if method == "official_tfblora_order":
            payload = official_tfb_fit_by_seed.get(seed)
            if payload and row.get("train_time_sec") in (None, ""):
                row["train_time_sec"] = payload.get("train_time_sec")
    return metrics_rows, command_rows


def refresh_exports(result_root: Path) -> None:
    status_dir = result_root / "status"
    metrics_rows, command_rows = collect_status_rows(status_dir)
    write_json(result_root / "all_commands.json", command_rows)
    write_json(result_root / "all_metrics.json", metrics_rows)
    write_csv(result_root / "command_times.csv", summarize_command_rows(command_rows))
    write_csv(result_root / "training_resource_summary.csv", summarize_training_resources(command_rows))
    write_csv(result_root / "all_metrics.csv", metrics_rows)
    write_csv(result_root / "summary_mean_sd.csv", summarize_rows(metrics_rows))


def build_map_train_command(
    module_name: str,
    seed: int,
    map_micro_bsz: int,
    map_grad_accum: int,
    map_eval_bsz: int,
) -> List[str]:
    code = (
        f"import {module_name} as m; "
        f"runner=getattr(m,'base',m); "
        f"runner.MICRO_BSZ={int(map_micro_bsz)}; "
        f"runner.GRAD_ACCUM={int(map_grad_accum)}; "
        f"runner.EVAL_BSZ={int(map_eval_bsz)}; "
        f"runner.SEEDS=[{seed}]; "
        f"runner.main()"
    )
    return [sys.executable, "-c", code]


def maybe_skip(status_path: Path, resume: bool) -> Optional[Dict[str, object]]:
    if resume and status_path.exists():
        payload = read_json(status_path)
        if isinstance(payload, dict) and payload.get("returncode") == 0:
            print(f"[Skip] found completed status: {status_path.name}")
            return payload
    return None


def run_and_record(
    *,
    name: str,
    cmd: Sequence[str],
    parser_fn,
    result_root: Path,
    cwd: Path,
    seed: int,
    source_order: str,
    artifacts: Optional[Dict[str, str]] = None,
    resume: bool = True,
) -> Dict[str, object]:
    status_path = result_root / "status" / f"{name}.json"
    skipped = maybe_skip(status_path, resume=resume)
    if skipped is not None:
        return skipped

    log_path = result_root / "logs" / f"{name}.log"
    print("\n" + "=" * 100)
    print(f"[Run] {name}")
    print("=" * 100)
    print("$ " + " ".join(cmd))

    started_at = utc_now()
    returncode, wall_time_sec = stream_subprocess(cmd, log_path=log_path, cwd=cwd)
    finished_at = utc_now()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    stage_times = parse_stage_times(text)
    stage_peaks = parse_stage_peaks(text)

    train_peak_alloc_gb = None
    train_peak_reserved_gb = None
    train_peak_tags: List[str] = []
    if name.startswith("train_map_"):
        train_peak_tags = [tag for tag in stage_peaks if tag.startswith("TRAIN MAP on ")]
    elif name.startswith("seq_constantq_order"):
        train_peak_tags = [tag for tag in stage_peaks if tag.startswith("TRAIN-STAGE Seq-LoRA posterior build on ")]
    elif name.startswith("laplace_order"):
        train_peak_tags = [tag for tag in stage_peaks if tag.startswith("OFFICIAL SOURCE Laplace fit on ")]
    elif name.startswith("blob_order"):
        train_peak_tags = [tag for tag in stage_peaks if tag.startswith("BLoB TRAIN (")]
    if train_peak_tags:
        train_peak_alloc_gb = max(stage_peaks[tag]["alloc_gb"] for tag in train_peak_tags)
        train_peak_reserved_gb = max(stage_peaks[tag]["reserved_gb"] for tag in train_peak_tags)

    if returncode != 0:
        payload = {
            "name": name,
            "seed": seed,
            "source_order": source_order,
            "command": list(cmd),
            "started_at": started_at,
            "finished_at": finished_at,
            "returncode": returncode,
            "wall_time_sec": wall_time_sec,
            "log_path": str(log_path),
            "artifacts": artifacts or {},
            "stage_times_sec": stage_times,
            "stage_peaks_gb": stage_peaks,
            "train_peak_alloc_gb": train_peak_alloc_gb,
            "train_peak_reserved_gb": train_peak_reserved_gb,
            "results": [],
        }
        write_json(status_path, payload)
        refresh_exports(result_root)
        raise RuntimeError(f"Command failed ({returncode}): {' '.join(cmd)}")

    parsed = parser_fn(text, seed, source_order, wall_time_sec)
    train_time_sec = None
    results = parsed
    if isinstance(parsed, tuple):
        results, train_time_sec = parsed

    payload = {
        "name": name,
        "seed": seed,
        "source_order": source_order,
        "command": list(cmd),
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": returncode,
        "wall_time_sec": wall_time_sec,
        "log_path": str(log_path),
        "artifacts": artifacts or {},
        "stage_times_sec": stage_times,
        "stage_peaks_gb": stage_peaks,
        "train_time_sec": train_time_sec,
        "train_peak_alloc_gb": train_peak_alloc_gb,
        "train_peak_reserved_gb": train_peak_reserved_gb,
        "results": results,
    }
    write_json(status_path, payload)
    refresh_exports(result_root)
    return payload


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


def build_blob_command(
    *,
    map_dir: Path,
    init_lora_path: Path,
    save_blob_dir: Path,
    eval_tasks: Sequence[str],
    seed: int,
    blob_eval_n: int,
) -> List[str]:
    return [
        sys.executable,
        "blob_eval_iid_official.py",
        "--task",
        SOURCE_TASK,
        "--base_model",
        "Qwen/Qwen3-8B-Base",
        "--map_adapter_dir",
        str(map_dir),
        "--shared_init_lora_path",
        str(init_lora_path),
        "--save_blob_dir",
        str(save_blob_dir),
        "--do_train",
        "--do_eval",
        "--seed",
        str(seed),
        "--blob_eval_n",
        str(blob_eval_n),
        "--eval_tasks",
        ",".join(eval_tasks),
    ]


def build_map_eval_command(map_dir: Path, eval_tasks: Sequence[str], seed: int) -> List[str]:
    return [
        sys.executable,
        "map_eval.py",
        "--task",
        SOURCE_TASK,
        "--map_adapter_dir",
        str(map_dir),
        "--eval_tasks",
        ",".join(eval_tasks),
        "--seed",
        str(seed),
    ]


def build_seq_command(
    map_dir: Path,
    slice_dir: Path,
    eval_tasks: Sequence[str],
    s_q: float,
    q_mode: str,
    seq_mc_eval_chunk: int = 0,
) -> List[str]:
    cmd = [
        sys.executable,
        "seq_eval_iid_constantq.py",
        "--task",
        SOURCE_TASK,
        "--slices_dir",
        str(slice_dir),
        "--map_dir",
        str(map_dir),
        "--eval_tasks",
        ",".join(eval_tasks),
        "--s_q",
        str(s_q),
        "--q_mode",
        str(q_mode),
        "--forecast_horizon",
        "0",
    ]
    if int(seq_mc_eval_chunk) > 0:
        cmd.extend(["--mc_eval_chunk", str(int(seq_mc_eval_chunk))])
    return cmd


def build_laplace_command(
    *,
    map_dir: Path,
    output_dir: Path,
    eval_tasks: Sequence[str],
    seed: int,
    fit_bsz: int,
    laplace_bsz: int,
    prior_optim_step: int,
    laplace_mc_samples: int,
    laplace_mc_chunk: int,
) -> List[str]:
    return [
        sys.executable,
        "laplace_lora_official_source_eval.py",
        "--task_name",
        SOURCE_TASK,
        "--map_adapter_dir",
        str(map_dir),
        "--output_dir",
        str(output_dir),
        "--eval_tasks",
        ",".join(eval_tasks),
        "--laplace_sub",
        "all",
        "--testing_set",
        "val",
        "--seed",
        str(POSTHOC_INTERNAL_SEED),
        "--fit_bsz",
        str(fit_bsz),
        "--laplace_bsz",
        str(laplace_bsz),
        "--prior_optim_step",
        str(prior_optim_step),
        "--laplace_mc_samples",
        str(laplace_mc_samples),
        "--laplace_mc_chunk",
        str(laplace_mc_chunk),
    ]


def build_mcdrop_command(map_dir: Path, eval_tasks: Sequence[str], seed: int, mc_samples: int, temp: float) -> List[str]:
    return [
        sys.executable,
        "mcdrop_eval.py",
        "--task",
        SOURCE_TASK,
        "--map_adapter_dir",
        str(map_dir),
        "--eval_tasks",
        ",".join(eval_tasks),
        "--seed",
        str(POSTHOC_INTERNAL_SEED),
        "--mc_samples",
        str(mc_samples),
        "--temp",
        str(temp),
    ]


def build_ensemble_command(
    *,
    map_dirs: Sequence[Path],
    eval_tasks: Sequence[str],
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "map_ensemble_eval.py",
        "--task",
        SOURCE_TASK,
        "--eval_tasks",
        ",".join(eval_tasks),
        "--seed",
        str(POSTHOC_INTERNAL_SEED),
    ]
    for map_dir in map_dirs:
        cmd.extend(["--map_adapter_dir", str(map_dir)])
    return cmd


def build_consecutive_ensemble_groups(total_seeds: int, num_groups: int) -> List[List[int]]:
    if total_seeds <= 0:
        raise ValueError("ensemble total_seeds must be positive")
    if num_groups <= 0:
        raise ValueError("ensemble num_groups must be positive")
    if total_seeds % num_groups != 0:
        raise ValueError(f"ensemble total_seeds={total_seeds} must be divisible by num_groups={num_groups}")
    seeds = list(range(total_seeds))
    group_size = total_seeds // num_groups
    return [seeds[i * group_size : (i + 1) * group_size] for i in range(num_groups)]


def load_status_payload(result_root: Path, name: str) -> Dict[str, object]:
    path = result_root / "status" / f"{name}.json"
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Malformed status payload: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ScienceQA 5-seed benchmark suite and save structured local results.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--result_root", type=str, default="./benchmark_suite_scienceqa")
    parser.add_argument("--eval_tasks", type=str, default=",".join(DEFAULT_EVAL_TASKS))
    parser.add_argument("--map_micro_bsz", type=int, default=4)
    parser.add_argument("--map_grad_accum", type=int, default=2)
    parser.add_argument("--map_eval_bsz", type=int, default=32)
    parser.add_argument("--s_q", type=float, default=1.0)
    parser.add_argument("--q_mode", type=str, choices=["module_constant", "constant"], default="module_constant")
    parser.add_argument("--constant_q_var", dest="s_q", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--seq_mc_eval_chunk", type=int, default=0)
    parser.add_argument("--blob_eval_n", type=int, default=10)
    parser.add_argument("--mcdrop_mc_samples", type=int, default=32)
    parser.add_argument("--mcdrop_temp", type=float, default=1.0)
    parser.add_argument("--laplace_fit_bsz", type=int, default=2)
    parser.add_argument("--laplace_bsz", type=int, default=4)
    parser.add_argument("--laplace_prior_optim_step", type=int, default=100)
    parser.add_argument("--laplace_mc_samples", type=int, default=32)
    parser.add_argument("--laplace_mc_chunk", type=int, default=8)
    parser.add_argument("--ensemble_total_seeds", type=int, default=DEFAULT_ENSEMBLE_TOTAL_SEEDS)
    parser.add_argument("--ensemble_num_groups", type=int, default=DEFAULT_ENSEMBLE_GROUPS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd()
    result_root = Path(args.result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds)
    eval_tasks = expand_eval_tasks([task.strip() for task in args.eval_tasks.split(",") if task.strip()])
    ensemble_groups = build_consecutive_ensemble_groups(int(args.ensemble_total_seeds), int(args.ensemble_num_groups))
    ensemble_all_seeds = sorted({seed for group in ensemble_groups for seed in group})
    configs = load_map_variant_configs()

    write_json(
        result_root / "suite_config.json",
        {
            "created_at": utc_now(),
            "cwd": str(cwd),
            "source_task": SOURCE_TASK,
            "seeds": seeds,
            "eval_tasks": eval_tasks,
            "map_micro_bsz": int(args.map_micro_bsz),
            "map_grad_accum": int(args.map_grad_accum),
            "map_eval_bsz": int(args.map_eval_bsz),
            "s_q": float(args.s_q),
            "q_mode": str(args.q_mode),
            "seq_mc_eval_chunk": int(args.seq_mc_eval_chunk),
            "blob_eval_n": int(args.blob_eval_n),
            "mcdrop_mc_samples": int(args.mcdrop_mc_samples),
            "mcdrop_temp": float(args.mcdrop_temp),
            "laplace_fit_bsz": int(args.laplace_fit_bsz),
            "laplace_bsz": int(args.laplace_bsz),
            "laplace_prior_optim_step": int(args.laplace_prior_optim_step),
            "laplace_mc_samples": int(args.laplace_mc_samples),
            "laplace_mc_chunk": int(args.laplace_mc_chunk),
            "ensemble_total_seeds": int(args.ensemble_total_seeds),
            "ensemble_num_groups": int(args.ensemble_num_groups),
            "ensemble_groups": ensemble_groups,
            "resume": bool(args.resume),
            "map_variants": {
                key: {
                    "module_name": cfg.module_name,
                    "output_dir": str(cfg.output_dir),
                    "run_tag": cfg.run_tag,
                    "slice_dir": str(cfg.slice_dir),
                }
                for key, cfg in configs.items()
            },
        },
    )

    # 1) Train + eval the three MAP variants over all seeds.
    for order_key in ("order", "reverse", "random"):
        cfg = configs[order_key]
        for seed in seeds:
            map_dir = cfg.map_dir(seed)
            run_and_record(
                name=f"train_map_{order_key}_seed{seed}",
                cmd=build_map_train_command(
                    cfg.module_name,
                    seed,
                    int(args.map_micro_bsz),
                    int(args.map_grad_accum),
                    int(args.map_eval_bsz),
                ),
                parser_fn=parse_map_train_output,
                result_root=result_root,
                cwd=cwd,
                seed=seed,
                source_order=order_key,
                artifacts={
                    "map_dir": str(map_dir),
                    "run_dir": str(cfg.run_dir(seed)),
                    "slice_dir": str(cfg.slice_dir),
                },
                resume=args.resume and map_dir.exists(),
            )
            run_and_record(
                name=f"eval_map_{order_key}_seed{seed}",
                cmd=build_map_eval_command(map_dir, eval_tasks, seed),
                parser_fn=parse_map_eval_output,
                result_root=result_root,
                cwd=cwd,
                seed=seed,
                source_order=order_key,
                artifacts={"map_dir": str(map_dir)},
                resume=args.resume,
            )

    # 2) Run the methods that depend on the sequential MAP seeds only.
    order_cfg = configs["order"]
    blob_root = (cwd / "blob_qwen35_8b_scienceqa_leftpad").resolve()
    for seed in seeds:
        map_dir = order_cfg.map_dir(seed)
        init_lora_path = order_cfg.init_lora_path(seed)

        blob_dir = blob_root / f"seed_{seed}" / "blob"
        run_and_record(
            name=f"blob_order_seed{seed}",
            cmd=build_blob_command(
                map_dir=map_dir,
                init_lora_path=init_lora_path,
                save_blob_dir=blob_dir,
                eval_tasks=eval_tasks,
                seed=seed,
                blob_eval_n=int(args.blob_eval_n),
            ),
            parser_fn=parse_blob_output,
            result_root=result_root,
            cwd=cwd,
            seed=seed,
            source_order="order",
            artifacts={
                "map_dir": str(map_dir),
                "blob_dir": str(blob_dir),
                "init_lora_path": str(init_lora_path),
            },
            resume=args.resume,
        )

        run_and_record(
            name=f"seq_constantq_order_seed{seed}",
            cmd=build_seq_command(
                map_dir=map_dir,
                slice_dir=order_cfg.slice_dir,
                eval_tasks=eval_tasks,
                s_q=float(args.s_q),
                q_mode=str(args.q_mode),
                seq_mc_eval_chunk=int(args.seq_mc_eval_chunk),
            ),
            parser_fn=parse_seq_output,
            result_root=result_root,
            cwd=cwd,
            seed=seed,
            source_order="order",
            artifacts={
                "map_dir": str(map_dir),
                "slice_dir": str(order_cfg.slice_dir),
            },
            resume=args.resume,
        )

        laplace_dir = (cwd / "outputs_laplace_official_source_qv_lmhead_suite" / f"seed_{seed}").resolve()
        run_and_record(
            name=f"laplace_order_seed{seed}",
            cmd=build_laplace_command(
                map_dir=map_dir,
                output_dir=laplace_dir,
                eval_tasks=eval_tasks,
                seed=seed,
                fit_bsz=int(args.laplace_fit_bsz),
                laplace_bsz=int(args.laplace_bsz),
                prior_optim_step=int(args.laplace_prior_optim_step),
                laplace_mc_samples=int(args.laplace_mc_samples),
                laplace_mc_chunk=int(args.laplace_mc_chunk),
            ),
            parser_fn=parse_laplace_output,
            result_root=result_root,
            cwd=cwd,
            seed=seed,
            source_order="order",
            artifacts={
                "map_dir": str(map_dir),
                "laplace_dir": str(laplace_dir),
            },
            resume=args.resume,
        )

        run_and_record(
            name=f"mcdrop_order_seed{seed}",
            cmd=build_mcdrop_command(
                map_dir=map_dir,
                eval_tasks=eval_tasks,
                seed=seed,
                mc_samples=int(args.mcdrop_mc_samples),
                temp=float(args.mcdrop_temp),
            ),
            parser_fn=parse_mcdrop_output,
            result_root=result_root,
            cwd=cwd,
            seed=seed,
            source_order="order",
            artifacts={"map_dir": str(map_dir)},
            resume=args.resume,
        )

    # 3) After the standard methods finish, expand order-MAP to 20 seeds and build 5 ensembles.
    additional_ensemble_seeds = [seed for seed in ensemble_all_seeds if seed not in seeds]
    for seed in additional_ensemble_seeds:
        map_dir = order_cfg.map_dir(seed)
        run_and_record(
            name=f"train_map_order_seed{seed}",
            cmd=build_map_train_command(
                order_cfg.module_name,
                seed,
                int(args.map_micro_bsz),
                int(args.map_grad_accum),
                int(args.map_eval_bsz),
            ),
            parser_fn=parse_train_wall_only,
            result_root=result_root,
            cwd=cwd,
            seed=seed,
            source_order="order",
            artifacts={
                "map_dir": str(map_dir),
                "run_dir": str(order_cfg.run_dir(seed)),
                "slice_dir": str(order_cfg.slice_dir),
                "ensemble_member_only": "true",
            },
            resume=args.resume and map_dir.exists(),
        )

    for group_idx, member_seeds in enumerate(ensemble_groups):
        member_map_dirs = [order_cfg.map_dir(seed) for seed in member_seeds]
        payload = run_and_record(
            name=f"ensemble_order_group{group_idx}",
            cmd=build_ensemble_command(
                map_dirs=member_map_dirs,
                eval_tasks=eval_tasks,
            ),
            parser_fn=parse_ensemble_output,
            result_root=result_root,
            cwd=cwd,
            seed=group_idx,
            source_order="order",
            artifacts={
                "member_seeds": ",".join(str(seed) for seed in member_seeds),
                "member_map_dirs": [str(path) for path in member_map_dirs],
            },
            resume=args.resume,
        )
        member_train_payloads = [load_status_payload(result_root, f"train_map_order_seed{seed}") for seed in member_seeds]
        ensemble_train_time_sec = sum(float(p.get("train_time_sec") or 0.0) for p in member_train_payloads)
        peak_alloc_vals = [float(p["train_peak_alloc_gb"]) for p in member_train_payloads if p.get("train_peak_alloc_gb") is not None]
        peak_reserved_vals = [
            float(p["train_peak_reserved_gb"]) for p in member_train_payloads if p.get("train_peak_reserved_gb") is not None
        ]
        ensemble_train_peak_alloc_gb = (max(peak_alloc_vals) if peak_alloc_vals else None)
        ensemble_train_peak_reserved_gb = (max(peak_reserved_vals) if peak_reserved_vals else None)
        payload["train_time_sec"] = ensemble_train_time_sec
        payload["train_peak_alloc_gb"] = ensemble_train_peak_alloc_gb
        payload["train_peak_reserved_gb"] = ensemble_train_peak_reserved_gb
        for row in payload.get("results", []):
            if isinstance(row, dict):
                row["train_time_sec"] = ensemble_train_time_sec
                row["ensemble_members"] = float(len(member_seeds))
                row["ensemble_member_seeds"] = ",".join(str(seed) for seed in member_seeds)
        write_json(result_root / "status" / f"ensemble_order_group{group_idx}.json", payload)
        refresh_exports(result_root)

    refresh_exports(result_root)
    print(f"\n[Done] Full benchmark suite finished. Results saved to: {result_root}")


if __name__ == "__main__":
    main()
