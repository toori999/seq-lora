from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import os
import subprocess
import sys
import time

from .benchmark_config import POSTHOC_INTERNAL_SEED, SOURCE_TASK, utc_now
from .benchmark_exports import read_json, refresh_exports, write_json
from .benchmark_parsing import parse_stage_peaks, parse_stage_times


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


def build_map_train_command(
    order_key: str,
    seed: int,
    map_micro_bsz: int,
    map_grad_accum: int,
    map_eval_bsz: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "refactor.bayesian_peft_layout.run.train_map",
        "--backend",
        "legacy",
        "--variant",
        str(order_key),
        "--seed",
        str(seed),
        "--micro-bsz",
        str(int(map_micro_bsz)),
        "--grad-accum",
        str(int(map_grad_accum)),
        "--eval-bsz",
        str(int(map_eval_bsz)),
    ]


def build_map_eval_command(map_dir: Path, eval_tasks: Sequence[str], seed: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "refactor.bayesian_peft_layout.run.main",
        "map",
        "--task",
        SOURCE_TASK,
        "--adapter-dir",
        str(map_dir),
        "--eval-tasks",
        ",".join(eval_tasks),
        "--seed",
        str(seed),
    ]


def build_seq_command(
    map_dir: Path,
    slice_dir: Path,
    eval_tasks: Sequence[str],
    constant_q_var: float,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "refactor.bayesian_peft_layout.run.main",
        "seq-constantq",
        "--task",
        SOURCE_TASK,
        "--slice-dir",
        str(slice_dir),
        "--map-dir",
        str(map_dir),
        "--eval-tasks",
        ",".join(eval_tasks),
        "--constant-q-var",
        str(constant_q_var),
        "--forecast-horizon",
        "0",
    ]


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
) -> list[str]:
    del seed
    return [
        sys.executable,
        "-m",
        "refactor.bayesian_peft_layout.run.main",
        "laplace",
        "--task",
        SOURCE_TASK,
        "--adapter-dir",
        str(map_dir),
        "--output-dir",
        str(output_dir),
        "--eval-tasks",
        ",".join(eval_tasks),
        "--testing-set",
        "val",
        "--seed",
        str(POSTHOC_INTERNAL_SEED),
        "--fit-bsz",
        str(fit_bsz),
        "--laplace-bsz",
        str(laplace_bsz),
        "--prior-optim-step",
        str(prior_optim_step),
        "--laplace-mc-samples",
        str(laplace_mc_samples),
        "--laplace-mc-chunk",
        str(laplace_mc_chunk),
    ]


def build_blob_command(
    *,
    map_dir: Path,
    init_lora_path: Path,
    save_blob_dir: Path,
    eval_tasks: Sequence[str],
    seed: int,
    blob_eval_n: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "refactor.bayesian_peft_layout.run.main",
        "blob",
        "--task",
        SOURCE_TASK,
        "--base-model",
        "Qwen/Qwen3-8B-Base",
        "--map-dir",
        str(map_dir),
        "--shared-init-lora-path",
        str(init_lora_path),
        "--save-blob-dir",
        str(save_blob_dir),
        "--do_train",
        "--do_eval",
        "--seed",
        str(seed),
        "--blob-eval-n",
        str(blob_eval_n),
        "--eval-tasks",
        ",".join(eval_tasks),
    ]


def build_mcdrop_command(
    map_dir: Path,
    eval_tasks: Sequence[str],
    seed: int,
    mc_samples: int,
    temp: float,
) -> list[str]:
    del seed
    return [
        sys.executable,
        "-m",
        "refactor.bayesian_peft_layout.run.main",
        "mcdrop",
        "--task",
        SOURCE_TASK,
        "--adapter-dir",
        str(map_dir),
        "--eval-tasks",
        ",".join(eval_tasks),
        "--seed",
        str(POSTHOC_INTERNAL_SEED),
        "--mc-samples",
        str(mc_samples),
        "--temp",
        str(temp),
    ]


def build_ensemble_command(
    *,
    map_dirs: Sequence[Path],
    eval_tasks: Sequence[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "refactor.bayesian_peft_layout.run.main",
        "prob-ensemble",
        "--task",
        SOURCE_TASK,
        "--adapter-dirs",
        ",".join(str(path) for path in map_dirs),
        "--eval-tasks",
        ",".join(eval_tasks),
        "--seed",
        str(POSTHOC_INTERNAL_SEED),
    ]


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
    artifacts: Optional[Dict[str, object]] = None,
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
    train_peak_tags: list[str] = []
    if name.startswith("seq_constantq_order"):
        train_peak_tags = [
            tag for tag in stage_peaks if tag.startswith("TRAIN-STAGE Seq-LoRA posterior build on ")
        ]
    elif name.startswith("laplace_order"):
        train_peak_tags = [
            tag for tag in stage_peaks if tag.startswith("OFFICIAL SOURCE Laplace fit on ")
        ]
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


def load_status_payload(result_root: Path, name: str) -> Dict[str, object]:
    path = result_root / "status" / f"{name}.json"
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Malformed status payload: {path}")
    return payload


__all__ = [
    "build_blob_command",
    "build_ensemble_command",
    "build_laplace_command",
    "build_map_eval_command",
    "build_map_train_command",
    "build_mcdrop_command",
    "build_seq_command",
    "load_status_payload",
    "run_and_record",
]
