from __future__ import annotations

from typing import Dict, List, Sequence
import random
import time

import torch

from .adapter_loading import (
    load_base_and_adapter,
    load_base_and_adapters,
)
from .common import (
    StageTimer,
    cuda_sync,
    parse_eval_tasks,
    peak_alloc_gb,
    peak_reserved_gb,
    prepare_eval_tasks,
    reset_cuda_peak,
    resolve_device_amp_dtype,
)
from .methods import (
    evaluate_deep_ensemble_dataset,
    evaluate_map_dataset,
    evaluate_mc_dropout_dataset,
    evaluate_probability_ensemble,
    predict_map_probabilities,
)

DEFAULT_EVAL_BATCH_SIZE = {
    "map": 256,
    "mcdrop": 32,
    "deep-ensemble": 256,
    "prob-ensemble": 256,
}


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _print_header(title: str) -> None:
    print("\n========================")
    print(f"{title:^24}")
    print("========================")


def _print_metrics(tag: str, method_label: str, metrics) -> None:
    extras: List[str] = []
    if metrics.std is not None:
        extras.append(f"std={metrics.std:.4f}")
    if metrics.mc_samples is not None:
        extras.append(f"mc={metrics.mc_samples}")
    if metrics.n_models is not None:
        extras.append(f"members={metrics.n_models}")
    suffix = ("  " + "  ".join(extras)) if extras else ""
    print(f"\n[{tag}][{method_label}]")
    print(
        f"  NLL={metrics.nll:.4f}  ACC={metrics.acc*100:.2f}%  "
        f"ECE={metrics.ece*100:.2f}%  Brier={metrics.brier:.4f}{suffix}"
    )


def _default_eval_bsz(method: str, eval_bsz: int | None) -> int:
    if eval_bsz is not None:
        return int(eval_bsz)
    return int(DEFAULT_EVAL_BATCH_SIZE[method])


def run_map_evaluation(
    *,
    task: str,
    adapter_dir: str,
    eval_tasks: Sequence[str] | None = None,
    eval_task_spec: str = "",
    max_seq_len: int = 300,
    eval_bsz: int | None = None,
    seed: int = 0,
    trust_remote_code: bool = False,
) -> None:
    _seed_everything(seed)
    device, amp_dtype = resolve_device_amp_dtype()
    print("Using device:", device, "amp_dtype:", amp_dtype)

    with StageTimer(f"LOAD-STAGE MAP on {task}"):
        loaded = load_base_and_adapter(
            task=task,
            adapter_dir=adapter_dir,
            amp_dtype=amp_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    requested_eval_tasks = (
        list(eval_tasks)
        if eval_tasks is not None
        else parse_eval_tasks(eval_task_spec, task)
    )
    prepared_tasks = prepare_eval_tasks(
        source_task=task,
        eval_tasks=requested_eval_tasks,
        tokenizer=loaded.tokenizer,
        max_seq_len=max_seq_len,
        eval_batch_size=_default_eval_bsz("map", eval_bsz),
        device=device,
        expected_num_classes=loaded.num_classes,
    )

    _print_header("MAP ONLY")
    for prepared in prepared_tasks:
        tag = f"{prepared.spec.name}({prepared.spec.split_name})"
        with StageTimer(f"INFER MAP on {tag}"):
            metrics = evaluate_map_dataset(
                loaded.model,
                prepared.loader,
                device=device,
                amp_dtype=amp_dtype,
            )
        _print_metrics(tag, "MAP", metrics)


def run_mc_dropout_evaluation(
    *,
    task: str,
    adapter_dir: str,
    eval_tasks: Sequence[str] | None = None,
    eval_task_spec: str = "",
    max_seq_len: int = 300,
    eval_bsz: int | None = None,
    seed: int = 0,
    mc_samples: int = 32,
    temp: float = 1.0,
    trust_remote_code: bool = False,
) -> None:
    _seed_everything(seed)
    device, amp_dtype = resolve_device_amp_dtype()
    print("Using device:", device, "amp_dtype:", amp_dtype)

    with StageTimer(f"LOAD-STAGE MCDrop on {task}"):
        loaded = load_base_and_adapter(
            task=task,
            adapter_dir=adapter_dir,
            amp_dtype=amp_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    requested_eval_tasks = (
        list(eval_tasks)
        if eval_tasks is not None
        else parse_eval_tasks(eval_task_spec, task)
    )
    prepared_tasks = prepare_eval_tasks(
        source_task=task,
        eval_tasks=requested_eval_tasks,
        tokenizer=loaded.tokenizer,
        max_seq_len=max_seq_len,
        eval_batch_size=_default_eval_bsz("mcdrop", eval_bsz),
        device=device,
        expected_num_classes=loaded.num_classes,
    )

    _print_header("MCDROP ONLY")
    print(f"[Config] MC_SAMPLES={mc_samples} TEMP={temp}")
    for prepared in prepared_tasks:
        tag = f"{prepared.spec.name}({prepared.spec.split_name})"
        with StageTimer(f"INFER MCDrop on {tag}"):
            metrics = evaluate_mc_dropout_dataset(
                loaded.model,
                prepared.loader,
                device=device,
                amp_dtype=amp_dtype,
                mc_samples=mc_samples,
                temp=temp,
            )
        _print_metrics(tag, "MCDROP", metrics)


def run_deep_ensemble_evaluation(
    *,
    task: str,
    adapter_dirs: Sequence[str],
    eval_tasks: Sequence[str] | None = None,
    eval_task_spec: str = "",
    max_seq_len: int = 300,
    eval_bsz: int | None = None,
    seed: int = 0,
    temp: float = 1.0,
    trust_remote_code: bool = False,
) -> None:
    adapter_dirs = [str(path) for path in adapter_dirs if str(path).strip()]
    if len(adapter_dirs) < 2:
        raise ValueError("Deep Ensemble evaluation expects at least two adapter dirs.")

    _seed_everything(seed)
    device, amp_dtype = resolve_device_amp_dtype()
    print("Using device:", device, "amp_dtype:", amp_dtype)

    with StageTimer(f"LOAD-STAGE DeepEns on {task}"):
        loaded = load_base_and_adapters(
            task=task,
            adapter_dirs=adapter_dirs,
            amp_dtype=amp_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    requested_eval_tasks = (
        list(eval_tasks)
        if eval_tasks is not None
        else parse_eval_tasks(eval_task_spec, task)
    )
    prepared_tasks = prepare_eval_tasks(
        source_task=task,
        eval_tasks=requested_eval_tasks,
        tokenizer=loaded.tokenizer,
        max_seq_len=max_seq_len,
        eval_batch_size=_default_eval_bsz("deep-ensemble", eval_bsz),
        device=device,
        expected_num_classes=loaded.num_classes,
    )

    _print_header("DEEP ENS ONLY")
    print(f"[Config] n_models={len(adapter_dirs)} TEMP={temp}")
    for prepared in prepared_tasks:
        tag = f"{prepared.spec.name}({prepared.spec.split_name})"
        with StageTimer(f"INFER DeepEns on {tag}"):
            metrics = evaluate_deep_ensemble_dataset(
                loaded.model,
                prepared.loader,
                device=device,
                amp_dtype=amp_dtype,
                temp=temp,
            )
        _print_metrics(tag, "DEEP-ENS", metrics)


def run_probability_ensemble_evaluation(
    *,
    task: str,
    adapter_dirs: Sequence[str],
    eval_tasks: Sequence[str] | None = None,
    eval_task_spec: str = "",
    max_seq_len: int = 300,
    eval_bsz: int | None = None,
    seed: int = 0,
    trust_remote_code: bool = False,
) -> None:
    adapter_dirs = [str(path) for path in adapter_dirs if str(path).strip()]
    if len(adapter_dirs) < 2:
        raise ValueError("Probability ensemble expects at least two adapter dirs.")

    _seed_everything(seed)
    device, amp_dtype = resolve_device_amp_dtype()
    print("Using device:", device, "amp_dtype:", amp_dtype)
    print(f"[Ensemble] members={len(adapter_dirs)}")

    requested_eval_tasks = (
        list(eval_tasks)
        if eval_tasks is not None
        else parse_eval_tasks(eval_task_spec, task)
    )

    prepared_tasks = None
    task_prob_sums: Dict[str, torch.Tensor] = {}
    task_labels: Dict[str, torch.Tensor] = {}
    task_time_sec: Dict[str, float] = {}
    task_peak_alloc: Dict[str, float] = {}
    task_peak_reserved: Dict[str, float] = {}

    for member_idx, adapter_dir in enumerate(adapter_dirs, start=1):
        _print_header(f"MEMBER {member_idx}/{len(adapter_dirs)}")
        loaded = load_base_and_adapter(
            task=task,
            adapter_dir=adapter_dir,
            amp_dtype=amp_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        if prepared_tasks is None:
            prepared_tasks = prepare_eval_tasks(
                source_task=task,
                eval_tasks=requested_eval_tasks,
                tokenizer=loaded.tokenizer,
                max_seq_len=max_seq_len,
                eval_batch_size=_default_eval_bsz("prob-ensemble", eval_bsz),
                device=device,
                expected_num_classes=loaded.num_classes,
            )
            for prepared in prepared_tasks:
                task_prob_sums[prepared.spec.name] = torch.zeros(
                    (len(prepared.processed_dataset), loaded.num_classes),
                    dtype=torch.float32,
                )
                task_time_sec[prepared.spec.name] = 0.0
                task_peak_alloc[prepared.spec.name] = 0.0
                task_peak_reserved[prepared.spec.name] = 0.0

        assert prepared_tasks is not None
        for prepared in prepared_tasks:
            tag = f"{prepared.spec.name}({prepared.spec.split_name})"
            reset_cuda_peak()
            cuda_sync()
            t0 = time.perf_counter()
            probs, labels = predict_map_probabilities(
                loaded.model,
                prepared.loader,
                device=device,
                amp_dtype=amp_dtype,
            )
            cuda_sync()
            dt = time.perf_counter() - t0
            task_time_sec[prepared.spec.name] += float(dt)
            task_peak_alloc[prepared.spec.name] = max(
                task_peak_alloc[prepared.spec.name],
                peak_alloc_gb(),
            )
            task_peak_reserved[prepared.spec.name] = max(
                task_peak_reserved[prepared.spec.name],
                peak_reserved_gb(),
            )
            print(
                f"[Member {member_idx}] {tag}: time={dt:.2f} sec ({dt/60:.2f} min) "
                f"alloc={peak_alloc_gb():.2f}GB reserved={peak_reserved_gb():.2f}GB"
            )

            if prepared.spec.name not in task_labels:
                task_labels[prepared.spec.name] = labels.detach().cpu()
            elif not torch.equal(task_labels[prepared.spec.name], labels.detach().cpu()):
                raise RuntimeError(
                    f"Label ordering mismatch while building ensemble for task {prepared.spec.name}."
                )
            task_prob_sums[prepared.spec.name] += probs.detach().cpu()

        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert prepared_tasks is not None
    _print_header("PROB ENS ONLY")
    for prepared in prepared_tasks:
        tag = f"{prepared.spec.name}({prepared.spec.split_name})"
        print(
            f"[TIME] INFER Ensemble on {tag}: "
            f"{task_time_sec[prepared.spec.name]:.2f} sec "
            f"({task_time_sec[prepared.spec.name]/60:.2f} min)"
        )
        print(
            f"[PEAK] INFER Ensemble on {tag}: "
            f"alloc={task_peak_alloc[prepared.spec.name]:.2f} GB  "
            f"reserved={task_peak_reserved[prepared.spec.name]:.2f} GB"
        )
        metrics = evaluate_probability_ensemble(
            probs_sum=task_prob_sums[prepared.spec.name],
            labels=task_labels[prepared.spec.name],
            n_members=len(adapter_dirs),
        )
        _print_metrics(tag, "ENSEMBLE", metrics)


__all__ = [
    "run_deep_ensemble_evaluation",
    "run_map_evaluation",
    "run_mc_dropout_evaluation",
    "run_probability_ensemble_evaluation",
]
