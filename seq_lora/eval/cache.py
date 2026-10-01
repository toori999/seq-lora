from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import torch


POSTERIOR_STATS_CACHE_FORMAT = "seq_lora_posterior_stats_v2"
LEGACY_POSTERIOR_STATS_CACHE_FORMATS = {
    "seq_lora_eval_posterior_stats_v2",
    "seq_lora_constantq_posterior_stats_v1",
    "seq_lora_constantq_bayesian_peft_posterior_stats_v1",
}
LEGACY_OPTIONAL_CACHE_SNAPSHOT_KEYS = {
    # Dataset fingerprints can change across code refactors even when the
    # materialized examples and slice ids are unchanged.
    "train_proc_fingerprint",
    "posterior_stats_dtype",
    "eval_protocol",
    "bayesian_peft_add_space",
    "bayesian_peft_add_eos",
}


def cache_norm_path(path: str) -> str:
    path = str(path or "").strip()
    return os.path.abspath(os.path.expanduser(path)) if path else ""


def dataset_fingerprint(ds: object) -> str:
    fingerprint = getattr(ds, "_fingerprint", None)
    if fingerprint is not None:
        return str(fingerprint)
    try:
        n = len(ds)  # type: ignore[arg-type]
    except Exception:
        n = "unknown"
    return f"{type(ds).__name__}:n={n}"


def serialize_module_specs_cpu(module_specs: List[Dict]) -> List[Dict[str, object]]:
    specs_cpu: List[Dict[str, object]] = []
    for spec in module_specs:
        subspace_info_cpu = {}
        for key, value in spec["subspace_info"].items():
            if torch.is_tensor(value):
                subspace_info_cpu[key] = value.detach().cpu()
            else:
                subspace_info_cpu[key] = value
        specs_cpu.append(
            {
                "name": str(spec["name"]),
                "offset": int(spec["offset"]),
                "L": int(spec["L"]),
                "subspace_info": subspace_info_cpu,
            }
        )
    return specs_cpu


def build_posterior_stats_cache_snapshot(
    args: argparse.Namespace,
    *,
    slice_ids: Sequence[int],
    train_raw_fingerprint: str,
    train_proc_fingerprint: str,
) -> Dict[str, object]:
    return {
        "task": str(args.task),
        "map_dir": cache_norm_path(str(args.map_dir)),
        "slices_dir": cache_norm_path(str(args.slices_dir)),
        "seed": int(args.seed),
        "kfac_backend": str(args.kfac_backend),
        "random_num_slices": int(args.random_num_slices),
        "slice_order": str(args.slice_order),
        "slice_order_seed": int(args.slice_order_seed),
        "slice_ids": [int(x) for x in slice_ids],
        "train_raw_fingerprint": str(train_raw_fingerprint),
        "train_proc_fingerprint": str(train_proc_fingerprint),
        "subspace_dim_per_module": int(args.subspace_dim_per_module),
        "max_seq_len": int(args.max_seq_len),
        "kfac_bsz": int(args.kfac_bsz),
        "n_kfac": int(args.n_kfac),
        "lr_threshold": int(args.lr_threshold),
        "posterior_stats_dtype": str(args.posterior_stats_dtype),
        "max_kfac_samples_per_slice": int(args.max_kfac_samples_per_slice),
        "kfac_tail_policy": "keep",
        "mu_obs_batches": int(args.mu_obs_batches),
        "disable_dropout_during_kfac_mu": bool(args.disable_dropout_during_kfac_mu),
        "tokenizer_padding_side": str(args.tokenizer_padding_side),
        "keep_full_vocab_lm_head": bool(args.keep_full_vocab_lm_head),
        "eval_protocol": str(args.eval_protocol),
        "bayesian_peft_add_space": bool(args.bayesian_peft_add_space),
        "bayesian_peft_add_eos": bool(args.bayesian_peft_add_eos),
    }


def build_posterior_stats_cache_payload(
    *,
    args: argparse.Namespace,
    snapshot: Dict[str, object],
    module_specs: List[Dict],
    module_R_lists: Dict[str, List[torch.Tensor]],
    mu_global_list_raw: List[torch.Tensor],
    T: int,
) -> Dict[str, object]:
    return {
        "format": POSTERIOR_STATS_CACHE_FORMAT,
        "args_snapshot": dict(snapshot),
        "module_specs": serialize_module_specs_cpu(module_specs),
        "module_R_lists": {
            str(name): [tensor.detach().cpu() for tensor in tensors]
            for name, tensors in module_R_lists.items()
        },
        "mu_global_list_raw": [mu_t.detach().cpu() for mu_t in mu_global_list_raw],
        "T": int(T),
        "num_modules": int(len(module_specs)),
        "L_total": int(sum(int(spec["L"]) for spec in module_specs)),
        "created_by": "seq_lora.eval.runner",
        "seed": int(args.seed),
    }


def validate_cache_snapshot(
    *,
    cache_path: str,
    snapshot: Dict[str, object],
    expected: Dict[str, object],
) -> None:
    mismatches = [
        f"{key}: cache={snapshot.get(key)!r} current={expected[key]!r}"
        for key in expected
        if snapshot.get(key) != expected[key]
    ]
    if mismatches:
        mismatch_text = "\n".join(mismatches[:18])
        raise RuntimeError(
            f"Posterior-stats cache at {cache_path} does not match this run:\n"
            f"{mismatch_text}\n"
            "Use a matching cache path or pass --force_rebuild_posterior_stats_cache."
        )


def load_posterior_stats_cache(
    cache_path: str,
    expected_snapshot: Dict[str, object],
) -> Tuple[List[Dict], Dict[str, List[torch.Tensor]], List[torch.Tensor], int]:
    payload = torch.load(cache_path, map_location="cpu")
    cache_format = payload.get("format") if isinstance(payload, dict) else None
    accepted_formats = {POSTERIOR_STATS_CACHE_FORMAT, *LEGACY_POSTERIOR_STATS_CACHE_FORMATS}
    if not isinstance(payload, dict) or cache_format not in accepted_formats:
        raise RuntimeError(
            f"Unsupported posterior-stats cache format in {cache_path}. "
            f"Expected one of {sorted(accepted_formats)!r}."
        )

    snapshot = dict(payload.get("args_snapshot") or {})
    expected_for_validation = dict(expected_snapshot)
    if cache_format in LEGACY_POSTERIOR_STATS_CACHE_FORMATS:
        for key in LEGACY_OPTIONAL_CACHE_SNAPSHOT_KEYS:
            expected_for_validation.pop(key, None)
    validate_cache_snapshot(
        cache_path=cache_path,
        snapshot=snapshot,
        expected=expected_for_validation,
    )

    module_specs = payload.get("module_specs")
    module_R_lists = payload.get("module_R_lists")
    mu_global_list_raw = payload.get("mu_global_list_raw")
    T = int(payload.get("T"))
    if (
        not isinstance(module_specs, list)
        or not isinstance(module_R_lists, dict)
        or not isinstance(mu_global_list_raw, list)
    ):
        raise RuntimeError(f"Posterior-stats cache at {cache_path} is missing required tensors.")
    return module_specs, module_R_lists, mu_global_list_raw, T
