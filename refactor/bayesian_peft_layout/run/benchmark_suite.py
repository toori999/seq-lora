from __future__ import annotations

import argparse
from pathlib import Path

from refactor.bayesian_peft_layout.utils.benchmark_commands import (
    build_blob_command,
    build_ensemble_command,
    build_laplace_command,
    build_map_eval_command,
    build_map_train_command,
    build_mcdrop_command,
    build_seq_command,
    load_status_payload,
    run_and_record,
)
from refactor.bayesian_peft_layout.utils.benchmark_config import (
    DEFAULT_ENSEMBLE_GROUPS,
    DEFAULT_ENSEMBLE_TOTAL_SEEDS,
    DEFAULT_EVAL_TASKS,
    SOURCE_TASK,
    build_consecutive_ensemble_groups,
    expand_eval_tasks,
    load_map_variant_configs,
    parse_int_list,
    utc_now,
)
from refactor.bayesian_peft_layout.utils.benchmark_exports import refresh_exports, write_json
from refactor.bayesian_peft_layout.utils.benchmark_parsing import (
    parse_blob_output,
    parse_ensemble_output,
    parse_laplace_output,
    parse_map_eval_output,
    parse_mcdrop_output,
    parse_seq_output,
    parse_train_wall_only,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ScienceQA benchmark suite using the bayesian-peft-style "
            "refactor layout."
        )
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument(
        "--result-root",
        type=str,
        default="./benchmark_suite_scienceqa_refactor",
    )
    parser.add_argument("--eval-tasks", type=str, default=",".join(DEFAULT_EVAL_TASKS))
    parser.add_argument("--map-micro-bsz", type=int, default=4)
    parser.add_argument("--map-grad-accum", type=int, default=2)
    parser.add_argument("--map-eval-bsz", type=int, default=32)
    parser.add_argument("--constant-q-var", type=float, default=1.0)
    parser.add_argument("--blob-eval-n", type=int, default=10)
    parser.add_argument("--mcdrop-mc-samples", type=int, default=32)
    parser.add_argument("--mcdrop-temp", type=float, default=1.0)
    parser.add_argument("--laplace-fit-bsz", type=int, default=2)
    parser.add_argument("--laplace-bsz", type=int, default=4)
    parser.add_argument("--laplace-prior-optim-step", type=int, default=100)
    parser.add_argument("--laplace-mc-samples", type=int, default=32)
    parser.add_argument("--laplace-mc-chunk", type=int, default=8)
    parser.add_argument(
        "--ensemble-total-seeds",
        type=int,
        default=DEFAULT_ENSEMBLE_TOTAL_SEEDS,
    )
    parser.add_argument(
        "--ensemble-num-groups",
        type=int,
        default=DEFAULT_ENSEMBLE_GROUPS,
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cwd = Path.cwd()
    result_root = Path(args.result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds)
    eval_tasks = expand_eval_tasks(
        [task.strip() for task in args.eval_tasks.split(",") if task.strip()]
    )
    ensemble_groups = build_consecutive_ensemble_groups(
        int(args.ensemble_total_seeds),
        int(args.ensemble_num_groups),
    )
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
            "constant_q_var": float(args.constant_q_var),
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

    for order_key in ("order", "reverse", "random"):
        cfg = configs[order_key]
        for seed in seeds:
            map_dir = cfg.map_dir(seed)
            run_and_record(
                name=f"train_map_{order_key}_seed{seed}",
                cmd=build_map_train_command(
                    order_key,
                    seed,
                    int(args.map_micro_bsz),
                    int(args.map_grad_accum),
                    int(args.map_eval_bsz),
                ),
                parser_fn=parse_train_wall_only,
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
                constant_q_var=float(args.constant_q_var),
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

        laplace_output_dir = (cwd / "outputs_laplace_official_source_qv_lmhead_suite" / f"seed_{seed}").resolve()
        run_and_record(
            name=f"laplace_order_seed{seed}",
            cmd=build_laplace_command(
                map_dir=map_dir,
                output_dir=laplace_output_dir,
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
                "output_dir": str(laplace_output_dir),
            },
            resume=args.resume,
        )

    additional_ensemble_seeds = [seed for seed in ensemble_all_seeds if seed not in seeds]
    for seed in additional_ensemble_seeds:
        map_dir = order_cfg.map_dir(seed)
        run_and_record(
            name=f"train_map_order_seed{seed}",
            cmd=build_map_train_command(
                "order",
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
        member_train_payloads = [
            load_status_payload(result_root, f"train_map_order_seed{seed}")
            for seed in member_seeds
        ]
        ensemble_train_time_sec = sum(
            float(p.get("train_time_sec") or 0.0) for p in member_train_payloads
        )
        peak_alloc_vals = [
            float(p["train_peak_alloc_gb"])
            for p in member_train_payloads
            if p.get("train_peak_alloc_gb") is not None
        ]
        peak_reserved_vals = [
            float(p["train_peak_reserved_gb"])
            for p in member_train_payloads
            if p.get("train_peak_reserved_gb") is not None
        ]
        ensemble_train_peak_alloc_gb = max(peak_alloc_vals) if peak_alloc_vals else None
        ensemble_train_peak_reserved_gb = (
            max(peak_reserved_vals) if peak_reserved_vals else None
        )
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
