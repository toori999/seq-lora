from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

from .benchmark_config import SOURCE_TASK
from .map_variants import load_map_variant_configs, load_map_variant_runner


@dataclass(frozen=True)
class ScienceQAMapTrainConfig:
    variant: str
    seed: int
    micro_bsz: int
    grad_accum: int
    eval_bsz: int
    task: str
    module_name: str
    run_tag: str
    output_dir: Path
    slice_dir: Path
    base_model_name: str
    trust_remote_code: bool
    attn_implementation: str
    fallback_attn_implementation: str
    source_eval_split: str
    max_seq_len: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    max_steps: int
    save_every: int
    eval_every: int
    map_step_for_table: int
    num_workers: int
    use_gradient_checkpointing: bool
    fast_but_nondeterministic: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_choices: int
    tokenizer_padding_side: str

    @property
    def effective_train_bsz(self) -> int:
        return int(self.micro_bsz) * int(self.grad_accum)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_tag / f"seed_{self.seed}"

    @property
    def map_dir(self) -> Path:
        return self.run_dir / f"map_step_{self.map_step_for_table}"

    @property
    def init_lora_path(self) -> Path:
        return self.run_dir / "init_lora.pt"

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["slice_dir"] = str(self.slice_dir)
        payload["run_dir"] = str(self.run_dir)
        payload["map_dir"] = str(self.map_dir)
        payload["init_lora_path"] = str(self.init_lora_path)
        payload["effective_train_bsz"] = self.effective_train_bsz
        return payload


def build_scienceqa_map_train_config(
    *,
    variant: str,
    seed: int,
    micro_bsz: int,
    grad_accum: int,
    eval_bsz: int,
) -> ScienceQAMapTrainConfig:
    configs = load_map_variant_configs()
    if variant not in configs:
        known = ", ".join(sorted(configs))
        raise KeyError(f"Unknown MAP variant {variant!r}; expected one of: {known}")

    runner = load_map_variant_runner(variant)
    variant_cfg = configs[variant]
    return ScienceQAMapTrainConfig(
        variant=variant,
        seed=int(seed),
        micro_bsz=int(micro_bsz),
        grad_accum=int(grad_accum),
        eval_bsz=int(eval_bsz),
        task=SOURCE_TASK,
        module_name=variant_cfg.module_name,
        run_tag=variant_cfg.run_tag,
        output_dir=variant_cfg.output_dir,
        slice_dir=variant_cfg.slice_dir,
        base_model_name=str(getattr(runner, "BASE_MODEL_NAME", "")),
        trust_remote_code=bool(getattr(runner, "TRUST_REMOTE_CODE", False)),
        attn_implementation=str(getattr(runner, "ATTN_IMPLEMENTATION", "sdpa")),
        fallback_attn_implementation=str(
            getattr(runner, "FALLBACK_ATTN_IMPLEMENTATION", "sdpa")
        ),
        source_eval_split=str(getattr(runner, "SOURCE_EVAL_SPLIT", "test")),
        max_seq_len=int(getattr(runner, "MAX_SEQ_LEN", 300)),
        lr=float(getattr(runner, "LR", 5e-5)),
        weight_decay=float(getattr(runner, "WEIGHT_DECAY", 0.01)),
        warmup_ratio=float(getattr(runner, "WARMUP_RATIO", 0.06)),
        max_steps=int(getattr(runner, "MAX_STEPS", 0)),
        save_every=int(getattr(runner, "SAVE_EVERY", 0)),
        eval_every=int(getattr(runner, "EVAL_EVERY", 0)),
        map_step_for_table=int(getattr(runner, "MAP_STEP_FOR_TABLE", 0)),
        num_workers=int(getattr(runner, "NUM_WORKERS", 0)),
        use_gradient_checkpointing=bool(
            getattr(runner, "USE_GRADIENT_CHECKPOINTING", False)
        ),
        fast_but_nondeterministic=bool(
            getattr(runner, "FAST_BUT_NONDETERMINISTIC", True)
        ),
        lora_r=int(getattr(runner, "LORA_R", 8)),
        lora_alpha=int(getattr(runner, "LORA_ALPHA", 16)),
        lora_dropout=float(getattr(runner, "LORA_DROPOUT", 0.05)),
        max_choices=int(getattr(runner, "MAX_CHOICES", 4)),
        tokenizer_padding_side=str(getattr(runner, "TOKENIZER_PADDING_SIDE", "left")),
    )


def apply_train_config_to_runner(runner, config: ScienceQAMapTrainConfig):
    runner.MICRO_BSZ = int(config.micro_bsz)
    runner.GRAD_ACCUM = int(config.grad_accum)
    runner.EVAL_BSZ = int(config.eval_bsz)
    runner.SEEDS = [int(config.seed)]
    return runner


__all__ = [
    "ScienceQAMapTrainConfig",
    "apply_train_config_to_runner",
    "build_scienceqa_map_train_config",
]
