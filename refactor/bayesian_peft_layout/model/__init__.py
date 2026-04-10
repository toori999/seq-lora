from .choice_head import (
    ChoiceHeadCache,
    build_choice_head_cache,
    logits_via_lm_head_last_token_for_kfac,
    restricted_choice_logits_last_token,
)
from .loaders import (
    LoadedAdapterModel,
    force_lora_fp32,
    load_base_and_adapter,
    load_base_and_adapters,
    peft_set_adapter,
    trim_lm_head_to_choice_tokens,
)
from .lora_training import (
    DEFAULT_ATTENTION_TARGET_MODULES,
    DEFAULT_LM_HEAD_TARGET_MODULES,
    freeze_base_enable_lora,
    get_lora_state_dict_cpu,
    get_normalized_lora_state_dict_cpu,
    load_lora_state_dict,
    load_normalized_lora_state_dict,
    resolve_qv_lm_head_target_modules,
    sync_or_create_shared_lora_init,
)
from .map_training import (
    build_scienceqa_dataloaders,
    run_refactor_scienceqa_map_training,
    train_map_lora,
)
from .train_runtime import (
    build_training_tokenizer,
    enable_gradient_checkpointing,
    load_causal_lm_with_attn_fallback,
    resolve_device_amp_dtype,
    seed_worker,
    set_seed,
)
from .training_eval import (
    compute_choice_logits,
    compute_multiclass_ece,
    eval_next_token,
    mask_invalid_choices,
)

__all__ = [
    "ChoiceHeadCache",
    "build_scienceqa_dataloaders",
    "build_training_tokenizer",
    "compute_choice_logits",
    "compute_multiclass_ece",
    "DEFAULT_ATTENTION_TARGET_MODULES",
    "DEFAULT_LM_HEAD_TARGET_MODULES",
    "enable_gradient_checkpointing",
    "eval_next_token",
    "LoadedAdapterModel",
    "build_choice_head_cache",
    "force_lora_fp32",
    "freeze_base_enable_lora",
    "get_lora_state_dict_cpu",
    "get_normalized_lora_state_dict_cpu",
    "load_causal_lm_with_attn_fallback",
    "load_base_and_adapter",
    "load_base_and_adapters",
    "load_lora_state_dict",
    "load_normalized_lora_state_dict",
    "logits_via_lm_head_last_token_for_kfac",
    "mask_invalid_choices",
    "peft_set_adapter",
    "resolve_device_amp_dtype",
    "resolve_qv_lm_head_target_modules",
    "restricted_choice_logits_last_token",
    "run_refactor_scienceqa_map_training",
    "seed_worker",
    "set_seed",
    "sync_or_create_shared_lora_init",
    "train_map_lora",
    "trim_lm_head_to_choice_tokens",
]
