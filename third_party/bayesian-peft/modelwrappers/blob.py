import os
import sys
import torch
import torch.nn as nn
from torch.optim import SGD
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import math
import logging
import re
from tqdm import tqdm
from pathlib import Path

from .wrapperbase import WrapperBase, _get_torchmetrics, get_linear_schedule_with_warmup
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from run.evaluation import *
from utils import StageTimer, create_if_not_exists

from transformers import PreTrainedModel

from peft.config import PeftConfig
from peft.tuners.lora import LoraLayer, Linear
from peft.tuners.lora.bnb import Linear8bitLt


## Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Bayesian By Backprop, BLoB.")
    add_management_args(parser)
    add_experiment_args(parser)
    # BLoB-specific arguments.
    parser.add_argument("--bayes-train-n-samples", type=int, default=1)
    parser.add_argument(
        "--bayes-eval-n-samples",
        type=int,
        default=1,
        help="Number of samples to use for evaluation during training.",
    )
    parser.add_argument(
        "--bayes-eval-n-samples-final",
        type=int,
        default=10,
        help="Number of samples to use for evaluation.",
    )

    parser.add_argument("--bayes-eps", type=float, default=0.05)
    parser.add_argument("--bayes-gamma", type=float, default=8)
    parser.add_argument("--bayes-kllr", type=float, default=0.02)
    parser.add_argument("--bayes-beta", type=float, default=0.2)
    parser.add_argument(
        "--bayes-inference-notsample",
        action="store_true",
        help="Whether to sample during inference.",
    )
    parser.add_argument(
        "--bayes-klreweighting", action="store_true", help="Whether to use reweighting."
    )
    parser.add_argument('--bayes-datasetrescaling', action='store_true',
                        help='Whether to use datasetrescaling.')

    return parser


@dataclass
class BLoBConfig:
    bayes_eps: float = field(metadata={"help": "Bayes epsilon"})
    bayes_beta: float = field(metadata={"help": "Bayes beta"})


def _load_seq_lora_helpers():
    try:
        import common_eval_utils as ceu  # type: ignore

        return ceu
    except Exception:
        pass

    candidates = []
    env_root = os.getenv("SEQ_LORA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path.cwd())

    for root in candidates:
        root = Path(root)
        if (root / "common_eval_utils.py").exists():
            sys.path.insert(0, str(root))
            import common_eval_utils as ceu  # type: ignore

            return ceu

    return None


_CEU = _load_seq_lora_helpers()
_LORA_ADAPTER_PLACEHOLDER = "__adapter__"
_LORA_ADAPTER_RE = re.compile(r"(\.lora_(?:A|B)\.)([^.]+)(\.)")


def _is_mc_dataset_type(dataset_type: str) -> bool:
    return str(dataset_type).strip().lower() in {"mcdataset", "benchmark_mcdataset"}


def _uses_trimmed_mc_head(dataset_type: str) -> bool:
    return str(dataset_type).strip().lower() == "benchmark_mcdataset"


def _multiclass_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = F.one_hot(labels, num_classes=probs.size(-1)).to(dtype=probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


def _normalize_lora_key(key: str) -> str:
    return _LORA_ADAPTER_RE.sub(rf"\1{_LORA_ADAPTER_PLACEHOLDER}\3", key)


def _denormalize_lora_key(key: str, adapter_name: str) -> str:
    return key.replace(f".{_LORA_ADAPTER_PLACEHOLDER}.", f".{adapter_name}.")


def _load_normalized_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor], adapter_name: str) -> None:
    mapped = {_denormalize_lora_key(k, adapter_name): v for k, v in lora_state.items()}
    model.load_state_dict(mapped, strict=False)


def _iter_lora_linear_modules(model: nn.Module):
    for name, mod in model.named_modules():
        if not isinstance(mod, LoraLayer):
            continue
        if isinstance(mod, Linear) or isinstance(mod, Linear8bitLt):
            yield name, mod


def _resolve_blob_paths(blob_dir: str):
    adapter_dir = blob_dir
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        subdir = os.path.join(blob_dir, "blob")
        if os.path.exists(os.path.join(subdir, "adapter_config.json")):
            adapter_dir = subdir
        else:
            raise FileNotFoundError(
                f"Could not find adapter_config.json in '{blob_dir}' or '{subdir}'"
            )

    rho_candidates = [
        os.path.join(blob_dir, "blob_rho.pt"),
        os.path.join(adapter_dir, "blob_rho.pt"),
    ]
    rho_path = next((p for p in rho_candidates if os.path.exists(p)), None)
    if rho_path is None:
        raise FileNotFoundError(f"Missing blob rho file. Tried: {rho_candidates}")
    return adapter_dir, rho_path


def _load_blob_rho(model: nn.Module, adapter_name: str, rho_path: str) -> None:
    saved = torch.load(rho_path, map_location="cpu")
    loaded = 0
    for i, (_, mod) in enumerate(_iter_lora_linear_modules(model)):
        if not hasattr(mod, "lora_A_rho") or adapter_name not in mod.lora_A_rho:
            continue
        key = f"{i}::{type(mod).__name__}"
        if key not in saved:
            raise KeyError(f"Missing rho tensor for key '{key}' in {rho_path}")
        rho = mod.lora_A_rho[adapter_name]
        rho.data.copy_(saved[key].to(device=rho.device, dtype=rho.dtype))
        loaded += 1
    print(f"[BLoB] loaded rho for {loaded} modules from: {rho_path}")


def _collect_blob_rho_state(model: nn.Module, adapter_name: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for i, (_, mod) in enumerate(_iter_lora_linear_modules(model)):
        if not hasattr(mod, "lora_A_rho") or adapter_name not in mod.lora_A_rho:
            continue
        out[f"{i}::{type(mod).__name__}"] = mod.lora_A_rho[adapter_name].detach().cpu()
    return out


def blob_linear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    previous_dtype = x.dtype

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result += lora_B(lora_A(dropout(x))) * scaling

    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        if self.blobsample:
            rank = int(lora_A.weight.shape[0])
            if self.bayes_eps < 0:
                A_sigma = torch.log1p(torch.exp(self.lora_A_rho[active_adapter]))
            else:
                A_sigma = self.lora_A_rho[active_adapter] ** 2

            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]

            x = x.to(lora_A.weight.dtype)
            if x.dim() == 2:
                r_A = (
                    torch.ones(
                        (x.size(0), self.in_features), device=x.device, dtype=x.dtype
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), rank),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
            else:
                r_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.in_features),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), x.size(1), rank),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )

            x = dropout(x)
            lora_noise_a = A_sigma * torch.randn_like(
                self.lora_A[active_adapter].weight
            )

            noise = (((x * r_A) @ lora_noise_a.transpose(0, 1)) * s_A) @ self.lora_B[
                active_adapter
            ].weight.transpose(0, 1)

            result += noise * scaling

        result = result.to(previous_dtype)

    return result


def blob_8bitlinear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x.dtype != compute_dtype:
                    x = x.to(compute_dtype)
            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling
            result += output
    if self.blobsample:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            rank = int(lora_A.weight.shape[0])
            if self.bayes_eps < 0:
                A_sigma = torch.log1p(torch.exp(self.lora_A_rho[active_adapter]))
            else:
                A_sigma = self.lora_A_rho[active_adapter] ** 2
            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x.dtype != compute_dtype:
                    x = x.to(compute_dtype)

            if x.dim() == 2:
                r_A = (
                    torch.ones(
                        (x.size(0), self.in_features), device=x.device, dtype=x.dtype
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), rank),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
            else:
                r_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.in_features),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), x.size(1), rank),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    .uniform_(-1, 1)
                    .sign()
                )

            x = dropout(x)
            lora_noise_a = A_sigma * torch.randn_like(
                self.lora_A[active_adapter].weight
            )

            noise = (((x * r_A) @ lora_noise_a.transpose(0, 1)) * s_A) @ self.lora_B[
                active_adapter
            ].weight.transpose(0, 1)

            if requires_conversion:
                noise = noise.to(expected_dtype)

            result += noise * scaling

    return result


def div_posterior_prior(self) -> torch.Tensor:
    def kl_div_stable(mu_q, sigma_q, mu_p, sigma_p):
        eps = 1e-6
        kl = (
            math.log(sigma_p + eps)
            - torch.log(sigma_q.to(torch.float64) + eps)
            + (sigma_q.to(torch.float64) ** 2 + (mu_q.to(torch.float64) - mu_p) ** 2)
            / (2 * (sigma_p**2) + eps)
            - 0.5
        )
        return kl.sum()

    kl = 0
    for active_adapter in self.active_adapters:
        if self.bayes_eps < 0:
            sigma_weight = torch.log1p(torch.exp(self.lora_A_rho[active_adapter]))
        else:
            sigma_weight = self.lora_A_rho[active_adapter] ** 2
        kl += kl_div_stable(
            self.lora_A[active_adapter].weight, sigma_weight, 0, self.bayes_beta
        )
    return kl


def sample(self, status=True):
    if self.training is True and status is False:
        raise ValueError("blobsample should be set to True only during training.")
    self.blobsample = status


class BLoB(WrapperBase):
    """BLoB model."""

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        super().__init__(model, peft_config, args, accelerator, adapter_name)

        self.blobconfig = BLoBConfig(
            bayes_eps=self.args.bayes_eps,
            bayes_beta=self.args.bayes_beta,
        )
        blob_adapter_dir = None
        blob_rho_path = None
        if args.load_blob_dir is not None:
            blob_adapter_dir, blob_rho_path = _resolve_blob_paths(args.load_blob_dir)
            print(f"[Load BLoB] adapter={blob_adapter_dir} rho={blob_rho_path}")
        elif args.shared_init_lora_path is not None:
            if not os.path.exists(args.shared_init_lora_path):
                raise FileNotFoundError(f"Missing shared init LoRA file: {args.shared_init_lora_path}")
            saved_init = torch.load(args.shared_init_lora_path, map_location="cpu")
            _load_normalized_lora_state_dict(self, saved_init, adapter_name=adapter_name)
            print(f"[Init LoRA] loaded shared init from {args.shared_init_lora_path}")
        self._modify_lora_layers(self.base_model)
        if blob_adapter_dir is not None:
            self.load_adapter(blob_adapter_dir, adapter_name)
            _load_blob_rho(self, adapter_name=adapter_name, rho_path=blob_rho_path)
        elif args.load_lora_path is not None:
            self.load_adapter(args.load_lora_path, adapter_name)

        self.i = 1  # for the KL re-weighting.
        self.ii = 1
        self.M = 0  # for the KL re-weighting.

        self.train_n_samples = self.args.bayes_train_n_samples
        self.eval_n_samples = self.args.bayes_eval_n_samples
        self.klreweighting = self.args.bayes_klreweighting

        if self.args.max_train_steps == 0:
            num_training_steps = (
                self.args.num_samples * self.args.n_epochs // self.args.batch_size
            )
        else:
            num_training_steps = self.args.max_train_steps
        warmup_steps = num_training_steps * self.args.warmup_ratio

        params = [param for name, param in self.named_parameters()]
        self.opt2 = SGD([{"params": params}], lr=args.bayes_kllr)
        self.scheduler2 = get_linear_schedule_with_warmup(
            self.opt2, warmup_steps, num_training_steps
        )

    def _save_blob_checkpoint(self, save_dir: str) -> None:
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return

        active_adapters = list(getattr(self, "active_adapters", []) or [])
        adapter_name = active_adapters[0] if active_adapters else "default"
        create_if_not_exists(save_dir)
        original_base_model = self.base_model
        try:
            self.base_model = self.accelerator.unwrap_model(self.base_model)
            self.save_pretrained(save_dir, save_function=self.accelerator.save)
        finally:
            self.base_model = original_base_model

        rho_state = _collect_blob_rho_state(self, adapter_name)
        torch.save(rho_state, os.path.join(save_dir, "blob_rho.pt"))
        print(f"[Save] saved BLoB adapter and rho to: {save_dir}")

    def _maybe_log_progress(self, stage: str, step_idx: int, total_steps: int, extra: str = ""):
        if not self.accelerator.is_local_main_process:
            return
        if total_steps <= 0:
            return
        step_num = step_idx + 1
        should_print = (
            step_idx == 0
            or step_num == total_steps
            or step_num % max(1, min(10, total_steps // 10 or 1)) == 0
        )
        if not should_print:
            return
        pct = 100.0 * step_num / total_steps
        suffix = f"  {extra}" if extra else ""
        print(f"[PROGRESS] {stage}: {step_num}/{total_steps} ({pct:.1f}%){suffix}", flush=True)

    def _mask_num_choices(self, logits: torch.Tensor, num_choices) -> torch.Tensor:
        if num_choices is None:
            return logits
        if not torch.is_tensor(num_choices):
            num_choices = torch.tensor(num_choices, device=logits.device, dtype=torch.long)
        else:
            num_choices = num_choices.to(device=logits.device, dtype=torch.long)

        if logits.dim() == 2:
            col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
            invalid = col_idx >= num_choices.view(-1, 1)
            return logits.masked_fill(invalid, -1e9)

        if logits.dim() == 3:
            col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, 1, -1)
            invalid = col_idx >= num_choices.view(-1, 1, 1)
            return logits.masked_fill(invalid, -1e9)

        return logits

    def _modify_lora_layers(self, module):
        """
        Recursively go through the model and modify LoraLayer instances.
        """
        for name, child in module.named_children():
            if isinstance(child, LoraLayer) and isinstance(child, Linear):
                self._wrap_lora_layer(child)
                # modify existing methods
                setattr(
                    child,
                    "forward",
                    blob_linear_forward.__get__(child, child.__class__),
                )
                # add new methods
                setattr(
                    child,
                    "div_posterior_prior",
                    div_posterior_prior.__get__(child, child.__class__),
                )
                setattr(child, "sample", sample.__get__(child, child.__class__))
            if isinstance(child, LoraLayer) and isinstance(child, Linear8bitLt):
                self._wrap_lora_layer(child)
                # modify existing methods
                setattr(
                    child,
                    "forward",
                    blob_8bitlinear_forward.__get__(child, child.__class__),
                )
                # add new methods
                setattr(
                    child,
                    "div_posterior_prior",
                    div_posterior_prior.__get__(child, child.__class__),
                )
                setattr(child, "sample", sample.__get__(child, child.__class__))
            else:
                self._modify_lora_layers(child)

    def _wrap_lora_layer(self, lora_layer):
        lora_layer.lora_A_rho = nn.ParameterDict({})
        lora_layer.bayes_eps = self.blobconfig.bayes_eps
        lora_layer.bayes_beta = self.blobconfig.bayes_beta
        lora_layer.blobsample = True

        for adapter_name in lora_layer._active_adapter:
            lora_layer.lora_A_rho[adapter_name] = nn.Parameter(
                lora_layer.lora_A[adapter_name].weight.new_zeros(
                    lora_layer.r[adapter_name], lora_layer.in_features
                )
            )

        if adapter_name in lora_layer.lora_A.keys():
            if lora_layer.bayes_eps < 0:
                nn.init.uniform_(
                    lora_layer.lora_A_rho[adapter_name],
                    lora_layer.bayes_eps - 1,
                    lora_layer.bayes_eps,
                )
            else:
                nn.init.uniform_(
                    lora_layer.lora_A_rho[adapter_name],
                    lora_layer.bayes_eps / math.sqrt(2),
                    lora_layer.bayes_eps,
                )

        return

    def div_posterior_prior(self, module):
        kl = 0
        for name, child in module.named_children():
            if isinstance(child, LoraLayer):
                kl_ = child.div_posterior_prior()
                # if not math.isnan(kl_):
                kl += kl_
            else:
                kl += self.div_posterior_prior(child)
        return kl

    def sample(self, module, status=True):
        """
        Set the sampling status of the model.
        """
        for name, child in module.named_children():
            if isinstance(child, LoraLayer):
                child.sample(status)
            else:
                self.sample(child, status)

    def forward_logits(self, batch, sample=True, n_samples=1, **kwargs) -> torch.Tensor:
        if _is_mc_dataset_type(self.args.dataset_type):
            inputs, _, _ = batch
            num_choices = None
            if isinstance(inputs, dict) and "num_choices" in inputs:
                inputs = dict(inputs)
                num_choices = inputs.pop("num_choices")
            if not sample:
                self.sample(self.base_model, False)
                output = self.base_model(**inputs)
                if _uses_trimmed_mc_head(self.args.dataset_type):
                    logits = output.logits[:, -1, :]
                else:
                    logits = output.logits[:, -1, self.target_ids]
                self.sample(self.base_model, True)
                logits = self._mask_num_choices(logits, num_choices)
                return logits.unsqueeze(1)
            else:
                logits_list = []
                for _ in range(n_samples):
                    output = self.base_model(**inputs)
                    if _uses_trimmed_mc_head(self.args.dataset_type):
                        logits = output.logits[:, -1, :]
                    else:
                        logits = output.logits[:, -1, self.target_ids]
                    logits = self._mask_num_choices(logits, num_choices)
                    logits_list.append(logits)
                return torch.stack(logits_list, dim=1)
        else:
            if not sample:
                self.sample(self.base_model, False)
                res = self.base_model(**batch).logits
                self.sample(self.base_model, True)
                return res.unsqueeze(1)
            else:
                res = []
                for _ in range(n_samples):
                    res.append(self.base_model(**batch).logits)
                return torch.stack(res, dim=1)

    def fit(self, train_loader, eval_loader, max_steps: Optional[int] = None):
        nll_losses = AverageMeter()
        kl_losses = AverageMeter()
        elbo_losses = AverageMeter()
        accs = AverageMeter()
        target_steps = int(max_steps if max_steps is not None else self.args.max_train_steps)
        if target_steps <= 0:
            return

        loader_iter = iter(train_loader)
        steps_per_epoch = max(int(len(train_loader)), 1)
        with tqdm(total=target_steps, desc="Total Training Steps", leave=True) as pbar:
            while self.global_step < target_steps:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    batch = next(loader_iter)

                if _is_mc_dataset_type(self.args.dataset_type):
                    _, golds, _ = batch
                elif self.args.dataset_type == "bertds":
                    golds = batch["labels"]
                else:
                    raise NotImplementedError(
                        f"Dataset type {self.args.dataset_type} not implemented."
                    )

                logits = self.forward_logits(
                    batch, sample=True, n_samples=self.train_n_samples
                ).mean(1)
                output = torch.log_softmax(logits, dim=1)
                nll = self.loss(output, golds, reduction="mean")

                self.accelerator.backward(nll)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                kl_divs = []
                for _ in range(self.train_n_samples):
                    if hasattr(self.base_model, "module"):
                        kl_divs.append(self.div_posterior_prior(self.base_model.module))
                    else:
                        kl_divs.append(self.div_posterior_prior(self.base_model))
                kl = torch.mean(torch.stack(kl_divs), dim=0)

                if self.klreweighting:
                    cycle_step = self.M if self.i % self.M == 0 else self.i % self.M
                    self.pi = 2**cycle_step / (2 ** (self.M + 1) - 1)
                    self.i += 1
                else:
                    self.pi = 1 / self.M
                kl_div = kl * self.pi
                self.accelerator.backward(kl_div)
                self.opt2.step()
                self.opt2.zero_grad()
                self.scheduler2.step()

                acc = accuracy_topk(output.data, golds)
                loss, acc, nll_loss, kl_loss = (
                    float((kl + nll).detach().to(dtype=torch.float32).item()),
                    acc.item(),
                    float(nll.detach().to(dtype=torch.float32).item()),
                    float(kl_div.detach().to(dtype=torch.float32).item()),
                )

                if _is_mc_dataset_type(self.args.dataset_type):
                    _, classes, _ = batch
                    references = self.accelerator.gather(classes)
                else:
                    references = self.accelerator.gather(batch["labels"])
                len_batch = int(references.shape[0])
                kl_losses.update(kl_loss, len_batch)
                nll_losses.update(nll_loss, len_batch)
                elbo_losses.update(loss, len_batch)
                accs.update(acc, len_batch)

                assert not math.isnan(nll_loss)
                assert not math.isnan(kl_loss)
                if self.accelerator.is_local_main_process and self.wandb_logger is not None:
                    self.wandb_logger.log(
                        {
                            "train_acc": accs.avg,
                            "train_nll_loss": nll_losses.avg,
                            "kl_loss": kl_losses.avg,
                            "elbo_loss": elbo_losses.avg,
                            "lr": self.opt.param_groups[0]["lr"],
                            "pi": self.pi,
                        }
                    )

                self.global_step += 1
                self.step += 1
                pbar.update(1)

                self._maybe_log_progress(
                    stage="BLOB train",
                    step_idx=self.global_step - 1,
                    total_steps=target_steps,
                    extra=f"nll={float(nll_loss):.4f}",
                )
                if self.args.eval_per_steps > 0 and self.step >= self.args.eval_per_steps:
                    self.step -= self.args.eval_per_steps
                    self.evaluate(eval_loader)

    def evaluate(self, eval_loader):
        if _uses_trimmed_mc_head(self.args.dataset_type) and _CEU is not None:
            sample = not self.args.bayes_inference_notsample
            metrics = self._evaluate_benchmark_common(eval_loader, sample=sample, n_samples=self.eval_n_samples)
            return (
                metrics["acc"],
                metrics["ece"],
                metrics["nll"],
                metrics["brier"],
                metrics["std"],
            )

        print("self.eval_n_samples:", self.eval_n_samples)
        self.eval()
        status = self.training
        nlls = AverageMeter()
        Accuracy, CalibrationError = _get_torchmetrics()
        metric_kwargs = {"task": "multiclass", "num_classes": self.num_classes}
        acc_metric = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric = CalibrationError(**metric_kwargs, n_bins=self.args.num_bins).to(
            self.accelerator.device
        )
        briers = AverageMeter()

        samples_seen = 0
        total_eval_batches = len(eval_loader)
        for step, batch in enumerate(eval_loader):
            with torch.no_grad() and torch.inference_mode():
                logits = self.forward_logits(
                    batch,
                    sample=not self.args.bayes_inference_notsample,
                    n_samples=self.eval_n_samples,
                ).detach()
                if _is_mc_dataset_type(self.args.dataset_type):
                    _, labels, _ = batch
                else:
                    labels = batch["labels"]
                logits, labels = self.accelerator.gather([logits, labels])
                if self.accelerator.num_processes > 1:
                    if step == len(eval_loader) - 1:
                        labels = labels[: len(eval_loader.dataset) - samples_seen]
                        logits = logits[: len(eval_loader.dataset) - samples_seen]
                    else:
                        samples_seen += labels.shape[0]
                probs = torch.softmax(logits, dim=-1).mean(dim=1)
                std = torch.softmax(logits, dim=-1).std(dim=1, unbiased=False).mean()

                acc_metric(probs, labels)
                ece_metric(probs, labels)
                nll = self.loss(torch.log(probs), labels, reduction="mean")
                if torch.isnan(nll):
                    if self.accelerator.is_local_main_process:
                        print("nll:", nll)
                        print("probs:", probs)
                        print("logits:", logits)
                        exit()
                nlls.update(nll)

                brier = (
                    (probs - F.one_hot(labels, num_classes=logits.size(-1)))
                    .pow(2)
                    .sum(dim=-1)
                    .mean()
                )
                briers.update(brier)
                self._maybe_log_progress(
                    stage="BLOB eval",
                    step_idx=step,
                    total_steps=total_eval_batches,
                    extra=f"mc={self.eval_n_samples}",
                )

        val_acc = acc_metric.compute().item()
        val_ece = ece_metric.compute().item()
        val_nll = nlls.avg
        val_brier = briers.avg
        self.train(status)

        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {
                        "val_acc": val_acc,
                        "val_ece": val_ece,
                        "val_nll": val_nll,
                        "std": std,
                        "val_brier": val_brier,
                    }
                )
        if isinstance(std, torch.Tensor):
            std = std.item()
        return val_acc, val_ece, val_nll, val_brier, float(std)

    def _evaluate_benchmark_common(self, eval_loader, sample: bool, n_samples: int) -> Dict[str, float]:
        if _CEU is None:
            raise RuntimeError("common_eval_utils is required for benchmark_mcdataset evaluation.")

        self.eval()
        status = self.training
        acc_metric = _CEU.make_accuracy(self.accelerator.device, self.num_classes)
        ece_metric = _CEU.make_ece(self.accelerator.device, self.num_classes, int(self.args.num_bins))
        acc_metric.reset()
        ece_metric.reset()

        total = 0
        nll_sum = 0.0
        all_probs: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        std_values: List[float] = []
        samples_seen = 0

        for step, batch in enumerate(eval_loader):
            with torch.no_grad(), torch.inference_mode():
                logits_samples = self.forward_logits(
                    batch,
                    sample=bool(sample),
                    n_samples=max(int(n_samples), 1),
                ).detach()
                _, labels, _ = batch
                logits_samples, labels = self.accelerator.gather([logits_samples, labels])
                if self.accelerator.num_processes > 1:
                    if step == len(eval_loader) - 1:
                        keep = len(eval_loader.dataset) - samples_seen
                        labels = labels[:keep]
                        logits_samples = logits_samples[:keep]
                    else:
                        samples_seen += labels.shape[0]

                bsz = int(labels.size(0))
                total += bsz
                if sample:
                    probs = torch.softmax(logits_samples, dim=-1).mean(dim=1)
                    std_values.append(
                        float(
                            torch.softmax(logits_samples, dim=-1)
                            .std(dim=1, unbiased=False)
                            .mean()
                            .item()
                        )
                    )
                    idx = torch.arange(bsz, device=labels.device)
                    nll_sum += float((-torch.log(probs[idx, labels].clamp_min(1e-12))).sum().item())
                else:
                    logits = logits_samples[:, 0, :]
                    probs = torch.softmax(logits, dim=-1)
                    std_values.append(0.0)
                    nll_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())

                acc_metric.update(probs, labels)
                ece_metric.update(probs, labels)
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())

        probs_all = (
            torch.cat(all_probs, dim=0)
            if all_probs
            else torch.empty((0, self.num_classes), dtype=torch.float32)
        )
        labels_all = (
            torch.cat(all_labels, dim=0)
            if all_labels
            else torch.empty((0,), dtype=torch.long)
        )
        self.train(status)
        return {
            "nll": nll_sum / max(total, 1),
            "acc": float(acc_metric.compute().item()),
            "ece": float(ece_metric.compute().item()),
            "brier": _multiclass_brier_score(probs_all, labels_all) if total > 0 else float("nan"),
            "std": float(sum(std_values) / max(len(std_values), 1)),
        }

    def fit_evaluate(self):
        if self.accelerator.is_local_main_process:
            save_folder = f"checkpoints/{self.args.modelwrapper}/{self.args.model}/{self.args.dataset}/{self.args.log_path}"
            create_if_not_exists(save_folder)
            logging.basicConfig(
                format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                level=logging.INFO,
                filename=save_folder + "/log.txt",
            )

        dataset_obj = getattr(self, "dataset_obj", None)
        source_task = str(getattr(dataset_obj, "source_task", self.args.dataset))
        eval_task_name = str(getattr(dataset_obj, "eval_task_name", source_task))
        eval_split_name = str(getattr(dataset_obj, "eval_split_name", "validation"))
        eval_tag = f"{eval_task_name}({eval_split_name})"

        effective_max_steps = int(self.args.max_train_steps)
        if self.args.early_stop_steps > 0:
            effective_max_steps = min(effective_max_steps, int(self.args.early_stop_steps))

        with StageTimer(f"FIT BLOB on {source_task}(train)"):
            self.fit(self.train_loader, self.test_loader, max_steps=effective_max_steps)

        if effective_max_steps > 0 and getattr(self.args, "save_blob_dir", None):
            with StageTimer(f"SAVE BLoB to {self.args.save_blob_dir}"):
                self._save_blob_checkpoint(self.args.save_blob_dir)

        final_eval_n = int(getattr(self.args, "bayes_eval_n_samples_final", self.eval_n_samples))
        self.eval_n_samples = final_eval_n

        if _uses_trimmed_mc_head(self.args.dataset_type) and _CEU is not None:
            eval_loaders = getattr(self, "eval_loaders", None) or {eval_task_name: self.test_loader}
            eval_split_by_task = getattr(self, "eval_split_by_task", None) or {eval_task_name: eval_split_name}

            for task_name, loader in eval_loaders.items():
                split_name = str(eval_split_by_task.get(task_name, "ood"))
                task_eval_tag = f"{task_name}({split_name})"
                task_key = re.sub(r"[^0-9A-Za-z_]+", "_", str(task_name)).strip("_") or "eval"

                with StageTimer(f"INFER BLoB mean on {task_eval_tag}"):
                    m_mean = self._evaluate_benchmark_common(loader, sample=False, n_samples=1)
                with StageTimer(f"INFER BLoB samp on {task_eval_tag}"):
                    m_samp = self._evaluate_benchmark_common(loader, sample=True, n_samples=final_eval_n)

                print(f"\n[{task_eval_tag} Results]")
                print(
                    f"  BLoB mean  : NLL={m_mean['nll']:.4f}  ACC={m_mean['acc']*100:.2f}%  "
                    f"ECE={m_mean['ece']*100:.2f}%  Brier={m_mean['brier']:.4f} (N=0)"
                )
                print(
                    f"  BLoB samp  : NLL={m_samp['nll']:.4f}  ACC={m_samp['acc']*100:.2f}%  "
                    f"ECE={m_samp['ece']*100:.2f}%  Brier={m_samp['brier']:.4f} (N={final_eval_n})"
                )

                logging.info(
                    f"{task_key}.blob_mean: "
                    f"acc={m_mean['acc']}, ece={m_mean['ece']}, nll={m_mean['nll']}, brier={m_mean['brier']}"
                )
                logging.info(
                    f"{task_key}.blob_samp: "
                    f"acc={m_samp['acc']}, ece={m_samp['ece']}, nll={m_samp['nll']}, "
                    f"brier={m_samp['brier']}, std={m_samp['std']}, mc={final_eval_n}"
                )
                if self.accelerator.is_local_main_process and self.wandb_logger is not None:
                    payload = {
                        f"final_blob_mean_acc/{task_key}": m_mean["acc"],
                        f"final_blob_mean_ece/{task_key}": m_mean["ece"],
                        f"final_blob_mean_nll/{task_key}": m_mean["nll"],
                        f"final_blob_mean_brier/{task_key}": m_mean["brier"],
                        f"final_blob_samp_acc/{task_key}": m_samp["acc"],
                        f"final_blob_samp_ece/{task_key}": m_samp["ece"],
                        f"final_blob_samp_nll/{task_key}": m_samp["nll"],
                        f"final_blob_samp_brier/{task_key}": m_samp["brier"],
                        f"final_blob_samp_std/{task_key}": m_samp["std"],
                    }
                    if task_name == eval_task_name:
                        payload.update(
                            {
                                "final_blob_mean_acc": m_mean["acc"],
                                "final_blob_mean_ece": m_mean["ece"],
                                "final_blob_mean_nll": m_mean["nll"],
                                "final_blob_mean_brier": m_mean["brier"],
                                "final_blob_samp_acc": m_samp["acc"],
                                "final_blob_samp_ece": m_samp["ece"],
                                "final_blob_samp_nll": m_samp["nll"],
                                "final_blob_samp_brier": m_samp["brier"],
                                "final_blob_samp_std": m_samp["std"],
                            }
                        )
                    self.wandb_logger.log(payload)
            return

        with StageTimer(f"INFER BLOB on {eval_tag}"):
            val_acc, val_ece, val_nll, val_brier, std = self.evaluate(self.test_loader)

        print(f"\n[{eval_tag}][BLOB]")
        print(
            f"  NLL={val_nll:.4f}  ACC={val_acc*100:.2f}%  "
            f"ECE={val_ece*100:.2f}%  Brier={val_brier:.4f}  "
            f"mc={int(self.eval_n_samples)}  std={float(std):.6f}"
        )

        logging.info(
            f"val_acc: {val_acc}, val_ece: {val_ece}, val_nll: {val_nll}, val_brier: {val_brier}, std: {std}"
        )
        if self.accelerator.is_local_main_process and self.wandb_logger is not None:
            self.wandb_logger.log(
                {
                    "final_val_acc": val_acc,
                    "final_val_ece": val_ece,
                    "final_val_nll": val_nll,
                    "final_val_brier": val_brier,
                    "final_std": std,
                }
            )

    def prepare_for_fit_evaluate(self, dataset, wandb_logger=None):
        """
        Prepare the model for training and evaluation.
        """
        self.wandb_logger = wandb_logger
        self.dataset_obj = dataset
        train_loader, test_loader = dataset.train_dataloader, dataset.test_dataloader
        raw_eval_loaders = dict(getattr(dataset, "eval_loaders", {}) or {})
        raw_eval_splits = dict(getattr(dataset, "eval_split_name_by_task", {}) or {})
        source_task = str(getattr(dataset, "source_task", self.args.dataset))

        if _is_mc_dataset_type(self.args.dataset_type):
            self.tokenizer = dataset.tokenizer
            self.target_ids = dataset.target_ids.squeeze(-1)

        l_train = len(train_loader)

        num_update_steps_per_epoch = len(train_loader)
        if self.args.max_train_steps == 0:
            self.args.max_train_steps = self.args.n_epochs * num_update_steps_per_epoch
        self.args.n_epochs = (
            math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
            if self.args.ood_ori_dataset is None
            else 0
        )
        if self.args.early_stop_steps > 0:
            self.earlystop_n_epochs = (
                math.ceil(self.args.early_stop_steps / num_update_steps_per_epoch)
                if self.args.ood_ori_dataset is None
                else 0
            )
        else:
            self.earlystop_n_epochs = 0
        if self.accelerator.is_local_main_process:
            print("len(train_loader):", len(train_loader))
            print("max train steps:", self.args.max_train_steps)
        self.step = 0
        self.global_step = 0

        (
            self.base_model,
            self.opt,
            train_loader,
            test_loader,
            self.scheduler,
            self.scheduler2,
            self.opt2,
        ) = self.accelerator.prepare(
            self.base_model,
            self.opt,
            train_loader,
            test_loader,
            self.scheduler,
            self.scheduler2,
            self.opt2,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.eval_loaders = {}
        self.eval_split_by_task = {}
        if raw_eval_loaders:
            if source_task in raw_eval_loaders:
                self.eval_loaders[source_task] = self.test_loader
                self.eval_split_by_task[source_task] = str(
                    raw_eval_splits.get(
                        source_task,
                        getattr(dataset, "source_eval_split_name", getattr(dataset, "eval_split_name", "validation")),
                    )
                )

            extra_eval_items = [
                (task_name, loader)
                for task_name, loader in raw_eval_loaders.items()
                if task_name != source_task
            ]
            if extra_eval_items:
                prepared = self.accelerator.prepare(*[loader for _, loader in extra_eval_items])
                if len(extra_eval_items) == 1:
                    prepared = (prepared,)
                for (task_name, _), prepared_loader in zip(extra_eval_items, prepared):
                    self.eval_loaders[task_name] = prepared_loader
                    self.eval_split_by_task[task_name] = str(raw_eval_splits.get(task_name, "ood"))
        else:
            eval_task_name = str(getattr(dataset, "eval_task_name", source_task))
            self.eval_loaders[eval_task_name] = self.test_loader
            self.eval_split_by_task[eval_task_name] = str(
                getattr(dataset, "eval_split_name", getattr(dataset, "source_eval_split_name", "validation"))
            )

        if self.args.bayes_datasetrescaling:
            self.M = int(
                100
                * (dataset.num_samples ** (math.pi / self.args.bayes_gamma))
                / (l_train / len(train_loader))
                / self.args.batch_size
            )
        else:
            self.M = len(train_loader)

        print("M:", self.M)
