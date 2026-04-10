import torch
import torch.nn as nn
from torch.optim import SGD
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
import math
import logging
from tqdm import tqdm
from pathlib import Path

from .wrapperbase import WrapperBase, get_linear_schedule_with_warmup
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


def _is_mc_dataset_type(dataset_type: str) -> bool:
    return str(dataset_type).strip().lower() in {"mcdataset", "benchmark_mcdataset"}


def _uses_trimmed_mc_head(dataset_type: str) -> bool:
    return str(dataset_type).strip().lower() == "benchmark_mcdataset"


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
        self._modify_lora_layers(self.base_model)
        if args.load_lora_path is not None:
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

    def fit(self, train_loader, eval_loader):
        nll_losses = AverageMeter()
        kl_losses = AverageMeter()
        elbo_losses = AverageMeter()
        accs = AverageMeter()
        samples_seen = 0
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}",
            leave=False,
        ) as pbar:
            total_train_batches = len(train_loader)
            for i, batch in enumerate(train_loader):
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
                    if self.i % self.M == 0:
                        i = self.M
                    else:
                        i = self.i % self.M
                    self.pi = 2**i / (2 ** (self.M + 1) - 1)
                    self.i += 1
                else:
                    self.pi = 1 / self.M
                kl_div = kl * self.pi
                self.accelerator.backward(kl_div)
                self.opt2.step()
                self.opt2.zero_grad()
                self.scheduler2.step()

                acc = accuracy_topk(output.data, golds)

                loss, acc, nll_loss, kl = (
                    (kl + nll).detach().cpu().numpy(),
                    acc.item(),
                    nll.detach().cpu().numpy(),
                    kl_div.detach().cpu().numpy(),
                )

                if _is_mc_dataset_type(self.args.dataset_type):
                    _, classes, _ = batch
                    references = self.accelerator.gather(classes)
                else:
                    references = self.accelerator.gather(batch["labels"])
                if self.accelerator.num_processes > 1:
                    if i == len(train_loader) - 1:
                        references = references[
                            : len(train_loader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += references.shape[0]
                len_batch = references.shape[0]
                kl_losses.update(kl, len_batch)
                nll_losses.update(nll_loss, len_batch)
                elbo_losses.update(loss, len_batch)
                accs.update(acc, len_batch)

                assert not math.isnan(nll_loss)
                assert not math.isnan(kl)
                if self.accelerator.is_local_main_process:
                    if self.wandb_logger is not None:
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

                self.step += self.accelerator.num_processes
                pbar.update(1)
                self._maybe_log_progress(
                    stage=f"BLOB train epoch {self.args.epoch + 1}/{self.args.n_epochs}",
                    step_idx=i,
                    total_steps=total_train_batches,
                    extra=f"nll={float(nll_loss):.4f}",
                )
                if self.step >= self.args.eval_per_steps:
                    self.step -= self.args.eval_per_steps
                    self.evaluate(eval_loader)

    def evaluate(self, eval_loader):
        print("self.eval_n_samples:", self.eval_n_samples)
        self.eval()
        status = self.training
        nlls = AverageMeter()
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
                std = torch.softmax(logits, dim=-1).std(dim=1).mean()

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

        with StageTimer(f"FIT BLOB on {source_task}(train)"):
            with tqdm(total=self.args.n_epochs, desc="Total Training Epochs", leave=True) as pbar:
                for epoch in range(self.args.n_epochs):
                    if self.args.early_stop_steps > 0 and epoch >= self.earlystop_n_epochs:
                        break
                    self.args.epoch = epoch
                    self.fit(self.train_loader, self.test_loader)
                    pbar.update(1)

        if hasattr(self.args, "bayes_eval_n_samples_final"):
            self.eval_n_samples = self.args.bayes_eval_n_samples_final

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
            print("num of epochs:", self.args.n_epochs)
        self.step = 0

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
