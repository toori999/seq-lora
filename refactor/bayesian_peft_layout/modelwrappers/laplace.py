from __future__ import annotations

from .base import EvalWrapperBase


class LaplaceEvalWrapper(EvalWrapperBase):
    method_name = "laplace"

    def run(self) -> None:
        cli_args = [
            "--task_name",
            self.args.task,
            "--map_adapter_dir",
            self.args.adapter_dir,
            "--output_dir",
            self.args.output_dir,
            "--eval_tasks",
            self.args.eval_tasks,
            "--max_length",
            str(self.args.max_seq_len),
            "--per_device_eval_batch_size",
            str(self.args.eval_bsz),
            "--fit_bsz",
            str(self.args.fit_bsz),
            "--laplace_bsz",
            str(self.args.laplace_bsz),
            "--prior_optim_step",
            str(self.args.prior_optim_step),
            "--laplace_mc_samples",
            str(self.args.laplace_mc_samples),
            "--laplace_mc_chunk",
            str(self.args.laplace_mc_chunk),
            "--testing_set",
            self.args.testing_set,
            "--seed",
            str(self.args.seed),
        ]
        if self.args.model_name_or_path:
            cli_args.extend(["--model_name_or_path", self.args.model_name_or_path])
        if self.args.force_refit:
            cli_args.append("--force_refit")
        if self.args.force_reprior:
            cli_args.append("--force_reprior")
        self.run_legacy_script("laplace_lora_official_source_eval.py", cli_args)
