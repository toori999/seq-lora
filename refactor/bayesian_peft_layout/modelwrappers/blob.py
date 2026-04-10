from __future__ import annotations

from .base import EvalWrapperBase


class BlobEvalWrapper(EvalWrapperBase):
    method_name = "blob"

    def run(self) -> None:
        cli_args = [
            "--task",
            self.args.task,
            "--base_model",
            self.args.base_model,
            "--max_seq_len",
            str(self.args.max_seq_len),
            "--eval_bsz",
            str(self.args.eval_bsz),
            "--seed",
            str(self.args.seed),
            "--blob_eval_n",
            str(self.args.blob_eval_n),
            "--eval_tasks",
            self.args.eval_tasks,
        ]
        if self.args.map_dir:
            cli_args.extend(["--map_adapter_dir", self.args.map_dir])
        if self.args.shared_init_lora_path:
            cli_args.extend(["--shared_init_lora_path", self.args.shared_init_lora_path])
        if self.args.save_blob_dir:
            cli_args.extend(["--save_blob_dir", self.args.save_blob_dir])
        if self.args.load_blob_dir:
            cli_args.extend(["--load_blob_dir", self.args.load_blob_dir])
        if self.args.do_train:
            cli_args.append("--do_train")
        if self.args.do_eval:
            cli_args.append("--do_eval")
        self.run_legacy_script("blob_eval_iid_official.py", cli_args)
