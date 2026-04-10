from __future__ import annotations

from .base import EvalWrapperBase


class SeqConstantQEvalWrapper(EvalWrapperBase):
    method_name = "seq-constantq"

    def run(self) -> None:
        cli_args = [
            "--task",
            self.args.task,
            "--map_dir",
            self.args.map_dir,
            "--eval_tasks",
            self.args.eval_tasks,
            "--constant_q_var",
            str(self.args.constant_q_var),
            "--forecast_horizon",
            str(self.args.forecast_horizon),
        ]
        if self.args.slice_dir:
            cli_args.extend(["--slices_dir", self.args.slice_dir])
        self.run_legacy_script("seq_eval_iid_constantq.py", cli_args)
