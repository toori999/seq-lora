from refactor.seq_lora.eval import run_mc_dropout_evaluation

from .base import EvalWrapperBase


class MCDropEvalWrapper(EvalWrapperBase):
    method_name = "mcdrop"

    def run(self) -> None:
        run_mc_dropout_evaluation(
            task=self.args.task,
            adapter_dir=self.args.adapter_dir,
            eval_task_spec=self.args.eval_tasks,
            max_seq_len=self.args.max_seq_len,
            eval_bsz=self.args.eval_bsz,
            seed=self.args.seed,
            mc_samples=self.args.mc_samples,
            temp=self.args.temp,
            trust_remote_code=self.args.trust_remote_code,
        )
