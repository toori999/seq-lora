from refactor.seq_lora.eval import run_deep_ensemble_evaluation

from .base import EvalWrapperBase


class DeepEnsembleEvalWrapper(EvalWrapperBase):
    method_name = "deep-ensemble"

    def run(self) -> None:
        run_deep_ensemble_evaluation(
            task=self.args.task,
            adapter_dirs=self.split_csv(self.args.adapter_dirs),
            eval_task_spec=self.args.eval_tasks,
            max_seq_len=self.args.max_seq_len,
            eval_bsz=self.args.eval_bsz,
            seed=self.args.seed,
            temp=self.args.temp,
            trust_remote_code=self.args.trust_remote_code,
        )
