from refactor.seq_lora.eval import run_probability_ensemble_evaluation

from .base import EvalWrapperBase


class ProbabilityEnsembleEvalWrapper(EvalWrapperBase):
    method_name = "prob-ensemble"

    def run(self) -> None:
        run_probability_ensemble_evaluation(
            task=self.args.task,
            adapter_dirs=self.split_csv(self.args.adapter_dirs),
            eval_task_spec=self.args.eval_tasks,
            max_seq_len=self.args.max_seq_len,
            eval_bsz=self.args.eval_bsz,
            seed=self.args.seed,
            trust_remote_code=self.args.trust_remote_code,
        )
