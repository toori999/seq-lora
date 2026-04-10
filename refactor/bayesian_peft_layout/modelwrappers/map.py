from refactor.seq_lora.eval import run_map_evaluation

from .base import EvalWrapperBase


class MapEvalWrapper(EvalWrapperBase):
    method_name = "map"

    def run(self) -> None:
        run_map_evaluation(
            task=self.args.task,
            adapter_dir=self.args.adapter_dir,
            eval_task_spec=self.args.eval_tasks,
            max_seq_len=self.args.max_seq_len,
            eval_bsz=self.args.eval_bsz,
            seed=self.args.seed,
            trust_remote_code=self.args.trust_remote_code,
        )
