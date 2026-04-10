from __future__ import annotations

from refactor.bayesian_peft_layout.modelwrappers.registry import get_wrapper_cls
from refactor.bayesian_peft_layout.utils.args import build_parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    wrapper_cls = get_wrapper_cls(args.method)
    wrapper = wrapper_cls(args)
    wrapper.run()


if __name__ == "__main__":
    main()
