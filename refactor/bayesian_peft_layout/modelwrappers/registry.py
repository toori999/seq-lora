from .blob import BlobEvalWrapper
from .deepensemble import DeepEnsembleEvalWrapper
from .laplace import LaplaceEvalWrapper
from .map import MapEvalWrapper
from .mcdrop import MCDropEvalWrapper
from .probensemble import ProbabilityEnsembleEvalWrapper
from .seqconstantq import SeqConstantQEvalWrapper

WRAPPER_REGISTRY = {
    MapEvalWrapper.method_name: MapEvalWrapper,
    MCDropEvalWrapper.method_name: MCDropEvalWrapper,
    DeepEnsembleEvalWrapper.method_name: DeepEnsembleEvalWrapper,
    ProbabilityEnsembleEvalWrapper.method_name: ProbabilityEnsembleEvalWrapper,
    SeqConstantQEvalWrapper.method_name: SeqConstantQEvalWrapper,
    LaplaceEvalWrapper.method_name: LaplaceEvalWrapper,
    BlobEvalWrapper.method_name: BlobEvalWrapper,
}


def get_wrapper_cls(method_name: str):
    if method_name not in WRAPPER_REGISTRY:
        raise KeyError(f"Unknown wrapper method: {method_name}")
    return WRAPPER_REGISTRY[method_name]


__all__ = ["WRAPPER_REGISTRY", "get_wrapper_cls"]
