import logging

from laplace.curvature.curvature import CurvatureInterface, GGNInterface, EFInterface

_ASDL_IMPORT_ERROR = None

try:
    from laplace.curvature.asdl import AsdlHessian, AsdlGGN, AsdlEF, AsdlInterface
except Exception as exc:
    _ASDL_IMPORT_ERROR = exc
    AsdlInterface = None
    AsdlGGN = None
    AsdlEF = None
    AsdlHessian = None
    logging.info('asdfghjkl/asdl backend not available: %s', exc)

__all__ = ['CurvatureInterface', 'GGNInterface', 'EFInterface',
           'AsdlInterface', 'AsdlGGN', 'AsdlEF', 'AsdlHessian']


def __getattr__(name):
    if name in {'AsdlInterface', 'AsdlGGN', 'AsdlEF', 'AsdlHessian'} and globals().get(name) is None:
        raise ImportError(
            "ASDL backend is unavailable. Install the dependency package "
            "`asdfghjkl==0.1a4` in this environment."
        ) from _ASDL_IMPORT_ERROR
    raise AttributeError(name)
