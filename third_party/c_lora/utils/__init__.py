# Utility functions for the project.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from contextlib import ContextDecorator
try:
    import ipdb
except Exception:
    ipdb = None


def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    Args:
        -path: the complete path of the folder to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def valid_loss(f):
    """
    Decorator function that prevents training to nan loss value.
    """
    def decorated_f(*args, **kwargs):
        loss = f(*args, **kwargs)
        if torch.isnan(loss):
            if ipdb is not None:
                ipdb.set_trace()
            loss = f(*args, **kwargs)
        return loss
    return decorated_f

def timer(func):
    """
    Decorator function that prints the execution time of the decorated function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")
        return result
    return wrapper


def is_module_differentiable(module):
    """
    Checks if a module is differentiable.
    """
    parameters = list(module.parameters())
    buffers = list(module.buffers())

    if len(parameters) == 0 and len(buffers) == 0:
        # If the module has no parameters or buffers, it is not differentiable
        return False

    # Check if all the parameters and buffers are differentiable
    for param in parameters + buffers:
        if not param.requires_grad:
            return False

    return True


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mem_gb(x: int) -> float:
    return float(x) / (1024 ** 3)


def reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def peak_alloc_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return _mem_gb(torch.cuda.max_memory_allocated())


def peak_reserved_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return _mem_gb(torch.cuda.max_memory_reserved())


class StageTimer(ContextDecorator):
    def __init__(self, tag: str):
        self.tag = tag
        self.t0 = None

    def __enter__(self):
        reset_cuda_peak()
        cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            cuda_sync()
            dt = time.perf_counter() - self.t0
            print(f"[TIME] {self.tag}: {dt:.2f} sec ({dt/60:.2f} min)")
            print(f"[PEAK] {self.tag}: alloc={peak_alloc_gb():.2f} GB  reserved={peak_reserved_gb():.2f} GB")
        else:
            try:
                dt = time.perf_counter() - self.t0
                print(f"[TIME] {self.tag}: {dt:.2f} sec ({dt/60:.2f} min)")
                print(f"[PEAK] {self.tag}: alloc={peak_alloc_gb():.2f} GB  reserved={peak_reserved_gb():.2f} GB")
            except Exception:
                pass
        return False
