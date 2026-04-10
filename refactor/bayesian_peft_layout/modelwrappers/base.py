from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence
import subprocess
import sys


class EvalWrapperBase(ABC):
    method_name = ""

    def __init__(self, args):
        self.args = args

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    @staticmethod
    def split_csv(spec: str) -> list[str]:
        return [part.strip() for part in spec.split(",") if part.strip()]

    @staticmethod
    def repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def run_legacy_script(self, script_name: str, cli_args: Sequence[str]) -> None:
        cmd = [sys.executable, script_name, *cli_args]
        subprocess.run(cmd, cwd=self.repo_root(), check=True)
