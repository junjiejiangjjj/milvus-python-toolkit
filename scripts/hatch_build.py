from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ROOT = Path(__file__).resolve().parents[1]


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        del build_data
        if self.target_name != "wheel" or version == "editable":
            return
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_milvus_storage.py")],
            check=True,
        )
