from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path

CONAN_REMOTE_NAME = "default-conan-local2"
CONAN_REMOTE_URL = "https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2"
ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
CHECKOUT = THIRD_PARTY / "milvus-storage"


def main() -> int:
    _parse_args()
    _ensure_submodule_checkout(CHECKOUT)
    _ensure_conan_setup()
    _run(["make", "-C", str(CHECKOUT / "cpp"), "python-lib"])
    native_lib = _find_native_library(CHECKOUT)
    package_lib_dir = CHECKOUT / "python" / "milvus_storage" / "lib"
    package_lib_dir.mkdir(parents=True, exist_ok=True)
    staged_lib = package_lib_dir / native_lib.name
    shutil.copy2(native_lib, staged_lib)

    print(staged_lib)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and stage milvus-storage native library for milvus-toolkit"
    )
    return parser.parse_args()


def _ensure_submodule_checkout(checkout: Path) -> None:
    missing_paths = [
        checkout,
        checkout / "cpp",
        checkout / "python" / "pyproject.toml",
    ]
    missing = [path for path in missing_paths if not path.exists()]
    if missing:
        formatted = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise RuntimeError(
            "milvus-storage submodule is not initialized correctly; missing "
            f"{formatted}. Run git submodule update --init --recursive."
        )


def _ensure_conan_setup() -> None:
    _run(["conan", "profile", "detect", "--force"])
    result = subprocess.run(
        ["conan", "remote", "list"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if CONAN_REMOTE_NAME not in result.stdout:
        _run(["conan", "remote", "add", CONAN_REMOTE_NAME, CONAN_REMOTE_URL])


def _find_native_library(checkout: Path) -> Path:
    names = _native_library_names()
    candidates = [
        path
        for name in names
        for path in (checkout / "cpp" / "build").rglob(name)
        if path.is_file()
    ]
    if not candidates:
        raise RuntimeError(
            "Could not find built milvus-storage native library under cpp/build after "
            "make -C cpp python-lib"
        )
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _native_library_names() -> tuple[str, ...]:
    system = platform.system()
    if system == "Darwin":
        return ("libmilvus-storage.dylib",)
    if system == "Windows":
        return ("milvus-storage.dll", "libmilvus-storage.dll")
    return ("libmilvus-storage.so",)


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
