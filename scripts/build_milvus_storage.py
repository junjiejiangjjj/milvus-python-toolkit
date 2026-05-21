from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/milvus-io/milvus-storage.git"
CONAN_REMOTE_NAME = "default-conan-local2"
CONAN_REMOTE_URL = "https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2"
ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
CHECKOUT = DEPS / "milvus-storage"
WHEELHOUSE = DEPS / "wheelhouse"


def main() -> int:
    args = _parse_args()
    ref = args.ref or os.environ.get("MILVUS_STORAGE_REF", "main")
    checkout = args.checkout.resolve() if args.checkout else CHECKOUT
    wheelhouse = args.wheelhouse.resolve() if args.wheelhouse else WHEELHOUSE

    if args.clean and checkout.exists():
        shutil.rmtree(checkout)

    _ensure_checkout(checkout, ref)
    _ensure_conan_setup()
    _run(["make", "-C", str(checkout / "cpp"), "python-lib"])
    native_lib = _find_native_library(checkout)
    package_lib_dir = checkout / "python" / "milvus_storage" / "lib"
    package_lib_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(native_lib, package_lib_dir / native_lib.name)

    wheelhouse.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "pip", "install", "build>=1"])
    _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheelhouse)],
        cwd=checkout / "python",
    )
    wheel = _latest_wheel(wheelhouse)

    if not args.no_auditwheel and platform.system() == "Linux":
        repaired = _repair_with_auditwheel(wheel, wheelhouse)
        if repaired is not None:
            wheel = repaired

    print(wheel)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build milvus-storage from upstream Git source")
    parser.add_argument("--ref", help="Git ref to checkout; defaults to MILVUS_STORAGE_REF or main")
    parser.add_argument("--checkout", type=Path, help="Local milvus-storage checkout path")
    parser.add_argument("--wheelhouse", type=Path, help="Directory for built wheels")
    parser.add_argument("--clean", action="store_true", help="Delete the checkout before cloning")
    parser.add_argument(
        "--no-auditwheel",
        action="store_true",
        help="Skip auditwheel repair on Linux",
    )
    return parser.parse_args()


def _ensure_checkout(checkout: Path, ref: str) -> None:
    checkout.parent.mkdir(parents=True, exist_ok=True)
    if checkout.exists():
        _run(["git", "fetch", "--all", "--tags"], cwd=checkout)
    else:
        _run(["git", "clone", REPO_URL, str(checkout)])
    _run(["git", "checkout", ref], cwd=checkout)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=checkout)


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


def _latest_wheel(wheelhouse: Path) -> Path:
    wheels = sorted(
        wheelhouse.glob("milvus_storage-*.whl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not wheels:
        raise RuntimeError(f"No milvus-storage wheel found in {wheelhouse}")
    return wheels[0]


def _repair_with_auditwheel(wheel: Path, wheelhouse: Path) -> Path | None:
    if shutil.which("auditwheel") is None:
        print("auditwheel not found; keeping unrepaired local development wheel", file=sys.stderr)
        return None
    _run([sys.executable, "-m", "auditwheel", "repair", str(wheel), "-w", str(wheelhouse)])
    return _latest_wheel(wheelhouse)


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
