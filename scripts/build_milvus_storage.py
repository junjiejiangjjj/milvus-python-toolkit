from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

CONAN_REMOTE_NAME = "default-conan-local2"
CONAN_REMOTE_URL = "https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2"
ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
CHECKOUT = THIRD_PARTY / "milvus-storage"
VENDOR_PACKAGE = ROOT / "ray_milvus" / "_vendor" / "milvus_storage"


def main() -> int:
    _parse_args()
    _ensure_submodule_checkout(CHECKOUT)
    _ensure_conan_setup()
    shim_dir = _stage_conan_shim()
    make_env = {**os.environ, "PATH": f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    _run(["make", "-C", str(CHECKOUT / "cpp"), "python-lib"], env=make_env)
    native_lib = _find_native_library(CHECKOUT)
    source_package = CHECKOUT / "python" / "milvus_storage"
    package_lib_dir = source_package / "lib"
    package_lib_dir.mkdir(parents=True, exist_ok=True)
    staged_lib = package_lib_dir / native_lib.name
    shutil.copy2(native_lib, staged_lib)
    _copy_native_dependencies(native_lib.parent / "libs", package_lib_dir)
    _sync_vendor_package(source_package, VENDOR_PACKAGE)

    print(VENDOR_PACKAGE)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and stage milvus-storage native library for ray-milvus"
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
    _run(_conan_command("profile", "detect", "--force"))
    result = subprocess.run(
        _conan_command("remote", "list"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if CONAN_REMOTE_NAME not in result.stdout:
        _run(_conan_command("remote", "add", CONAN_REMOTE_NAME, CONAN_REMOTE_URL))


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


def _copy_native_dependencies(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    for path in source.iterdir():
        if path.is_dir() and path.name == "ossl-modules":
            target = destination / path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target, symlinks=True)
        elif path.is_file() or path.is_symlink():
            shutil.copy2(path, destination / path.name, follow_symlinks=False)


def _sync_vendor_package(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _stage_conan_shim() -> Path:
    shim_dir = ROOT / ".build" / "bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "conan"
    shim.write_text(
        "#!/usr/bin/env sh\n"
        f"exec {sys.executable!s} -c "
        "'from conan.cli.cli import main; import sys; "
        "raise SystemExit(main(sys.argv[1:]))' \"$@\"\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim_dir


def _conan_command(*args: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        "from conan.cli.cli import main; import sys; raise SystemExit(main(sys.argv[1:]))",
        *args,
    ]


def _run(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
