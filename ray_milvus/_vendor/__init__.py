from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path

_PRELOADED: list[ctypes.CDLL] = []


def preload_milvus_storage_libraries() -> None:
    lib_dir = Path(__file__).with_name("milvus_storage") / "lib"
    if not lib_dir.exists():
        return

    libaio = ctypes.util.find_library("aio")
    if libaio is not None:
        _PRELOADED.append(ctypes.CDLL(libaio, mode=ctypes.RTLD_GLOBAL))

    for name in _MILVUS_STORAGE_PRELOAD_ORDER:
        path = lib_dir / name
        if path.exists():
            _PRELOADED.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))


_MILVUS_STORAGE_PRELOAD_ORDER = (
    "libcrypto.so.3",
    "libssl.so.3",
    "libaws-c-common.so.1",
    "libaws-c-cal.so.1.0.0",
    "libaws-c-io.so.1.0.0",
    "libaws-c-compression.so.1.0.0",
    "libaws-c-http.so.1.0.0",
    "libaws-c-sdkutils.so.1.0.0",
    "libaws-c-auth.so.1.0.0",
    "libaws-c-event-stream.so.1.0.0",
    "libaws-c-mqtt.so.1.0.0",
    "libaws-c-s3.so.0unstable",
    "libgflags_nothreads.so.2.2",
)


preload_milvus_storage_libraries()
