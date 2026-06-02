import os
from pathlib import Path

import pytest


def test_milvus_storage_import_exposes_reader_api():
    if os.environ.get("MILVUS_STORAGE_IMPORT_SMOKE") != "1":
        pytest.skip("set MILVUS_STORAGE_IMPORT_SMOKE=1 after scripts/install_dev.sh")

    milvus_storage = pytest.importorskip("ray_milvus._vendor.milvus_storage")

    assert hasattr(milvus_storage, "Reader")
    assert hasattr(milvus_storage, "Transaction")
    assert hasattr(milvus_storage, "ColumnGroups")

    package_dir = Path(milvus_storage.__file__).parent
    native_libs = list((package_dir / "lib").glob("libmilvus-storage.*"))
    assert native_libs
