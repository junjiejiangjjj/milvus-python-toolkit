import milvus_toolkit as mt


def test_public_api_exports_mvp_symbols():
    storage = mt.StorageConfig(endpoint="localhost:9000", bucket="bucket")

    assert storage.endpoint == "localhost:9000"
    assert mt.read_snapshot is not None
    assert mt.inspect_snapshot is not None
    assert issubclass(mt.UnsupportedSegmentError, mt.MilvusToolkitError)
