class MilvusToolkitError(Exception):
    """Base exception for Milvus Toolkit errors."""


class ConfigError(MilvusToolkitError):
    pass


class SnapshotError(MilvusToolkitError):
    pass


class SchemaError(MilvusToolkitError):
    pass


class ManifestError(MilvusToolkitError):
    pass


class StorageError(MilvusToolkitError):
    pass


class UnsupportedStorageError(MilvusToolkitError):
    pass


class UnsupportedSegmentError(MilvusToolkitError):
    pass


class UnsupportedFeatureError(MilvusToolkitError):
    pass


class EngineError(MilvusToolkitError):
    pass
