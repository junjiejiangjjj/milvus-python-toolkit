from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

import pyarrow.parquet as pq

import ray_milvus as mt
from ray_milvus.errors import MilvusToolkitError


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    if args.command == "inspect":
        return _inspect(args)
    if args.command == "create-snapshot":
        return _create_snapshot(args)
    if args.command == "import-milvus-snapshot":
        return _import_milvus_snapshot(args)
    if args.command == "import-native-milvus-snapshot":
        return _import_native_milvus_snapshot(args)
    if args.command == "create-milvus-snapshot":
        return _create_milvus_snapshot(args)
    if args.command == "write-native-segment":
        return _write_native_segment(args)
    if args.command == "backfill-snapshot":
        return _backfill_snapshot(args)
    parser.print_help()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ray-milvus")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a Milvus snapshot")
    inspect_parser.add_argument("--snapshot", required=True)
    inspect_parser.add_argument("--s3-endpoint")
    inspect_parser.add_argument("--s3-bucket")
    inspect_parser.add_argument("--s3-access-key")
    inspect_parser.add_argument("--s3-secret-key")
    inspect_parser.add_argument("--s3-region")
    inspect_parser.add_argument("--s3-root-path")
    inspect_parser.add_argument("--s3-use-ssl", action="store_true")
    inspect_parser.add_argument("--json", action="store_true")

    create_parser = subparsers.add_parser(
        "create-snapshot",
        help="Create a toolkit snapshot JSON from schema and segment metadata",
    )
    create_parser.add_argument("--schema-file", required=True)
    create_parser.add_argument("--segments-file", required=True)
    create_parser.add_argument("--output", required=True)
    create_parser.add_argument("--collection-name")
    create_parser.add_argument("--overwrite", action="store_true")
    create_parser.add_argument("--compact", action="store_true")

    import_parser = subparsers.add_parser(
        "import-milvus-snapshot",
        help="Import an existing Milvus snapshot into ray-milvus JSON",
    )
    import_parser.add_argument("--uri", required=True)
    import_parser.add_argument("--collection-name", required=True)
    import_parser.add_argument("--snapshot-name", required=True)
    import_parser.add_argument("--output", required=True)
    import_parser.add_argument("--token")
    import_parser.add_argument("--user")
    import_parser.add_argument("--password")
    import_parser.add_argument("--db-name")
    _add_storage_options(import_parser)
    import_parser.add_argument("--overwrite", action="store_true")
    import_parser.add_argument("--compact", action="store_true")

    native_import_parser = subparsers.add_parser(
        "import-native-milvus-snapshot",
        help="Import a Milvus native snapshot from internal metadata paths",
    )
    native_import_source = native_import_parser.add_mutually_exclusive_group(required=True)
    native_import_source.add_argument("--metadata")
    native_import_source.add_argument("--snapshot-root")
    native_import_parser.add_argument("--manifest-dir")
    native_import_parser.add_argument("--collection-id")
    native_import_parser.add_argument("--snapshot-id")
    native_import_parser.add_argument("--output", required=True)
    native_import_parser.add_argument("--overwrite", action="store_true")
    native_import_parser.add_argument("--compact", action="store_true")

    milvus_snapshot_parser = subparsers.add_parser(
        "create-milvus-snapshot",
        help="Create a Milvus snapshot and import its S3 metadata into toolkit JSON",
    )
    milvus_snapshot_parser.add_argument("--uri", required=True)
    milvus_snapshot_parser.add_argument("--collection-name", required=True)
    milvus_snapshot_parser.add_argument("--snapshot-name")
    milvus_snapshot_parser.add_argument("--auto-snapshot-name", action="store_true")
    milvus_snapshot_parser.add_argument("--output", required=True)
    milvus_snapshot_parser.add_argument("--token")
    milvus_snapshot_parser.add_argument("--user")
    milvus_snapshot_parser.add_argument("--password")
    milvus_snapshot_parser.add_argument("--db-name")
    milvus_snapshot_parser.add_argument("--description")
    milvus_snapshot_parser.add_argument("--compaction-protection-seconds", type=int)
    _add_storage_options(milvus_snapshot_parser)
    milvus_snapshot_parser.add_argument("--overwrite", action="store_true")
    milvus_snapshot_parser.add_argument("--compact", action="store_true")

    write_parser = subparsers.add_parser(
        "write-native-segment",
        help="Write a Parquet file as a StorageV3 segment through internal metadata",
    )
    write_parser.add_argument("--input", required=True)
    write_parser.add_argument("--schema-file", required=True)
    write_parser.add_argument("--segment-path", required=True)
    write_parser.add_argument("--segment-id", required=True, type=int)
    write_parser.add_argument("--partition-id", type=int)
    write_parser.add_argument("--manifest-version")
    _add_storage_options(write_parser)
    write_parser.add_argument("--output")
    write_parser.add_argument("--snapshot-output")
    write_parser.add_argument("--collection-name")
    write_parser.add_argument("--overwrite", action="store_true")
    write_parser.add_argument("--compact", action="store_true")

    backfill_parser = subparsers.add_parser(
        "backfill-snapshot",
        help="Backfill fields into StorageV3 segments and write a toolkit snapshot",
    )
    backfill_parser.add_argument("--snapshot", required=True)
    backfill_parser.add_argument("--backfill", required=True)
    backfill_parser.add_argument("--schema-file", required=True)
    backfill_parser.add_argument("--primary-key", required=True)
    backfill_parser.add_argument("--fields", required=True)
    backfill_parser.add_argument("--output", required=True)
    backfill_parser.add_argument("--mode", default="coalesce")
    backfill_parser.add_argument("--segment-path-template", default="{manifest_path}")
    _add_storage_options(backfill_parser)
    backfill_parser.add_argument("--overwrite", action="store_true")
    backfill_parser.add_argument("--compact", action="store_true")
    return parser



def _add_storage_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--storage-root")
    parser.add_argument("--storage-type", default="local")
    parser.add_argument("--s3-endpoint")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-access-key")
    parser.add_argument("--s3-secret-key")
    parser.add_argument("--s3-region")
    parser.add_argument("--s3-use-ssl", action="store_true")
    parser.add_argument(
        "--storage-extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra milvus-storage property; can be repeated",
    )



def _inspect(args: argparse.Namespace) -> int:
    storage = mt.StorageConfig(
        endpoint=args.s3_endpoint,
        bucket=args.s3_bucket,
        access_key=args.s3_access_key,
        secret_key=args.s3_secret_key,
        use_ssl=args.s3_use_ssl,
        region=args.s3_region,
        root_path=args.s3_root_path,
    )
    try:
        result = mt.inspect_snapshot(args.snapshot, storage=storage)
    except MilvusToolkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
    else:
        print(f"Collection: {result.collection_name or '<unknown>'}")
        print(f"Segments: {result.segment_count}")
        if result.diagnostics:
            print("Diagnostics:")
            for diagnostic in result.diagnostics:
                segment = (
                    "" if diagnostic.segment_id is None else f" segment={diagnostic.segment_id}"
                )
                print(f"- {diagnostic.level}:{segment} {diagnostic.message}")
    return 0


def _create_snapshot(args: argparse.Namespace) -> int:
    try:
        result = mt.create_snapshot(
            args.schema_file,
            args.segments_file,
            output_path=args.output,
            collection_name=args.collection_name,
            overwrite=args.overwrite,
            pretty=not args.compact,
        )
    except MilvusToolkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        "Created snapshot "
        f"{args.output} for {len(result['segments'])} segment(s)"
    )
    return 0


def _import_milvus_snapshot(args: argparse.Namespace) -> int:
    try:
        result = mt.import_milvus_snapshot(
            uri=args.uri,
            collection_name=args.collection_name,
            snapshot_name=args.snapshot_name,
            output_path=args.output,
            storage=_storage_config_from_args(args),
            token=args.token,
            user=args.user,
            password=args.password,
            db_name=args.db_name,
            overwrite=args.overwrite,
            pretty=not args.compact,
        )
    except MilvusToolkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        "Imported Milvus snapshot "
        f"{args.snapshot_name} with {len(result['segments'])} segment(s) to {args.output}"
    )
    return 0



def _import_native_milvus_snapshot(args: argparse.Namespace) -> int:
    if args.snapshot_root is not None and (args.collection_id is None or args.snapshot_id is None):
        print(
            "error: --collection-id and --snapshot-id are required with --snapshot-root",
            file=sys.stderr,
        )
        return 1

    try:
        result = mt.import_native_milvus_snapshot(
            metadata_path=args.metadata,
            manifest_dir=args.manifest_dir,
            snapshot_root=args.snapshot_root,
            collection_id=args.collection_id,
            snapshot_id=args.snapshot_id,
            output_path=args.output,
            overwrite=args.overwrite,
            pretty=not args.compact,
        )
    except MilvusToolkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        "Imported native Milvus snapshot "
        f"{args.output} for {len(result['segments'])} segment(s)"
    )
    return 0



def _create_milvus_snapshot(args: argparse.Namespace) -> int:
    if args.snapshot_name is not None and args.auto_snapshot_name:
        print("error: --snapshot-name cannot be used with --auto-snapshot-name", file=sys.stderr)
        return 1
    if args.snapshot_name is None and not args.auto_snapshot_name:
        print(
            "error: --snapshot-name is required unless --auto-snapshot-name is set",
            file=sys.stderr,
        )
        return 1

    try:
        result = mt.create_snapshot_from_milvus(
            uri=args.uri,
            collection_name=args.collection_name,
            snapshot_name=args.snapshot_name,
            output_path=args.output,
            storage=_storage_config_from_args(args),
            auto_snapshot_name=args.auto_snapshot_name,
            token=args.token,
            user=args.user,
            password=args.password,
            db_name=args.db_name,
            description=args.description,
            compaction_protection_seconds=args.compaction_protection_seconds,
            overwrite=args.overwrite,
            pretty=not args.compact,
        )
    except MilvusToolkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        "Created Milvus snapshot "
        f"{result['snapshot_name']} and imported "
        f"{len(result['segments'])} segment(s) to {args.output}"
    )
    return 0



def _write_native_segment(args: argparse.Namespace) -> int:
    try:
        storage = _storage_config_from_args(args)
        schema = _load_json_file(args.schema_file)
        table = pq.read_table(args.input)
        snapshot = mt.write_snapshot(
            table,
            schema,
            storage,
            segment_path=args.segment_path,
            segment_id=args.segment_id,
            collection_name=args.collection_name,
            partition_id=args.partition_id,
            manifest_version=args.manifest_version,
        )
        segment = snapshot["segments"][0]
        if args.snapshot_output is not None:
            _write_json_file(
                args.snapshot_output,
                snapshot,
                overwrite=args.overwrite,
                pretty=not args.compact,
            )
        if args.output is not None:
            _write_json_file(
                args.output,
                segment,
                overwrite=args.overwrite,
                pretty=not args.compact,
            )
    except MilvusToolkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.snapshot_output is not None:
        print(
            "Wrote snapshot "
            f"{args.snapshot_output} for collection {snapshot.get('collection_name')} "
            f"with {len(snapshot['segments'])} segment(s)"
        )
    elif args.output is not None:
        print(f"Wrote segment {args.segment_id} metadata {args.output}")
    else:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0



def _backfill_snapshot(args: argparse.Namespace) -> int:
    try:
        storage = _storage_config_from_args(args)
        schema = _load_json_file(args.schema_file)
        backfill_table = pq.read_table(args.backfill)
        result = mt.backfill_snapshot(
            args.snapshot,
            storage,
            backfill_table,
            schema,
            primary_key=args.primary_key,
            fields=_split_fields(args.fields),
            output_path=args.output,
            mode=args.mode,
            segment_path_template=args.segment_path_template,
            overwrite=args.overwrite,
            pretty=not args.compact,
        )
    except MilvusToolkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Backfilled snapshot {args.output} for {len(result['segments'])} segment(s)")
    return 0



def _split_fields(value: str) -> tuple[str, ...]:
    return tuple(field.strip() for field in value.split(",") if field.strip())



def _split_optional_fields(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _split_fields(value)



def _storage_config_from_args(args: argparse.Namespace) -> mt.StorageConfig:
    return mt.StorageConfig(
        storage_type=args.storage_type,
        endpoint=args.s3_endpoint,
        bucket=args.s3_bucket,
        access_key=args.s3_access_key,
        secret_key=args.s3_secret_key,
        use_ssl=args.s3_use_ssl,
        region=args.s3_region,
        root_path=args.storage_root,
        extra=_parse_storage_extra(args.storage_extra),
    )



def _parse_storage_extra(values: list[str]) -> dict[str, str]:
    extra = {}
    for value in values:
        if "=" not in value:
            raise mt.ConfigError(f"--storage-extra must be KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        if not key:
            raise mt.ConfigError("--storage-extra key cannot be empty")
        extra[key] = item
    return extra



def _load_json_file(path: str) -> Any:
    with open(path, encoding="utf-8") as json_file:
        return json.load(json_file)



def _write_json_file(path: str, payload: Any, overwrite: bool, pretty: bool) -> None:
    output_path = _validate_output_path(path, overwrite=overwrite)
    if pretty:
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    output_path.write_text(f"{text}\n", encoding="utf-8")



def _validate_output_path(path: str, overwrite: bool):
    from pathlib import Path

    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise mt.ConfigError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path



def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
