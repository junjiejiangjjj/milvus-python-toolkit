from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

import pyarrow.parquet as pq

import milvus_toolkit as mt
from milvus_toolkit.errors import MilvusToolkitError


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
    if args.command == "write-segment":
        return _write_segment(args)
    if args.command == "backfill-snapshot":
        return _backfill_snapshot(args)
    parser.print_help()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="milvus-toolkit")
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
    schema_source = create_parser.add_mutually_exclusive_group(required=True)
    schema_source.add_argument("--schema-file")
    schema_source.add_argument("--schema-from-milvus", action="store_true")
    create_parser.add_argument("--segments-file", required=True)
    create_parser.add_argument("--output", required=True)
    create_parser.add_argument("--collection-name")
    create_parser.add_argument("--uri")
    create_parser.add_argument("--token")
    create_parser.add_argument("--user")
    create_parser.add_argument("--password")
    create_parser.add_argument("--db-name")
    create_parser.add_argument("--overwrite", action="store_true")
    create_parser.add_argument("--compact", action="store_true")

    import_parser = subparsers.add_parser(
        "import-milvus-snapshot",
        help="Import a Milvus native snapshot into toolkit snapshot JSON",
    )
    import_source = import_parser.add_mutually_exclusive_group(required=True)
    import_source.add_argument("--metadata")
    import_source.add_argument("--snapshot-root")
    import_parser.add_argument("--manifest-dir")
    import_parser.add_argument("--collection-id")
    import_parser.add_argument("--snapshot-id")
    import_parser.add_argument("--output", required=True)
    import_parser.add_argument("--overwrite", action="store_true")
    import_parser.add_argument("--compact", action="store_true")

    write_parser = subparsers.add_parser(
        "write-segment",
        help="Write a Parquet file as a StorageV3 segment through milvus-storage",
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
    if args.schema_from_milvus and args.uri is None:
        print("error: --uri is required with --schema-from-milvus", file=sys.stderr)
        return 1
    if args.schema_from_milvus and args.collection_name is None:
        print("error: --collection-name is required with --schema-from-milvus", file=sys.stderr)
        return 1

    try:
        if args.schema_from_milvus:
            result = mt.create_snapshot_from_milvus(
                uri=args.uri,
                collection_name=args.collection_name,
                segments=args.segments_file,
                output_path=args.output,
                token=args.token,
                user=args.user,
                password=args.password,
                db_name=args.db_name,
                overwrite=args.overwrite,
                pretty=not args.compact,
            )
        else:
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
    if args.snapshot_root is not None and (args.collection_id is None or args.snapshot_id is None):
        print(
            "error: --collection-id and --snapshot-id are required with --snapshot-root",
            file=sys.stderr,
        )
        return 1

    try:
        result = mt.import_milvus_snapshot(
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
        "Imported Milvus snapshot "
        f"{args.output} for {len(result['segments'])} segment(s)"
    )
    return 0



def _write_segment(args: argparse.Namespace) -> int:
    try:
        storage = _storage_config_from_args(args)
        schema = _load_json_file(args.schema_file)
        table = pq.read_table(args.input)
        if args.snapshot_output is None:
            segment = mt.write_segment(
                table,
                schema,
                storage,
                segment_path=args.segment_path,
                segment_id=args.segment_id,
                partition_id=args.partition_id,
                manifest_version=args.manifest_version,
            )
        else:
            snapshot = mt.write_snapshot(
                table,
                schema,
                storage,
                segment_path=args.segment_path,
                segment_id=args.segment_id,
                output_path=args.snapshot_output,
                collection_name=args.collection_name,
                partition_id=args.partition_id,
                manifest_version=args.manifest_version,
                overwrite=args.overwrite,
                pretty=not args.compact,
            )
            segment = snapshot["segments"][0]
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
        print(f"Wrote segment {args.segment_id} and snapshot {args.snapshot_output}")
    elif args.output is not None:
        print(f"Wrote segment {args.segment_id} metadata {args.output}")
    else:
        print(json.dumps(segment, indent=2, sort_keys=True))
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
    from pathlib import Path

    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise mt.ConfigError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    output_path.write_text(f"{text}\n", encoding="utf-8")



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
