from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

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
    return parser


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
