"""scripts/corpus.py -- get a large raw corpus ready for the ~50M model.

Every step is explicit; nothing runs on import and nothing is uploaded unless asked.

  inspect FILE                         format, compression, size and field names of an
                                       unknown file (prints no document content)
  train-tokenizer --input F ...        train a byte-level BPE tokenizer artifact
  build-shards --input F ...           tokenize into resumable uint16 shards + manifest
  layout --name N --version V          print the R2 object paths for a dataset
  registry-row --name N --version-label V [--file F]
                                       print the ored_datasets row (metadata only)
  register ...                         send that row to Supabase (needs ORED_SB_* env)
  update-registry --name N --version-label V --set key=value ...
                                       fill in status / counts / sha256 later (ORED_SB_*);
                                       manifest=@file.json reads JSON from a file
  upload-raw --file F --name N --version V
                                       upload one raw file to the dataset folder in R2
  publish --dir D --name N --version V upload finished shards to R2 (needs ORED_R2_* env)
  fetch --name N --version V --tokenizer-label L --out D
                                       download shards from R2 and verify them

The text and id fields are always passed explicitly (--text-field / --id-field): the
corpus's schema is not assumed anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional

from ored.data.readers import detect_format, file_sha256, inspect_file, open_reader


def _documents(paths: List[str], text_field: str, id_field: Optional[str], fmt: Optional[str],
               max_documents: int, max_bytes: int) -> Iterator[str]:
    used = 0
    count = 0
    for path in paths:
        for doc in open_reader(path, text_field, id_field, file_format=fmt).iter_documents():
            yield doc.text
            count += 1
            used += len(doc.text.encode("utf-8"))
            if (max_documents and count >= max_documents) or (max_bytes and used >= max_bytes):
                return


def cmd_inspect(args: argparse.Namespace) -> int:
    for path in args.files:
        print(json.dumps(inspect_file(path, args.max_records), indent=2))
    return 0


def cmd_train_tokenizer(args: argparse.Namespace) -> int:
    from ored.data.tokenizer import END_OF_TEXT, SubwordTokenizer, count_words
    from ored.data.tokenizer_artifact import save_artifact

    texts = _documents(args.input, args.text_field, None, args.format, args.max_documents, args.max_bytes)
    words, seen = count_words(texts)
    print(f"{len(words):,} distinct pre-tokenized words; learning merges ...", file=sys.stderr)
    tokenizer = SubwordTokenizer.from_word_counts(words, seen, args.vocab_size, [END_OF_TEXT])
    info = save_artifact(tokenizer, args.out, args.name, args.version, training={
        "sources": [{"file_name": Path(p).name, "sha256": file_sha256(p)} for p in args.input],
        "text_field": args.text_field, "max_documents": args.max_documents, "max_bytes": args.max_bytes,
        "vocab_size_requested": args.vocab_size, "distinct_words": len(words)})
    print(json.dumps(info, indent=2))
    return 0


def cmd_build_shards(args: argparse.Namespace) -> int:
    from ored.data.token_shards import BuildSettings, build_token_shards
    from ored.data.tokenizer_artifact import load_artifact

    tokenizer, info = load_artifact(args.tokenizer)
    train = 1.0 - args.val_fraction - args.test_fraction
    settings = BuildSettings(text_field=args.text_field, id_field=args.id_field, split_seed=args.split_seed,
                             fractions={"train": train, "val": args.val_fraction, "test": args.test_fraction},
                             max_tokens_per_shard=args.max_tokens_per_shard)
    manifest = build_token_shards(args.input, tokenizer, info, args.out, settings, args.name, args.version,
                                  file_format=args.format, workers=args.workers,
                                  log=lambda m: print(m, file=sys.stderr))
    print(json.dumps({"content_sha256": manifest["content_sha256"], "counts": manifest["counts"],
                      "shards": len(manifest["shards"])}, indent=2))
    return 0


def _layout():
    from ored.storage import ObjectLayout, R2Settings
    if R2Settings.configured():
        return ObjectLayout(R2Settings.from_env().prefix)
    return ObjectLayout()


def cmd_layout(args: argparse.Namespace) -> int:
    layout = _layout()
    paths = {"dataset_dir": layout.dataset_dir(args.name, args.version),
             "manifest": layout.dataset_manifest(args.name, args.version)}
    if args.tokenizer_label:
        paths["token_shards"] = layout.token_dir(args.name, args.version, args.tokenizer_label)
    print(json.dumps(paths, indent=2))
    return 0


def _entry(args: argparse.Namespace):
    from ored.data.dataset_registry import DatasetEntry

    entry = DatasetEntry(name=args.name, version_label=args.version_label, dataset_type=args.dataset_type,
                         storage_bucket=args.bucket, external_id=args.external_id, source_id=args.source_id,
                         summary=args.summary or "")
    entry.storage_path = _layout().dataset_dir(args.name, args.version_label)
    if args.metadata:
        entry.metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    if args.file:
        path = Path(args.file)
        detected = detect_format(path)
        entry.file_name = path.name
        entry.size_bytes = path.stat().st_size
        entry.sha256 = file_sha256(path)
        entry.file_format = detected.format
        entry.compression = detected.compression
    return entry.validate()


def cmd_registry_row(args: argparse.Namespace) -> int:
    print(json.dumps(_entry(args).to_row(), indent=2))
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    from ored.learning.supabase_store import SupabaseStore

    row = SupabaseStore.from_env().register_external_dataset(_entry(args))
    print(json.dumps({k: row.get(k) for k in ("id", "name", "version", "version_label", "status",
                                              "storage_path", "sha256")}, indent=2))
    return 0


def cmd_update_registry(args: argparse.Namespace) -> int:
    from ored.learning.supabase_store import SupabaseStore

    fields = {}
    for assignment in args.set:
        key, _, value = assignment.partition("=")
        if value == "null":
            fields[key] = None
        elif key in ("size_bytes", "document_count", "token_count"):
            fields[key] = int(value)
        elif key in ("manifest", "metadata"):
            fields[key] = json.loads(Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value)
        else:
            fields[key] = value
    store = SupabaseStore.from_env()
    row = store.external_dataset(args.name, args.version_label)
    if row is None:
        raise SystemExit(f"{args.name} {args.version_label} is not registered")
    row = store.update_external_dataset(row["id"], fields)
    print(json.dumps({k: row.get(k) for k in ("id", "name", "version_label", "status", *fields)}, indent=2))
    return 0


def cmd_upload_raw(args: argparse.Namespace) -> int:
    from ored.storage import ObjectLayout, R2Settings

    settings = R2Settings.from_env()
    path = Path(args.file)
    key = ObjectLayout(settings.prefix).dataset_file(args.name, args.version, path.name)
    store = settings.store()
    store.upload(path, key, upsert=False)
    if store.stat(key) != path.stat().st_size:
        raise SystemExit(f"r2://{settings.bucket}/{key} does not have the local file's size")
    print(json.dumps({"bucket": settings.bucket, "object": key, "size_bytes": path.stat().st_size,
                      "sha256": file_sha256(path)}, indent=2))
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    from ored.data.token_shards import load_manifest
    from ored.storage import ObjectLayout, R2Settings
    from ored.storage.objects import publish_token_dataset

    settings = R2Settings.from_env()
    tokenizer = load_manifest(args.dir)["tokenizer"]
    label = f"{tokenizer['name']}-{tokenizer['version']}"
    object_dir = ObjectLayout(settings.prefix).token_dir(args.name, args.version, label)
    result = publish_token_dataset(settings.store(), args.dir, object_dir, log=lambda m: print(m, file=sys.stderr))
    print(json.dumps(result, indent=2))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from ored.storage import ObjectLayout, R2Settings
    from ored.storage.objects import fetch_token_dataset

    settings = R2Settings.from_env()
    object_dir = ObjectLayout(settings.prefix).token_dir(args.name, args.version, args.tokenizer_label)
    manifest = fetch_token_dataset(settings.store(), object_dir, args.out, log=lambda m: print(m, file=sys.stderr))
    print(json.dumps({"content_sha256": manifest["content_sha256"], "counts": manifest["counts"]}, indent=2))
    return 0


def _registry_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", required=True)
    p.add_argument("--version-label", required=True)
    p.add_argument("--file", help="the raw file, to record its name, size, format and sha256")
    p.add_argument("--bucket", help="the R2 bucket (ORED_R2_BUCKET)")
    p.add_argument("--dataset-type", default=None)
    p.add_argument("--external-id")
    p.add_argument("--source-id")
    p.add_argument("--summary")
    p.add_argument("--metadata", help="a JSON file with metadata (licence, provenance ...)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("files", nargs="+")
    p.add_argument("--max-records", type=int, default=5)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("train-tokenizer")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--text-field", required=True)
    p.add_argument("--format", choices=["jsonl", "parquet"])
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--max-documents", type=int, default=0)
    p.add_argument("--max-bytes", type=int, default=0, help="train on a sample of this many bytes")
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_train_tokenizer)

    p = sub.add_parser("build-shards")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--text-field", required=True)
    p.add_argument("--id-field")
    p.add_argument("--format", choices=["jsonl", "parquet"])
    p.add_argument("--tokenizer", required=True, help="tokenizer artifact (train-tokenizer --out)")
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--split-seed", type=int, default=1337)
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--test-fraction", type=float, default=0.01)
    p.add_argument("--max-tokens-per-shard", type=int, default=64 * 1024 * 1024)
    p.set_defaults(func=cmd_build_shards)

    p = sub.add_parser("layout")
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--tokenizer-label")
    p.set_defaults(func=cmd_layout)

    p = sub.add_parser("registry-row")
    _registry_arguments(p)
    p.set_defaults(func=cmd_registry_row)

    p = sub.add_parser("register")
    _registry_arguments(p)
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("update-registry")
    p.add_argument("--name", required=True)
    p.add_argument("--version-label", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", required=True)
    p.set_defaults(func=cmd_update_registry)

    p = sub.add_parser("upload-raw")
    p.add_argument("--file", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.set_defaults(func=cmd_upload_raw)

    p = sub.add_parser("publish")
    p.add_argument("--dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("fetch")
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--tokenizer-label", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fetch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
