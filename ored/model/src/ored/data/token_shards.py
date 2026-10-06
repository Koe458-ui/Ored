from __future__ import annotations

import bisect
import hashlib
import json
import os
import shutil
import sys
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ored.data.readers import DatasetReader, file_sha256
from ored.data.tokenizer import END_OF_TEXT, Tokenizer
from ored.data.tokenizer_artifact import identity as tokenizer_identity
from ored.data.training_data import split_of

SHARDS_FORMAT = "ored-token-shards/1"
MANIFEST = "manifest.json"
PROGRESS = ".progress.json"
SPLITS = ("train", "val", "test")
DEFAULT_MAX_TOKENS_PER_SHARD = 64 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 128 * 1024 * 1024
FLUSH_TOKENS = 1 << 20


class ShardError(RuntimeError):
    pass


def dtype_for(vocab_size: int) -> str:
    return "uint16" if vocab_size <= 1 << 16 else "uint32"


def _array_code(dtype: str) -> str:
    return {"uint16": "H", "uint32": "I"}[dtype]


def _code_version() -> Dict[str, Any]:
    from ored.data.snapshot import code_version
    return code_version()


@dataclass
class _ShardWriter:
    directory: Path
    split: str
    unit: str
    dtype: str
    max_tokens: int
    entries: List[Dict[str, Any]] = field(default_factory=list)
    _index: int = 0
    _handle: Any = None
    _sha: Any = None
    _tokens: int = 0
    _documents: int = 0
    _buffer: Optional[array] = None

    def _name(self) -> str:
        return f"{self.split}/{self.unit}-{self._index:04d}.bin"

    def _open(self) -> None:
        path = self.directory / self._name()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path.with_suffix(".bin.part"), "wb")
        self._sha = hashlib.sha256()
        self._tokens = self._documents = 0
        self._buffer = array(_array_code(self.dtype))

    def _flush(self) -> None:
        if self._buffer:
            if sys.byteorder == "big":
                self._buffer.byteswap()
            data = self._buffer.tobytes()
            self._handle.write(data)
            self._sha.update(data)
            self._buffer = array(_array_code(self.dtype))

    def add(self, ids: Sequence[int]) -> None:
        if self._handle is None:
            self._open()
        self._buffer.extend(ids)
        self._tokens += len(ids)
        self._documents += 1
        if len(self._buffer) >= FLUSH_TOKENS:
            self._flush()
        if self._tokens >= self.max_tokens:
            self._close_shard()

    def _close_shard(self) -> None:
        self._flush()
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        name = self._name()
        final = self.directory / name
        os.replace(final.with_suffix(".bin.part"), final)
        self.entries.append({"path": name, "split": self.split, "unit": self.unit,
                             "tokens": self._tokens, "documents": self._documents,
                             "bytes": final.stat().st_size, "sha256": self._sha.hexdigest()})
        self._handle = None
        self._index += 1

    def close(self) -> None:
        if self._handle is not None:
            self._close_shard()


@dataclass
class BuildSettings:

    text_field: str
    id_field: Optional[str]
    split_seed: int = 1337
    fractions: Dict[str, float] = field(default_factory=lambda: {"train": 0.98, "val": 0.01, "test": 0.01})
    max_tokens_per_shard: int = DEFAULT_MAX_TOKENS_PER_SHARD
    chunk_bytes: int = DEFAULT_CHUNK_BYTES

    def validate(self) -> None:
        if abs(sum(self.fractions.values()) - 1.0) > 1e-6 or set(self.fractions) != set(SPLITS):
            raise ShardError(f"split fractions must cover {SPLITS} and sum to 1, got {self.fractions}")
        if self.max_tokens_per_shard < 1024:
            raise ShardError("max_tokens_per_shard must be >= 1024")


def _group(doc_id: Optional[str], text: str) -> str:
    if doc_id is not None:
        return "id:" + doc_id
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize_unit(unit: str, reader: DatasetReader, tokenizer: Tokenizer, out_dir: Path,
                  settings: BuildSettings, eos_id: int,
                  progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    dtype = dtype_for(tokenizer.vocab_size)
    for stale in out_dir.glob(f"*/{unit}-*"):
        stale.unlink()
    writers = {s: _ShardWriter(out_dir, s, unit, dtype, settings.max_tokens_per_shard) for s in SPLITS}
    documents = 0
    invalid = 0
    started = time.time()
    for doc in reader.iter_documents():
        try:
            ids = tokenizer.encode(doc.text)
        except UnicodeEncodeError:
            invalid += 1
            continue
        ids.append(eos_id)
        split = split_of(_group(doc.id, doc.text), settings.split_seed, settings.fractions)
        writers[split].add(ids)
        documents += 1
        if progress is not None and documents % 10000 == 0:
            progress(f"  unit {unit}: {documents:,} documents, {time.time() - started:,.0f}s")
    for writer in writers.values():
        writer.close()
    return {"unit": unit, "documents": documents, "skipped_empty": reader.skipped,
            "skipped_invalid_unicode": invalid,
            "shards": [e for s in SPLITS for e in writers[s].entries]}


def _read_progress(out_dir: Path, signature: str) -> Dict[str, Any]:
    path = out_dir / PROGRESS
    if not path.is_file():
        return {"signature": signature, "units": {}}
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("signature") != signature:
        raise ShardError(f"{out_dir} holds a partial build with different settings, sources or tokenizer; "
                         f"use an empty directory (or delete it) to start over")
    return progress


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def content_sha256(manifest: Dict[str, Any]) -> str:
    identity = {k: manifest[k] for k in ("format", "dataset", "reader", "tokenizer", "dtype",
                                          "eos_id", "split", "max_tokens_per_shard")}
    identity["dataset"] = {k: v for k, v in manifest["dataset"].items() if k != "registry_id"}
    identity["shards"] = [{k: s[k] for k in ("path", "split", "tokens", "documents", "sha256")}
                          for s in manifest["shards"]]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def unit_signature(tokenizer_sha256: str, settings: BuildSettings, file_format: Optional[str]) -> str:
    return hashlib.sha256(json.dumps({"tokenizer": tokenizer_sha256, "settings": settings.__dict__,
                                      "format": file_format}, sort_keys=True).encode()).hexdigest()


def plan_units(sources: List[Dict[str, Any]], paths: Sequence[Path], chunk_bytes: int) -> List[Dict[str, Any]]:
    from ored.data.readers import line_chunks
    units = []
    for index, (source, path) in enumerate(zip(sources, paths)):
        splittable = source["file_format"] == "jsonl" and source["compression"] is None
        ranges = line_chunks(path, chunk_bytes) if splittable else [None]
        for chunk, byte_range in enumerate(ranges):
            units.append({"key": f"{source['sha256'][:16]}-c{chunk:03d}", "source": index,
                          "source_sha256": source["sha256"], "byte_range": byte_range})
    return units


def _unit_worker(args: Tuple[Dict[str, Any], str, Dict[str, Any], str, Dict[str, Any], int, Optional[str]]
                 ) -> Dict[str, Any]:
    from ored.data.readers import open_reader
    unit, path, tokenizer_dict, out_dir, settings_dict, eos_id, fmt = args
    settings = BuildSettings(**settings_dict)
    byte_range = tuple(unit["byte_range"]) if unit["byte_range"] is not None else None
    reader = open_reader(path, settings.text_field, settings.id_field, file_format=fmt, byte_range=byte_range)
    result = tokenize_unit(unit["key"], reader, Tokenizer.from_dict(tokenizer_dict), Path(out_dir), settings,
                           eos_id)
    result.update(source_sha256=unit["source_sha256"], byte_range=unit["byte_range"])
    return result


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _reusable_units(reuse_from: Sequence[str | Path], signature: str) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
    found: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
    for directory in map(Path, reuse_from):
        manifest = load_manifest(directory)
        if manifest.get("unit_signature") != signature:
            continue
        for key, unit in (manifest.get("units") or {}).items():
            found.setdefault(key, (directory, unit))
    return found


def build_token_shards(sources: Sequence[str | Path], tokenizer: Tokenizer, tokenizer_info: Dict[str, Any],
                       out_dir: str | Path, settings: BuildSettings, dataset_name: str, dataset_version: str,
                       file_format: Optional[str] = None, workers: int = 1,
                       reuse_from: Sequence[str | Path] = (),
                       log: Callable[[str], None] = print) -> Dict[str, Any]:
    settings.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / MANIFEST).is_file():
        raise ShardError(f"{out_dir / MANIFEST} already exists: this dataset version is finished and immutable")
    if tokenizer_info.get("sha256") != tokenizer_identity(tokenizer)["sha256"]:
        raise ShardError("tokenizer_info does not describe this tokenizer")
    if not hasattr(tokenizer, "special_id"):
        raise ShardError("token shards need a tokenizer with an end-of-text token")
    eos_id = tokenizer.special_id(END_OF_TEXT)

    paths = [Path(p) for p in sources]
    log(f"hashing {len(paths)} source file(s) ...")
    from ored.data.readers import detect_format
    described = []
    for path in paths:
        detected = detect_format(path)
        described.append({"file_name": path.name, "size_bytes": path.stat().st_size,
                          "sha256": file_sha256(path), "file_format": file_format or detected.format,
                          "compression": detected.compression})
    if len({d["sha256"] for d in described}) != len(described):
        raise ShardError("the same file is listed twice")
    formats = {d["file_format"] for d in described}
    signature = unit_signature(tokenizer_info["sha256"], settings, formats.pop() if len(formats) == 1 else None)
    units = plan_units(described, paths, settings.chunk_bytes)
    progress = _read_progress(out_dir, hashlib.sha256(json.dumps({
        "unit_signature": signature, "units": [u["key"] for u in units],
        "name": dataset_name, "version": dataset_version}, sort_keys=True).encode()).hexdigest())
    _write_json(out_dir / PROGRESS, progress)

    reusable = _reusable_units(reuse_from, signature)
    for unit in units:
        if unit["key"] in progress["units"] or unit["key"] not in reusable:
            continue
        directory, record = reusable[unit["key"]]
        for shard in record["shards"]:
            _link(directory / shard["path"], out_dir / shard["path"])
        progress["units"][unit["key"]] = record
        _write_json(out_dir / PROGRESS, progress)
        log(f"unit {unit['key']} reused from {directory}")

    todo = [u for u in units if u["key"] not in progress["units"]]
    log(f"{len(units)} unit(s): {len(units) - len(todo)} ready, {len(todo)} to tokenize")
    jobs = [(u, str(paths[u["source"]]), tokenizer.to_dict(), str(out_dir), dict(settings.__dict__), eos_id,
             described[u["source"]]["file_format"]) for u in todo]

    def record(result: Dict[str, Any]) -> None:
        progress["units"][result["unit"]] = result
        _write_json(out_dir / PROGRESS, progress)
        tokens = sum(s["tokens"] for s in result["shards"])
        log(f"unit {result['unit']} done: {result['documents']:,} documents, {tokens:,} tokens, "
            f"{len(result['shards'])} shard(s)")

    if workers > 1 and len(jobs) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            for result in pool.imap_unordered(_unit_worker, jobs):
                record(result)
    else:
        for job in jobs:
            record(_unit_worker(job))

    done = [progress["units"][u["key"]] for u in units]
    shards = [s for u in done for s in u["shards"]]
    for index, source in enumerate(described):
        mine = [d for u, d in zip(units, done) if u["source"] == index]
        source.update(documents=sum(d["documents"] for d in mine),
                      skipped_empty=sum(d["skipped_empty"] for d in mine),
                      skipped_invalid_unicode=sum(d["skipped_invalid_unicode"] for d in mine))
    manifest: Dict[str, Any] = {
        "format": SHARDS_FORMAT,
        "dataset": {"name": dataset_name, "version": dataset_version, "registry_id": None,
                    "sources": described},
        "reader": {"text_field": settings.text_field, "id_field": settings.id_field,
                   "chunk_bytes": settings.chunk_bytes},
        "tokenizer": tokenizer_info,
        "dtype": dtype_for(tokenizer.vocab_size),
        "eos_id": eos_id,
        "split": {"seed": settings.split_seed, "fractions": settings.fractions,
                  "unit": "document: id_field when set, else the sha256 of its text"},
        "max_tokens_per_shard": settings.max_tokens_per_shard,
        "unit_signature": signature,
        "units": {u["key"]: d for u, d in zip(units, done)},
        "shards": shards,
        "counts": {
            "documents": {s: sum(x["documents"] for x in shards if x["split"] == s) for s in SPLITS},
            "tokens": {s: sum(x["tokens"] for x in shards if x["split"] == s) for s in SPLITS},
        },
        "code": _code_version(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["content_sha256"] = content_sha256(manifest)
    verify_shards(out_dir, manifest, "full" if reusable else "size")
    _write_json(out_dir / MANIFEST, manifest)
    (out_dir / PROGRESS).unlink()
    return manifest


def load_manifest(directory: str | Path) -> Dict[str, Any]:
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise ShardError(f"no {MANIFEST} in {directory}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != SHARDS_FORMAT:
        raise ShardError(f"{path} is not an {SHARDS_FORMAT} manifest")
    if content_sha256(manifest) != manifest.get("content_sha256"):
        raise ShardError(f"{path} was edited: its content no longer matches content_sha256")
    return manifest


def manifest_file_sha256(directory: str | Path) -> str:
    return file_sha256(Path(directory) / MANIFEST)


def verify_shards(directory: str | Path, manifest: Dict[str, Any], mode: str = "full",
                  splits: Iterable[str] = SPLITS) -> None:
    directory = Path(directory)
    wanted = set(splits)
    itemsize = np.dtype(manifest["dtype"]).itemsize
    problems = []
    for shard in manifest["shards"]:
        if shard["split"] not in wanted:
            continue
        path = directory / shard["path"]
        if not path.is_file():
            problems.append(f"{shard['path']} is missing")
        elif path.stat().st_size != shard["bytes"] or shard["bytes"] != shard["tokens"] * itemsize:
            problems.append(f"{shard['path']} has the wrong size")
        elif mode == "full" and file_sha256(path) != shard["sha256"]:
            problems.append(f"{shard['path']} fails its sha256")
    if problems:
        raise ShardError(f"{directory} is damaged: " + "; ".join(problems[:10]))


def dataset_identity(manifest: Dict[str, Any], manifest_sha256: str = "",
                     registry_id: str = "") -> Dict[str, Any]:
    dataset = manifest["dataset"]
    return {
        "format": manifest["format"],
        "name": dataset["name"],
        "version": dataset["version"],
        "registry_id": registry_id or dataset.get("registry_id"),
        "content_sha256": manifest["content_sha256"],
        "manifest_sha256": manifest_sha256 or None,
        "sources": [{k: s.get(k) for k in ("file_name", "sha256", "size_bytes")} for s in dataset["sources"]],
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "tokens": manifest["counts"]["tokens"],
        "documents": manifest["counts"]["documents"],
    }


class TokenWindowDataset(Dataset):

    def __init__(self, directory: str | Path, manifest: Dict[str, Any], split: str,
                 block_size: int, stride: int) -> None:
        if stride < 1 or block_size < 1:
            raise ValueError("block_size and stride must be >= 1")
        self.directory = Path(directory)
        self.dtype = np.dtype(manifest["dtype"]).newbyteorder("<")
        self.block_size = block_size
        self.stride = stride
        self.paths: List[str] = []
        self.lengths: List[int] = []
        starts = [0]
        for shard in manifest["shards"]:
            if shard["split"] != split:
                continue
            windows = (shard["tokens"] - block_size - 1) // stride + 1 if shard["tokens"] > block_size else 0
            if windows <= 0:
                continue
            self.paths.append(shard["path"])
            self.lengths.append(shard["tokens"])
            starts.append(starts[-1] + windows)
        self.starts = starts
        self.n_tokens = sum(s["tokens"] for s in manifest["shards"] if s["split"] == split)
        self._maps: Dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.starts[-1]

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_maps"] = {}
        return state

    def _map(self, index: int) -> np.memmap:
        found = self._maps.get(index)
        if found is None:
            found = np.memmap(self.directory / self.paths[index], dtype=self.dtype, mode="r",
                              shape=(self.lengths[index],))
            self._maps[index] = found
        return found

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.starts, index) - 1
        start = (index - self.starts[shard]) * self.stride
        window = torch.from_numpy(self._map(shard)[start:start + self.block_size + 1].astype(np.int64))
        return window[:-1], window[1:]

    def describe(self) -> str:
        return (f"{len(self):>9,} windows | {self.n_tokens:>13,} tokens | {len(self.paths)} shard(s) | "
                f"block_size {self.block_size} | stride {self.stride}")
