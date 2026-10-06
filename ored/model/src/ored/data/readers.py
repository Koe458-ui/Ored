from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Sequence, Tuple

RAW_SUFFIXES = (".jsonl", ".ndjson", ".json", ".jsonl.gz", ".json.gz", ".ndjson.gz", ".jsonl.zst",
                ".json.zst", ".parquet")
GZIP_MAGIC = b"\x1f\x8b"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
PARQUET_MAGIC = b"PAR1"

FORMATS = ("jsonl", "parquet")
COMPRESSIONS = ("gzip", "zstd")


class ReaderError(ValueError):
    pass


@dataclass
class Document:
    id: Optional[str]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileFormat:
    format: Optional[str]
    compression: Optional[str]

    def describe(self) -> str:
        return f"{self.format or 'unknown'}{'+' + self.compression if self.compression else ''}"


def detect_format(path: str | Path) -> FileFormat:
    path = Path(path)
    with open(path, "rb") as handle:
        head = handle.read(4)
    suffixes = [s.lower() for s in path.suffixes]
    if head == PARQUET_MAGIC:
        return FileFormat("parquet", None)
    compression = "gzip" if head[:2] == GZIP_MAGIC else "zstd" if head == ZSTD_MAGIC else None
    inner = [s for s in suffixes if s not in (".gz", ".zst", ".zstd")]
    fmt = None
    if inner and inner[-1] in (".jsonl", ".ndjson", ".json"):
        fmt = "jsonl"
    elif inner and inner[-1] == ".parquet":
        fmt = "parquet"
    return FileFormat(fmt, compression)


def _open_binary(path: Path, compression: Optional[str]) -> BinaryIO:
    if compression is None:
        return open(path, "rb")
    if compression == "gzip":
        return gzip.open(path, "rb")
    if compression == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise ReaderError(f"{path.name} is zstd-compressed: pip install zstandard "
                              f"(optional dependency, see pyproject extras 'corpus')") from exc
        raw = open(path, "rb")
        return io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(
            raw, read_across_frames=True, closefd=True))
    raise ReaderError(f"unsupported compression {compression!r}")


class DatasetReader:

    def __init__(self, path: str | Path, text_field: str, id_field: Optional[str] = None,
                 metadata_fields: Sequence[str] = ()) -> None:
        if not text_field:
            raise ReaderError("a reader needs text_field: the field that holds each document's text")
        self.path = Path(path)
        self.text_field = text_field
        self.id_field = id_field
        self.metadata_fields = list(metadata_fields)
        self.skipped = 0

    def _records(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def iter_documents(self) -> Iterator[Document]:
        for number, record in enumerate(self._records()):
            text = _field(record, self.text_field)
            if not isinstance(text, str) or not text:
                self.skipped += 1
                continue
            doc_id = _field(record, self.id_field) if self.id_field else None
            if self.id_field and doc_id is None:
                raise ReaderError(f"{self.path.name} record {number} has no {self.id_field!r}")
            metadata = {name: _field(record, name) for name in self.metadata_fields}
            yield Document(None if doc_id is None else str(doc_id), text, metadata)

    def __iter__(self) -> Iterator[Document]:
        return self.iter_documents()


def _field(record: Dict[str, Any], dotted: Optional[str]) -> Any:
    value: Any = record
    for part in (dotted or "").split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


class JsonlReader(DatasetReader):

    def __init__(self, path: str | Path, text_field: str, id_field: Optional[str] = None,
                 metadata_fields: Sequence[str] = (), compression: Optional[str] = "detect",
                 byte_range: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(path, text_field, id_field, metadata_fields)
        self.compression = detect_format(self.path).compression if compression == "detect" else compression
        if byte_range is not None and self.compression is not None:
            raise ReaderError(f"{self.path.name} is compressed: it cannot be read in byte ranges")
        self.byte_range = byte_range

    def _lines(self, handle: BinaryIO) -> Iterator[bytes]:
        if self.byte_range is None:
            yield from handle
            return
        start, end = self.byte_range
        handle.seek(start)
        position = start
        while position < end:
            line = handle.readline()
            if not line:
                return
            position += len(line)
            yield line

    def _records(self) -> Iterator[Dict[str, Any]]:
        with _open_binary(self.path, self.compression) as handle:
            for number, line in enumerate(self._lines(handle), start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ReaderError(f"{self.path.name} line {number} is not JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise ReaderError(f"{self.path.name} line {number} is not a JSON object")
                yield record


class ParquetReader(DatasetReader):

    def __init__(self, path: str | Path, text_field: str, id_field: Optional[str] = None,
                 metadata_fields: Sequence[str] = (), batch_size: int = 1024) -> None:
        super().__init__(path, text_field, id_field, metadata_fields)
        self.batch_size = batch_size

    def _records(self) -> Iterator[Dict[str, Any]]:
        parquet = _pyarrow_parquet(self.path)
        top = {f.split(".")[0] for f in [self.text_field, self.id_field, *self.metadata_fields] if f}
        for batch in parquet.ParquetFile(self.path).iter_batches(batch_size=self.batch_size,
                                                                 columns=sorted(top)):
            yield from batch.to_pylist()


def _pyarrow_parquet(path: Path) -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ReaderError(f"{path.name} is Parquet: install pyarrow (pip install -r requirements.txt)") from exc
    return parquet


READERS = {"jsonl": JsonlReader, "parquet": ParquetReader}


def open_reader(path: str | Path, text_field: str, id_field: Optional[str] = None,
                metadata_fields: Sequence[str] = (), file_format: Optional[str] = None,
                byte_range: Optional[Tuple[int, int]] = None) -> DatasetReader:
    detected = detect_format(path)
    fmt = file_format or detected.format
    if fmt not in READERS:
        raise ReaderError(f"cannot tell how to read {Path(path).name} (detected {detected.describe()}); "
                          f"pass the format explicitly: one of {', '.join(READERS)}")
    if byte_range is not None:
        if fmt != "jsonl":
            raise ReaderError(f"only uncompressed JSONL can be read in byte ranges, not {fmt}")
        return JsonlReader(path, text_field, id_field, metadata_fields, byte_range=byte_range)
    return READERS[fmt](path, text_field, id_field, metadata_fields)


def line_chunks(path: str | Path, chunk_bytes: int) -> List[Tuple[int, int]]:
    path = Path(path)
    size = path.stat().st_size
    if chunk_bytes <= 0 or size <= chunk_bytes:
        return [(0, size)]
    starts = [0]
    with open(path, "rb") as handle:
        target = chunk_bytes
        while target < size:
            handle.seek(target)
            handle.readline()
            position = handle.tell()
            if position >= size:
                break
            if position > starts[-1]:
                starts.append(position)
            target = position + chunk_bytes
    return [(a, b) for a, b in zip(starts, starts[1:] + [size])]


def file_sha256(path: str | Path, chunk: int = 4 * 1024 * 1024) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            sha.update(block)
    return sha.hexdigest()


def inspect_file(path: str | Path, max_records: int = 5) -> Dict[str, Any]:
    path = Path(path)
    detected = detect_format(path)
    report: Dict[str, Any] = {"file_name": path.name, "size_bytes": path.stat().st_size,
                              "file_format": detected.format, "compression": detected.compression,
                              "records_sampled": 0, "fields": {}}
    if detected.format == "jsonl":
        records: List[Dict[str, Any]] = []
        with _open_binary(path, detected.compression) as handle:
            for line in handle:
                if line.strip():
                    try:
                        value = json.loads(line)
                    except ValueError:
                        report["error"] = "a line is not JSON"
                        break
                    if isinstance(value, dict):
                        records.append(value)
                if len(records) >= max_records:
                    break
    elif detected.format == "parquet":
        parquet = _pyarrow_parquet(path)
        handle = parquet.ParquetFile(path)
        report["parquet_rows"] = handle.metadata.num_rows
        report["parquet_schema"] = [f"{f.name}: {f.type}" for f in handle.schema_arrow]
        records = next(handle.iter_batches(batch_size=max_records), None)
        records = records.to_pylist() if records is not None else []
    else:
        records = []
    report["records_sampled"] = len(records)
    fields: Dict[str, set] = {}
    for record in records:
        for key, value in _flatten(record).items():
            fields.setdefault(key, set()).add(type(value).__name__)
    report["fields"] = {k: sorted(v) for k, v in sorted(fields.items())}
    return report


def _flatten(record: Dict[str, Any], prefix: str = "", depth: int = 2) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and depth > 0:
            out.update(_flatten(value, name + ".", depth - 1))
        else:
            out[name] = value
    return out


TEXT_NAMES = ("text", "content", "body", "document", "raw_content", "raw_text", "article", "passage")
ID_NAMES = ("id", "doc_id", "document_id", "docid", "uuid", "_id", "key")


def _pick(candidates: Dict[str, float], preferred: Sequence[str]) -> Optional[str]:
    lowered = {name.lower(): name for name in candidates}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def detect_fields(path: str | Path, file_format: Optional[str] = None) -> Dict[str, Any]:
    path = Path(path)
    detected = detect_format(path)
    fmt = file_format or detected.format
    text_sizes: Dict[str, float] = {}
    id_fields: Dict[str, float] = {}
    if fmt == "parquet":
        import pyarrow as pa
        handle = _pyarrow_parquet(path).ParquetFile(path)
        meta = handle.metadata
        sizes: Dict[str, float] = {}
        for group in range(meta.num_row_groups):
            row_group = meta.row_group(group)
            for column in range(row_group.num_columns):
                chunk = row_group.column(column)
                sizes[chunk.path_in_schema] = sizes.get(chunk.path_in_schema, 0.0) + chunk.total_uncompressed_size
        for field in handle.schema_arrow:
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                text_sizes[field.name] = sizes.get(field.name, 0.0)
            if (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
                    or pa.types.is_integer(field.type)):
                id_fields[field.name] = 0.0
    elif fmt == "jsonl":
        records = []
        with _open_binary(path, detected.compression) as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        records.append(_flatten(value))
                if len(records) >= 200:
                    break
        for record in records:
            for key, value in record.items():
                if isinstance(value, str):
                    text_sizes[key] = text_sizes.get(key, 0.0) + len(value)
                if isinstance(value, (str, int)) and not isinstance(value, bool):
                    id_fields[key] = 0.0
    else:
        raise ReaderError(f"cannot detect the fields of {path.name}: unknown format {detected.describe()}")
    text_field = _pick(text_sizes, TEXT_NAMES)
    if text_field is None:
        raise ReaderError(f"{path.name} has no text column")
    lowered = {name.lower(): name for name in id_fields if name != text_field}
    id_field = next((lowered[n] for n in ID_NAMES if n in lowered), None)
    return {"text_field": text_field, "id_field": id_field, "file_format": fmt,
            "text_candidates": sorted(text_sizes, key=text_sizes.get, reverse=True)}
