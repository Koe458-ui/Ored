from __future__ import annotations

import gzip
import io
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

from ored.data.readers import RAW_SUFFIXES, ReaderError, file_sha256, inspect_file
from ored.storage.layout import DEFAULT_PREFIX

COMPRESSED_SUFFIXES = {".gz": "gzip", ".zst": "zstd", ".zstd": "zstd"}


def list_raw(store: Any, prefix: str = "") -> List[Dict[str, Any]]:
    objects = store.list_objects(prefix)
    return [{"key": key, "size_bytes": size} for key, size in sorted(objects.items())
            if key.lower().endswith(RAW_SUFFIXES) and not key.startswith(DEFAULT_PREFIX + "/")]


class RangedObject(io.RawIOBase):

    def __init__(self, store: Any, key: str, size: int) -> None:
        self.store = store
        self.key = key
        self.size = size
        self.position = 0
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, min(self.size, base + offset))
        return self.position

    def readinto(self, buffer: Any) -> int:
        length = min(len(buffer), self.size - self.position)
        if length <= 0:
            return 0
        data = self.store.read_range(self.key, self.position, length)
        buffer[:len(data)] = data
        self.position += len(data)
        self.fetched += len(data)
        return len(data)


def inspect_parquet_object(store: Any, key: str, size: int) -> Dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ReaderError("reading Parquet needs pyarrow: pip install -r requirements.txt") from exc
    source = RangedObject(store, key, size)
    meta = parquet.ParquetFile(io.BufferedReader(source, buffer_size=1024 * 1024))
    schema = meta.schema_arrow
    return {"key": key, "file_name": Path(key).name, "size_bytes": size, "file_format": "parquet",
            "compression": None, "parquet_rows": meta.metadata.num_rows,
            "parquet_row_groups": meta.metadata.num_row_groups,
            "parquet_schema": [f"{f.name}: {f.type}" for f in schema],
            "fields": {f.name: [str(f.type)] for f in schema},
            "downloaded_bytes": source.fetched}


def inspect_object(store: Any, key: str, head_bytes: int = 8 * 1024 * 1024) -> Dict[str, Any]:
    size = store.stat(key)
    if store.read_head(key, 4) == b"PAR1":
        return inspect_parquet_object(store, key, size)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(key).name
        local.write_bytes(store.read_head(key, head_bytes))
        try:
            report = inspect_file(local)
        except (EOFError, OSError, ReaderError) as exc:
            report = {"file_name": local.name, "error": f"could not read the first {head_bytes} bytes: {exc}"}
    report["key"] = key
    report["size_bytes"] = size
    report["sampled_bytes"] = head_bytes
    return report


def _decompress(path: Path, kind: str) -> Path:
    target = path.with_suffix("")
    part = target.with_name(target.name + ".part")
    if kind == "gzip":
        source = gzip.open(path, "rb")
    else:
        try:
            import zstandard
        except ImportError as exc:
            raise ReaderError(f"{path.name} is zstd-compressed: pip install zstandard") from exc
        source = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(
            open(path, "rb"), read_across_frames=True, closefd=True))
    with source, open(part, "wb") as out:
        shutil.copyfileobj(source, out, 4 * 1024 * 1024)
    part.replace(target)
    path.unlink()
    return target


def fetch_raw(store: Any, prefix: str, out_dir: str | Path, decompress: bool = False,
              log: Callable[[str], None] = print) -> List[Dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for item in list_raw(store, prefix):
        key = item["key"]
        name = key[len(prefix):].lstrip("/") if prefix and key.startswith(prefix) else Path(key).name
        local = out_dir / name
        kind = COMPRESSED_SUFFIXES.get(local.suffix.lower()) if decompress else None
        final = local.with_suffix("") if kind else local
        if final.is_file() and (kind or final.stat().st_size == item["size_bytes"]):
            log(f"  have {final.name}")
        else:
            log(f"  downloading {key} ({item['size_bytes'] / 2**30:.2f} GiB)")
            store.download(key, local)
            if local.stat().st_size != item["size_bytes"]:
                raise ReaderError(f"{key}: downloaded {local.stat().st_size} bytes, R2 has {item['size_bytes']}")
            if kind:
                log(f"  decompressing {local.name}")
                final = _decompress(local, kind)
        fetched.append({"key": key, "r2_size_bytes": item["size_bytes"], "path": str(final),
                        "size_bytes": final.stat().st_size, "sha256": file_sha256(final)})
    return fetched
