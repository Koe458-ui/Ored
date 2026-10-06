from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from ored.data.readers import file_sha256
from ored.data.token_shards import MANIFEST, load_manifest, verify_shards
from ored.learning.store import StoreError


def replace_object(store: Any, path: str | Path, object_path: str) -> Dict[str, Any]:
    path = Path(path)
    size = path.stat().st_size
    store.upload(path, object_path, upsert=True)
    stored = store.stat(object_path)
    if stored != size:
        raise StoreError(f"r2://{object_path} is {stored} bytes after upload, {size} locally")
    return {"object_path": object_path, "size_bytes": size}


def publish_token_dataset(store: Any, local_dir: str | Path, object_dir: str,
                          log: Callable[[str], None] = print) -> Dict[str, Any]:
    local_dir = Path(local_dir)
    manifest = load_manifest(local_dir)
    verify_shards(local_dir, manifest, "full")
    from ored.learning.checkpoints import ObjectExistsError
    for shard in manifest["shards"]:
        target = f"{object_dir}/{shard['path']}"
        try:
            store.upload(local_dir / shard["path"], target, upsert=False)
        except ObjectExistsError:
            if store.stat(target) != shard["bytes"]:
                raise StoreError(f"r2://{target} already exists with different content")
        log(f"  uploaded {shard['path']}")
    store.upload(local_dir / MANIFEST, f"{object_dir}/{MANIFEST}", upsert=False)
    return {"object_dir": object_dir, "manifest_sha256": file_sha256(local_dir / MANIFEST),
            "content_sha256": manifest["content_sha256"]}


def fetch_token_dataset(store: Any, object_dir: str, local_dir: str | Path, verify: str = "full",
                        log: Callable[[str], None] = print) -> Dict[str, Any]:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = local_dir / MANIFEST
    if not manifest_path.is_file():
        store.download(f"{object_dir}/{MANIFEST}", manifest_path.with_suffix(".json.part"))
        manifest_path.with_suffix(".json.part").replace(manifest_path)
    manifest = load_manifest(local_dir)
    for shard in manifest["shards"]:
        path = local_dir / shard["path"]
        if path.is_file() and path.stat().st_size == shard["bytes"] and file_sha256(path) == shard["sha256"]:
            continue
        store.download(f"{object_dir}/{shard['path']}", path)
        log(f"  downloaded {shard['path']}")
    verify_shards(local_dir, manifest, verify)
    return manifest

