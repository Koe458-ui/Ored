from __future__ import annotations

import json
import urllib.parse

import pytest

from ored.learning import StoreError
from ored.learning.checkpoints import CheckpointStore, ObjectExistsError, digest, upload_verified

LIMIT = 10


class HttpBucket(CheckpointStore):
    """Speaks the Storage REST API through _send and rejects objects over LIMIT bytes."""

    def __init__(self, max_object_bytes=LIMIT):
        super().__init__("https://example.supabase.co", "key", "ored-checkpoints",
                         max_object_bytes=max_object_bytes)
        self.objects = {}

    def _path(self, url):
        prefix = f"{self.base}/object/ored-checkpoints/"
        return urllib.parse.unquote(url[len(prefix):])

    def _send(self, method, url, data=None, headers=None):
        if url.startswith(f"{self.base}/object/list/"):
            body = json.loads(data)
            folder = body["prefix"].strip("/")
            start = f"{folder}/" if folder else ""
            items, dirs = [], set()
            for path, blob in sorted(self.objects.items()):
                if not path.startswith(start):
                    continue
                rest = path[len(start):]
                if "/" in rest:
                    dirs.add(rest.split("/")[0])
                elif rest.startswith(body["search"]):
                    items.append({"name": rest, "id": "x", "metadata": {"size": len(blob)}})
            items += [{"name": d, "id": None} for d in sorted(dirs)]
            return json.dumps(items).encode()
        path = self._path(url)
        if method == "POST":
            if len(data) > LIMIT:
                raise StoreError(f"POST {url} -> 400 EntityTooLarge")
            if path in self.objects and headers["x-upsert"] != "true":
                raise ObjectExistsError(f"POST {url} -> 409 Duplicate")
            self.objects[path] = bytes(data)
            return b"{}"
        if path not in self.objects:
            raise StoreError(f"{method} {url} -> 404 not found")
        if method == "GET":
            return self.objects[path]
        if method == "DELETE":
            del self.objects[path]
            return b"{}"
        raise AssertionError(method)


def a_file(tmp_path, size, name="ckpt.pt", seed=1):
    path = tmp_path / name
    path.write_bytes(bytes((seed * 31 + i) % 251 for i in range(size)))
    return path


def test_small_file_stays_one_object(tmp_path):
    bucket = HttpBucket()
    bucket.upload(a_file(tmp_path, 7), "run/best/a.pt")
    assert list(bucket.objects) == ["run/best/a.pt"]
    assert bucket.stat("run/best/a.pt") == 7


def test_large_file_is_split_and_reassembled(tmp_path):
    bucket = HttpBucket()
    path = a_file(tmp_path, 25)
    result = bucket.upload(path, "run/best/a.pt")
    assert result["size_bytes"] == 25
    assert sorted(bucket.objects) == [f"run/best/a.pt.part0000{i}" for i in range(3)]
    assert bucket.stat("run/best/a.pt") == 25
    assert bucket.list_objects("run") == {"run/best/a.pt": 25}
    landed = bucket.download("run/best/a.pt", tmp_path / "out" / "a.pt")
    assert landed.read_bytes() == path.read_bytes()


def test_split_upload_respects_upsert_and_clears_stale_parts(tmp_path):
    bucket = HttpBucket()
    bucket.upload(a_file(tmp_path, 35), "run/best/a.pt")
    with pytest.raises(ObjectExistsError):
        bucket.upload(a_file(tmp_path, 25, seed=2), "run/best/a.pt")
    bucket.upload(a_file(tmp_path, 25, seed=2), "run/best/a.pt", upsert=True)
    assert len(bucket.objects) == 3
    assert bucket.stat("run/best/a.pt") == 25
    bucket.upload(a_file(tmp_path, 5, seed=3), "run/best/a.pt", upsert=True)
    assert list(bucket.objects) == ["run/best/a.pt"]


def test_remove_deletes_every_part_and_404s_when_missing(tmp_path):
    bucket = HttpBucket()
    bucket.upload(a_file(tmp_path, 25), "run/best/a.pt")
    bucket.upload(a_file(tmp_path, 25), "run/best/b.pt")
    bucket.remove("run/best/a.pt")
    assert bucket.stat("run/best/a.pt") is None
    assert bucket.stat("run/best/b.pt") == 25
    with pytest.raises(StoreError):
        bucket.remove("run/best/a.pt")


def test_upload_verified_round_trips_a_split_checkpoint(tmp_path):
    bucket = HttpBucket()
    path = a_file(tmp_path, 42)
    upload_verified(bucket, path, "run/best/a.pt", digest(path), workdir=tmp_path / "w")
    assert bucket.stat("run/best/a.pt") == 42


def test_limit_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ORED_SB_URL", "https://example.supabase.co")
    monkeypatch.setenv("ORED_SB_SERVICE_KEY", "key")
    monkeypatch.setenv("ORED_SB_MAX_OBJECT_MB", "20")
    assert CheckpointStore.from_env().max_object_bytes == 20 * 1024 * 1024
    monkeypatch.setenv("ORED_SB_MAX_OBJECT_MB", "zero")
    with pytest.raises(StoreError):
        CheckpointStore.from_env()
