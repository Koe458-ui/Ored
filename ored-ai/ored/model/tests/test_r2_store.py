from __future__ import annotations

import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from xml.sax.saxutils import escape

import pytest

from ored.learning import StoreError
from ored.learning.checkpoints import CheckpointStore, ObjectExistsError, digest, upload_verified
from ored.learning.r2 import R2Store, copy_bucket


class FakeS3(BaseHTTPRequestHandler):
    objects: dict = {}
    page = 2

    def log_message(self, *args):
        pass

    def _key(self):
        path = urllib.parse.urlsplit(self.path).path
        bucket, _, key = path.lstrip("/").partition("/")
        return bucket, urllib.parse.unquote(key)

    def _reply(self, status, body=b"", headers=None):
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(body)) if self.command != "HEAD" else
                         (headers or {}).get("x-size", "0"))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _authorized(self):
        auth = self.headers.get("authorization", "")
        if not auth.startswith("AWS4-HMAC-SHA256 Credential=key-id/") or "x-amz-date" not in self.headers:
            self._reply(403, b"<Error>SignatureDoesNotMatch</Error>")
            return False
        return True

    def do_PUT(self):
        if not self._authorized():
            return
        _, key = self._key()
        data = self.rfile.read(int(self.headers["content-length"]))
        if self.headers.get("if-none-match") == "*" and key in self.objects:
            return self._reply(412, b"<Error>PreconditionFailed</Error>")
        self.objects[key] = data
        self._reply(200)

    def do_GET(self):
        if not self._authorized():
            return
        _, key = self._key()
        if key:
            if key not in self.objects:
                return self._reply(404, b"<Error>NoSuchKey</Error>")
            return self._reply(200, self.objects[key])
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
        keys = sorted(k for k in self.objects if k.startswith(query.get("prefix", "")))
        start = int(query.get("continuation-token", "0"))
        chunk = keys[start:start + self.page]
        more = start + self.page < len(keys)
        body = '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        body += "".join(f"<Contents><Key>{escape(k)}</Key><Size>{len(self.objects[k])}</Size></Contents>"
                        for k in chunk)
        body += f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"
        if more:
            body += f"<NextContinuationToken>{start + self.page}</NextContinuationToken>"
        self._reply(200, (body + "</ListBucketResult>").encode())

    def do_HEAD(self):
        if not self._authorized():
            return
        _, key = self._key()
        if key not in self.objects:
            return self._reply(404)
        self._reply(200, headers={"x-size": str(len(self.objects[key]))})

    def do_DELETE(self):
        if not self._authorized():
            return
        _, key = self._key()
        self.objects.pop(key, None)
        self._reply(204)


@pytest.fixture()
def r2():
    FakeS3.objects = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeS3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield R2Store("acct", "key-id", "secret", endpoint=f"http://127.0.0.1:{server.server_port}")
    server.shutdown()


def a_file(tmp_path, size, name="ckpt.pt", seed=1):
    path = tmp_path / name
    path.write_bytes(bytes((seed * 31 + i) % 251 for i in range(size)))
    return path


def test_round_trip_keeps_the_same_name(r2, tmp_path):
    path = a_file(tmp_path, 3000)
    r2.upload(path, "ored_v202/best/epoch_0025_step_00002125.pt")
    assert list(FakeS3.objects) == ["ored_v202/best/epoch_0025_step_00002125.pt"]
    assert r2.stat("ored_v202/best/epoch_0025_step_00002125.pt") == 3000
    out = r2.download("ored_v202/best/epoch_0025_step_00002125.pt", tmp_path / "o" / "x.pt")
    assert out.read_bytes() == path.read_bytes()


def test_no_upsert_refuses_to_overwrite(r2, tmp_path):
    r2.upload(a_file(tmp_path, 10), "run/live/a.pt")
    with pytest.raises(ObjectExistsError):
        r2.upload(a_file(tmp_path, 12, seed=2), "run/live/a.pt")
    r2.upload(a_file(tmp_path, 12, seed=2), "run/live/a.pt", upsert=True)
    assert r2.stat("run/live/a.pt") == 12


def test_missing_objects(r2, tmp_path):
    assert r2.stat("run/live/none.pt") is None
    with pytest.raises(StoreError):
        r2.download("run/live/none.pt", tmp_path / "x.pt")
    with pytest.raises(StoreError):
        r2.remove("run/live/none.pt")


def test_list_is_paged_and_scoped_to_the_folder(r2, tmp_path):
    for name in ["a/1.pt", "a/2.pt", "a/b/3.pt", "ab/4.pt", "c/5.pt"]:
        r2.upload(a_file(tmp_path, 5), name)
    assert set(r2.list_objects("a")) == {"a/1.pt", "a/2.pt", "a/b/3.pt"}
    assert len(r2.list_objects()) == 5
    r2.remove("a/1.pt")
    assert "a/1.pt" not in r2.list_objects("a")


def test_upload_verified_works_against_r2(r2, tmp_path):
    path = a_file(tmp_path, 700)
    upload_verified(r2, path, "run/best/a.pt", digest(path), workdir=tmp_path / "w")
    assert r2.stat("run/best/a.pt") == 700


def test_from_env_picks_r2(monkeypatch):
    monkeypatch.setenv("ORED_R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("ORED_R2_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("ORED_R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("ORED_R2_ENDPOINT", raising=False)
    files = CheckpointStore.from_env()
    assert isinstance(files, R2Store)
    assert files.bucket == "ored-checkpoints"
    assert files.endpoint == "https://acct.r2.cloudflarestorage.com"
    monkeypatch.delenv("ORED_R2_SECRET_ACCESS_KEY")
    with pytest.raises(StoreError):
        CheckpointStore.from_env()


def test_copy_bucket_moves_everything_and_can_rerun(r2, tmp_path):
    from test_checkpoint_parts import HttpBucket

    source = HttpBucket()
    small, big = a_file(tmp_path, 7, "s.pt"), a_file(tmp_path, 25, "b.pt", seed=4)
    source.upload(small, "run/export/s.pt")
    source.upload(big, "run/best/b.pt")
    report = copy_bucket(source, r2, expected={"run/best/b.pt": digest(big)}, workdir=tmp_path)
    assert sorted(report["copied"]) == ["run/best/b.pt", "run/export/s.pt"]
    assert FakeS3.objects["run/best/b.pt"] == big.read_bytes()
    again = copy_bucket(source, r2)
    assert again["copied"] == [] and len(again["skipped"]) == 2


def test_copy_bucket_refuses_a_bad_hash(r2, tmp_path):
    from test_checkpoint_parts import HttpBucket

    source = HttpBucket()
    source.upload(a_file(tmp_path, 7), "run/best/a.pt")
    report = copy_bucket(source, r2, expected={"run/best/a.pt": "0" * 64}, workdir=tmp_path)
    assert report["copied"] == [] and len(report["failed"]) == 1
    assert "run/best/a.pt" not in FakeS3.objects
