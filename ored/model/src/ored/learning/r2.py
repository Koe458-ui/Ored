from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ored.learning.checkpoints import (
    DEFAULT_BUCKET,
    TIMEOUT_SECONDS,
    CheckpointStore,
    ObjectExistsError,
    digest,
)
from ored.learning.store import StoreError

# One PUT to R2 takes at most 5 GiB; bigger files would need a multipart upload.
MAX_PUT_BYTES = 5 * 1024 * 1024 * 1024
DOWNLOAD_CHUNK = 4 * 1024 * 1024
REGION = "auto"
SERVICE = "s3"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def r2_configured() -> bool:
    return bool(os.environ.get("ORED_R2_ACCOUNT_ID", "").strip())


def _quote(value: str, safe: str = "-_.~") -> str:
    return urllib.parse.quote(value, safe=safe)


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


class R2Store(CheckpointStore):
    """Checkpoint files in a Cloudflare R2 bucket, through its S3-compatible API.

    Object paths are the same as in Supabase Storage, so the rows in Postgres keep
    pointing at the right files after a move.
    """

    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str = DEFAULT_BUCKET,
        timeout: int = TIMEOUT_SECONDS,
        endpoint: str = "",
    ) -> None:
        if not account_id or not access_key_id or not secret_access_key:
            raise StoreError("R2Store needs an account id, an access key id and a secret access key")
        self.endpoint = (endpoint or f"https://{account_id}.r2.cloudflarestorage.com").rstrip("/")
        self.host = urllib.parse.urlsplit(self.endpoint).netloc
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.bucket = bucket
        self.timeout = timeout
        self.max_object_bytes = MAX_PUT_BYTES

    @classmethod
    def from_env(cls, bucket: str = DEFAULT_BUCKET) -> "R2Store":
        account = os.environ.get("ORED_R2_ACCOUNT_ID", "").strip()
        key_id = os.environ.get("ORED_R2_ACCESS_KEY_ID", "").strip()
        secret = os.environ.get("ORED_R2_SECRET_ACCESS_KEY", "").strip()
        if not account or not key_id or not secret:
            raise StoreError(
                "set ORED_R2_ACCOUNT_ID, ORED_R2_ACCESS_KEY_ID and ORED_R2_SECRET_ACCESS_KEY "
                "to reach checkpoint storage on Cloudflare R2"
            )
        return cls(account, key_id, secret, bucket,
                   endpoint=os.environ.get("ORED_R2_ENDPOINT", "").strip())

    # -- signing -------------------------------------------------------------

    def _key_url(self, object_path: str) -> str:
        return f"/{_quote(self.bucket)}/{_quote(object_path.strip('/'), safe='-_.~/')}"

    def _signed_headers(self, method: str, path: str, query: List[Tuple[str, str]],
                        payload_hash: str, extra: Dict[str, str]) -> Dict[str, str]:
        now = dt.datetime.now(dt.timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        day = now.strftime("%Y%m%d")
        headers = {k.lower(): v.strip() for k, v in extra.items()}
        headers.update({"host": self.host, "x-amz-date": amz_date, "x-amz-content-sha256": payload_hash})
        names = sorted(headers)
        canonical = "\n".join([
            method,
            path,
            "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(query)),
            "".join(f"{n}:{headers[n]}\n" for n in names),
            ";".join(names),
            payload_hash,
        ])
        scope = f"{day}/{REGION}/{SERVICE}/aws4_request"
        to_sign = "\n".join([
            "AWS4-HMAC-SHA256", amz_date, scope,
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        ])
        key = _sign(("AWS4" + self.secret_access_key).encode("utf-8"), day)
        for part in (REGION, SERVICE, "aws4_request"):
            key = _sign(key, part)
        signature = hmac.new(key, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        headers["authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self.access_key_id}/{scope}, "
            f"SignedHeaders={';'.join(names)}, Signature={signature}"
        )
        del headers["host"]
        return headers

    def _request(self, method: str, path: str, query: Optional[List[Tuple[str, str]]] = None,
                 data: bytes = b"", headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, str], bytes]:
        query = query or []
        signed = self._signed_headers(method, path, query, hashlib.sha256(data).hexdigest(), headers or {})
        url = self.endpoint + path
        if query:
            url += "?" + "&".join(f"{_quote(k)}={_quote(v)}" for k, v in query)
        request = urllib.request.Request(url, data=data if method == "PUT" else None,
                                         method=method, headers=signed)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code == 404:
                return 404, dict(exc.headers or {}), b""
            if exc.code == 412:
                raise ObjectExistsError(f"{method} {path} -> 412 already exists") from exc
            raise StoreError(f"{method} r2://{path.lstrip('/')} -> {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise StoreError(f"{method} {self.endpoint} could not be reached: {exc.reason}") from exc

    # -- CheckpointStore -----------------------------------------------------

    def upload(self, path: str | Path, object_path: str, upsert: bool = False) -> Dict[str, Any]:
        path = Path(path)
        if not path.is_file():
            raise StoreError(f"checkpoint not found: {path}")
        size = path.stat().st_size
        if size > MAX_PUT_BYTES:
            raise StoreError(f"{path} is {size} bytes; one R2 upload takes at most {MAX_PUT_BYTES}")
        headers = {"content-type": "application/octet-stream"}
        if not upsert:
            headers["if-none-match"] = "*"
        self._request("PUT", self._key_url(object_path), data=path.read_bytes(), headers=headers)
        return {"object_path": object_path, "size_bytes": size, "sha256": digest(path)}

    def download(self, object_path: str, path: str | Path) -> Path:
        """Stream the object to disk in chunks (a dataset shard or checkpoint never has to
        fit in memory), then move it into place."""
        path = Path(path)
        key = self._key_url(object_path)
        signed = self._signed_headers("GET", key, [], hashlib.sha256(b"").hexdigest(), {})
        request = urllib.request.Request(self.endpoint + key, method="GET", headers=signed)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response, open(tmp, "wb") as out:
                shutil.copyfileobj(response, out, DOWNLOAD_CHUNK)
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            if exc.code == 404:
                raise StoreError(f"GET r2://{self.bucket}/{object_path} -> 404 not found") from exc
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise StoreError(f"GET r2://{self.bucket}/{object_path} -> {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            tmp.unlink(missing_ok=True)
            raise StoreError(f"GET {self.endpoint} could not be reached: {exc.reason}") from exc
        tmp.replace(path)
        return path

    def remove(self, object_path: str) -> None:
        if self.stat(object_path) is None:
            raise StoreError(f"DELETE r2://{self.bucket}/{object_path} -> 404 not found")
        self._request("DELETE", self._key_url(object_path))

    def stat(self, object_path: str) -> Optional[int]:
        status, headers, _ = self._request("HEAD", self._key_url(object_path))
        if status == 404:
            return None
        lowered = {k.lower(): v for k, v in headers.items()}
        return int(lowered.get("content-length") or 0)

    def list_objects(self, prefix: str = "") -> Dict[str, int]:
        folder = prefix.strip("/")
        base = [("list-type", "2")] + ([("prefix", folder + "/")] if folder else [])
        found: Dict[str, int] = {}
        token = ""
        while True:
            query = base + ([("continuation-token", token)] if token else [])
            _, _, body = self._request("GET", f"/{_quote(self.bucket)}", query=query)
            root = ET.fromstring(body)
            for item in root.iter(f"{S3_NS}Contents"):
                found[item.findtext(f"{S3_NS}Key") or ""] = int(item.findtext(f"{S3_NS}Size") or 0)
            if (root.findtext(f"{S3_NS}IsTruncated") or "").lower() != "true":
                return found
            token = root.findtext(f"{S3_NS}NextContinuationToken") or ""


def copy_bucket(source: CheckpointStore, target: CheckpointStore, dry_run: bool = False,
                expected: Optional[Dict[str, str]] = None,
                workdir: Optional[str | Path] = None) -> Dict[str, List[str]]:
    """Copy every object from source to target under the same path.

    Objects the target already holds at the same size are skipped, so an
    interrupted copy can simply be run again. ``expected`` maps object paths to
    the sha256 their database row records; a copy that does not match is refused.
    """
    import tempfile

    report: Dict[str, List[str]] = {"copied": [], "skipped": [], "failed": []}
    expected = expected or {}
    for object_path, size in sorted(source.list_objects().items()):
        if target.stat(object_path) == size:
            report["skipped"].append(object_path)
            continue
        if dry_run:
            report["copied"].append(object_path)
            continue
        try:
            with tempfile.TemporaryDirectory(dir=workdir) as tmp:
                local = source.download(object_path, Path(tmp) / "object")
                if local.stat().st_size != size:
                    raise StoreError(f"downloaded {local.stat().st_size} bytes, listed {size}")
                want = expected.get(object_path)
                if want and digest(local) != want:
                    raise StoreError("sha256 does not match its database row")
                target.upload(local, object_path, upsert=True)
            if target.stat(object_path) != size:
                raise StoreError("size in R2 does not match after upload")
            report["copied"].append(object_path)
        except StoreError as exc:
            report["failed"].append(f"{object_path}: {exc}")
    return report
