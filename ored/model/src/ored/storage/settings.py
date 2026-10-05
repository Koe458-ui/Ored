"""R2 credentials and bucket, read from the environment only.

    ORED_R2_ACCOUNT_ID         Cloudflare account id (or set ORED_R2_ENDPOINT instead)
    ORED_R2_ENDPOINT           optional; default https://<account>.r2.cloudflarestorage.com
    ORED_R2_ACCESS_KEY_ID      R2 API token's access key id
    ORED_R2_SECRET_ACCESS_KEY  R2 API token's secret
    ORED_R2_BUCKET             bucket for the ored-ai/ object layout
    ORED_R2_PREFIX             optional; default "ored-ai"

These are the names the existing checkpoint code (ored.learning.r2) already uses, plus
ORED_R2_BUCKET / ORED_R2_PREFIX for the versioned layout. Nothing here is ever printed:
repr() and describe() show whether a secret is set, never its value.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ored.storage.layout import DEFAULT_PREFIX

BUCKET = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")


class StorageConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class R2Settings:
    bucket: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    account_id: str = field(default="", repr=False)
    endpoint: str = ""
    prefix: str = DEFAULT_PREFIX

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "R2Settings":
        env = os.environ if environ is None else environ
        get = lambda name: (env.get(name) or "").strip()  # noqa: E731
        missing = [name for name in ("ORED_R2_ACCESS_KEY_ID", "ORED_R2_SECRET_ACCESS_KEY", "ORED_R2_BUCKET")
                   if not get(name)]
        if not get("ORED_R2_ACCOUNT_ID") and not get("ORED_R2_ENDPOINT"):
            missing.append("ORED_R2_ACCOUNT_ID (or ORED_R2_ENDPOINT)")
        if missing:
            raise StorageConfigError("R2 is not configured; set " + ", ".join(missing)
                                     + " (see ored/model/.env.example)")
        settings = cls(bucket=get("ORED_R2_BUCKET"), access_key_id=get("ORED_R2_ACCESS_KEY_ID"),
                       secret_access_key=get("ORED_R2_SECRET_ACCESS_KEY"),
                       account_id=get("ORED_R2_ACCOUNT_ID"), endpoint=get("ORED_R2_ENDPOINT"),
                       prefix=get("ORED_R2_PREFIX") or DEFAULT_PREFIX)
        settings.validate()
        return settings

    @staticmethod
    def configured(environ: Optional[Mapping[str, str]] = None) -> bool:
        try:
            R2Settings.from_env(environ)
        except StorageConfigError:
            return False
        return True

    def validate(self) -> None:
        if not BUCKET.match(self.bucket):
            raise StorageConfigError(f"ORED_R2_BUCKET {self.bucket!r} is not a valid bucket name")
        if self.endpoint and not self.endpoint.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise StorageConfigError("ORED_R2_ENDPOINT must be an https:// URL")

    def describe(self) -> Dict[str, Any]:
        """Safe to log."""
        return {"bucket": self.bucket, "prefix": self.prefix,
                "endpoint": self.endpoint or "https://<account>.r2.cloudflarestorage.com",
                "account_id": "set" if self.account_id else "unset",
                "access_key_id": "set", "secret_access_key": "set"}

    def store(self) -> Any:
        """An ored.learning.r2.R2Store for this bucket (the repository's S3 client)."""
        from ored.learning.r2 import R2Store
        return R2Store(self.account_id or "-", self.access_key_id, self.secret_access_key,
                       self.bucket, endpoint=self.endpoint)
