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
    checkpoint_bucket: str = ""

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "R2Settings":
        env = os.environ if environ is None else environ
        get = lambda name: (env.get(name) or "").strip()
        missing = [name for name in ("ORED_R2_ACCESS_KEY_ID", "ORED_R2_SECRET_ACCESS_KEY", "ORED_R2_BUCKET")
                   if not get(name)]
        if not get("ORED_R2_ACCOUNT_ID") and not get("ORED_R2_ENDPOINT"):
            missing.append("ORED_R2_ACCOUNT_ID (or ORED_R2_ENDPOINT)")
        if missing:
            raise StorageConfigError("R2 is not configured; set " + ", ".join(missing))
        settings = cls(bucket=get("ORED_R2_BUCKET"), access_key_id=get("ORED_R2_ACCESS_KEY_ID"),
                       secret_access_key=get("ORED_R2_SECRET_ACCESS_KEY"),
                       account_id=get("ORED_R2_ACCOUNT_ID"), endpoint=get("ORED_R2_ENDPOINT"),
                       prefix=get("ORED_R2_PREFIX") or DEFAULT_PREFIX,
                       checkpoint_bucket=get("ORED_R2_CHECKPOINT_BUCKET"))
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
        if self.checkpoint_bucket and not BUCKET.match(self.checkpoint_bucket):
            raise StorageConfigError(f"ORED_R2_CHECKPOINT_BUCKET {self.checkpoint_bucket!r} is not a valid bucket name")
        if self.endpoint and not self.endpoint.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise StorageConfigError("ORED_R2_ENDPOINT must be an https:// URL")

    def describe(self) -> Dict[str, Any]:
        return {"bucket": self.bucket, "checkpoint_bucket": self.checkpoint_bucket or self.bucket,
                "prefix": self.prefix,
                "endpoint": self.endpoint or "https://<account>.r2.cloudflarestorage.com",
                "account_id": "set" if self.account_id else "unset",
                "access_key_id": "set", "secret_access_key": "set"}

    def store(self, bucket: str = "") -> Any:
        from ored.learning.r2 import R2Store
        return R2Store(self.account_id or "-", self.access_key_id, self.secret_access_key,
                       bucket or self.bucket, endpoint=self.endpoint)

    def checkpoint_store(self) -> Any:
        return self.store(self.checkpoint_bucket)
