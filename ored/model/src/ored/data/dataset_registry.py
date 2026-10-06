from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

STATUSES = ("registered", "uploading", "uploaded", "processing", "ready", "failed", "deprecated")
STORAGE_PROVIDERS = ("r2", "supabase_storage", "local")
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 256 * 1024

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SLUG = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
FORMAT = re.compile(r"^[a-z0-9][a-z0-9_.+-]{0,31}$")
BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,62}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RegistryError(ValueError):
    pass


@dataclass
class DatasetEntry:
    name: str
    version_label: str
    source: str = "external"
    kind: str = "text"
    summary: str = ""
    dataset_type: Optional[str] = None
    status: str = "registered"
    storage_provider: Optional[str] = "r2"
    storage_bucket: Optional[str] = None
    storage_path: Optional[str] = None
    file_name: Optional[str] = None
    file_format: Optional[str] = None
    compression: Optional[str] = None
    external_id: Optional[str] = None
    source_id: Optional[str] = None
    size_bytes: Optional[int] = None
    document_count: Optional[int] = None
    token_count: Optional[int] = None
    sha256: Optional[str] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "DatasetEntry":
        problems = []
        if not NAME.match(self.name or "") or self.name == "all":
            problems.append(f"name {self.name!r} must be letters, digits, _ . - (not 'all')")
        if not LABEL.match(self.version_label or ""):
            problems.append(f"version_label {self.version_label!r} must be letters, digits, . _ -")
        if self.source != "external":
            problems.append("registry entries are source = 'external'")
        if self.kind not in ("text", "tabular"):
            problems.append("kind must be text or tabular")
        if self.status not in STATUSES:
            problems.append(f"status must be one of {', '.join(STATUSES)}")
        if self.storage_provider is not None and self.storage_provider not in STORAGE_PROVIDERS:
            problems.append(f"storage_provider must be one of {', '.join(STORAGE_PROVIDERS)}")
        if self.storage_bucket is not None and not BUCKET.match(self.storage_bucket):
            problems.append(f"storage_bucket {self.storage_bucket!r} is not a bucket name")
        if self.storage_path is not None and (len(self.storage_path) > 1024 or self.storage_path.startswith("/")
                                              or ".." in self.storage_path.split("/")
                                              or "//" in self.storage_path):
            problems.append("storage_path must be a relative object path without '..' or '//'")
        if self.dataset_type is not None and not SLUG.match(self.dataset_type):
            problems.append("dataset_type must be lowercase letters, digits and _")
        for name in ("file_format", "compression"):
            value = getattr(self, name)
            if value is not None and not FORMAT.match(value):
                problems.append(f"{name} {value!r} must be a short lowercase name, e.g. jsonl, parquet, gzip")
        for name in ("external_id", "source_id", "file_name"):
            value = getattr(self, name)
            if value is not None and not 1 <= len(value) <= 512:
                problems.append(f"{name} must be 1-512 characters")
        for name in ("size_bytes", "document_count", "token_count"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                problems.append(f"{name} must be a non-negative integer or null")
        if self.sha256 is not None and not SHA256.match(self.sha256):
            problems.append("sha256 must be 64 lowercase hex characters")
        if self.status == "ready" and self.sha256 is None:
            problems.append("a ready dataset needs its sha256")
        for name, limit in (("manifest", MAX_MANIFEST_BYTES), ("metadata", MAX_METADATA_BYTES)):
            value = getattr(self, name)
            if not isinstance(value, dict):
                problems.append(f"{name} must be a JSON object")
            elif len(json.dumps(value).encode("utf-8")) > limit:
                problems.append(f"{name} is larger than {limit} bytes: it is metadata, the data belongs in R2")
        if problems:
            raise RegistryError("; ".join(problems))
        return self

    def to_row(self) -> Dict[str, Any]:
        return asdict(self.validate())
