"""Object storage for large training files (Cloudflare R2, through its S3 API).

Supabase keeps only metadata (ored_datasets rows); every large file -- raw corpora,
token shards, manifests, tokenizer artifacts and checkpoints -- lives in R2 under the
layout in ored.storage.layout.
"""
from ored.storage.layout import ObjectLayout
from ored.storage.settings import R2Settings, StorageConfigError

__all__ = ["ObjectLayout", "R2Settings", "StorageConfigError"]
