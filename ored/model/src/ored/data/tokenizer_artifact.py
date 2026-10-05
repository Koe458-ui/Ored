"""Versioned tokenizer artifacts.

The original language model rebuilds its tokenizer from train.txt on every run and keeps
it inside each checkpoint. A model trained on a large corpus needs the opposite: the
tokenizer is trained once, saved as a named, versioned file, and every dataset shard and
checkpoint records which one it is by sha256. This module is that file format.

    {"format": "ored-tokenizer/1", "name": "ored-bpe-16k", "version": "v1",
     "sha256": "<sha256 of the canonical tokenizer JSON>", "created_at": "...",
     "training": {...free-form provenance...}, "tokenizer": {...Tokenizer.to_dict()...}}

The sha256 is the same digest ored.data.snapshot.tokenizer_digest computes, so an
artifact and a snapshot manifest agree on a tokenizer's identity.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ored.data.tokenizer import Tokenizer

ARTIFACT_FORMAT = "ored-tokenizer/1"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class TokenizerArtifactError(ValueError):
    pass


def tokenizer_sha256(tokenizer: Tokenizer | Dict[str, Any]) -> str:
    data = tokenizer.to_dict() if isinstance(tokenizer, Tokenizer) else tokenizer
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def identity(tokenizer: Tokenizer, name: str = "", version: str = "") -> Dict[str, Any]:
    """What a checkpoint or a shard manifest records about its tokenizer."""
    data = tokenizer.to_dict()
    return {
        "format": ARTIFACT_FORMAT,
        "name": name,
        "version": version,
        "kind": data.get("kind", data["name"]),
        "tokenizer": data["name"],
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": list(data.get("special_tokens", [])),
        "sha256": tokenizer_sha256(data),
    }


def save_artifact(tokenizer: Tokenizer, path: str | Path, name: str, version: str,
                  training: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    for label, value in (("name", name), ("version", version)):
        if not NAME.match(value or ""):
            raise TokenizerArtifactError(f"tokenizer {label} {value!r} must be letters, digits, . _ -")
    path = Path(path)
    data = tokenizer.to_dict()
    artifact = {
        "format": ARTIFACT_FORMAT,
        "name": name,
        "version": version,
        "sha256": tokenizer_sha256(data),
        "vocab_size": tokenizer.vocab_size,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training": dict(training or {}),
        "tokenizer": data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return identity(tokenizer, name, version)


def load_artifact(path: str | Path, expected_sha256: str = "") -> Tuple[Tokenizer, Dict[str, Any]]:
    """Load and verify an artifact. Refuses a file whose content does not match its hash."""
    path = Path(path)
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TokenizerArtifactError(f"{path} is not a readable tokenizer artifact: {exc}") from exc
    if artifact.get("format") != ARTIFACT_FORMAT:
        raise TokenizerArtifactError(f"{path} is not an {ARTIFACT_FORMAT} file "
                                     f"(format {artifact.get('format')!r})")
    actual = tokenizer_sha256(artifact["tokenizer"])
    if actual != artifact.get("sha256"):
        raise TokenizerArtifactError(f"{path} is damaged: its tokenizer hashes to {actual[:16]}, "
                                     f"the file records {str(artifact.get('sha256'))[:16]}")
    if expected_sha256 and actual != expected_sha256:
        raise TokenizerArtifactError(f"{path} is tokenizer {actual[:16]}, expected {expected_sha256[:16]}")
    tokenizer = Tokenizer.from_dict(artifact["tokenizer"])
    return tokenizer, identity(tokenizer, artifact.get("name", ""), artifact.get("version", ""))
