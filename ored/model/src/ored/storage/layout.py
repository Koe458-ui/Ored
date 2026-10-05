"""The versioned object layout in the R2 bucket.

    ored-ai/
      datasets/<dataset-name>/<version>/manifest.json         (raw-file or token-shard manifest)
      datasets/<dataset-name>/<version>/<file>                (raw files, e.g. the source corpus)
      datasets/<dataset-name>/<version>/tokens/<tokenizer>/   (token shards + their manifest)
      tokenizers/<tokenizer-name>/<version>/tokenizer.json
      checkpoints/<model>/best.pt                             (the only checkpoint object per model)

Every segment is checked, so a name can never climb out of its folder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_PREFIX = "ored-ai"
SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")


class LayoutError(ValueError):
    pass


def _segment(label: str, value: str) -> str:
    if not SEGMENT.match(value or "") or ".." in value:
        raise LayoutError(f"{label} {value!r} must be letters, digits, . _ - (no slashes, no '..')")
    return value


@dataclass(frozen=True)
class ObjectLayout:
    prefix: str = DEFAULT_PREFIX

    def __post_init__(self) -> None:
        for part in self.prefix.strip("/").split("/"):
            _segment("prefix", part)

    def _join(self, *parts: str) -> str:
        return "/".join([self.prefix.strip("/"), *parts])

    def dataset_dir(self, name: str, version: str) -> str:
        return self._join("datasets", _segment("dataset name", name), _segment("dataset version", version))

    def dataset_manifest(self, name: str, version: str) -> str:
        return f"{self.dataset_dir(name, version)}/manifest.json"

    def dataset_file(self, name: str, version: str, file_name: str) -> str:
        if not FILE_NAME.match(file_name or "") or ".." in file_name or "//" in file_name:
            raise LayoutError(f"file name {file_name!r} is not a safe relative path")
        return f"{self.dataset_dir(name, version)}/{file_name}"

    def token_dir(self, name: str, version: str, tokenizer: str) -> str:
        return f"{self.dataset_dir(name, version)}/tokens/{_segment('tokenizer', tokenizer)}"

    def tokenizer_artifact(self, name: str, version: str) -> str:
        return self._join("tokenizers", _segment("tokenizer name", name),
                          _segment("tokenizer version", version), "tokenizer.json")

    def checkpoint_best(self, model: str) -> str:
        return self._join("checkpoints", _segment("model", model), "best.pt")
