from __future__ import annotations

import gc
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

CHECKPOINT_SCHEMA = "ored-best-checkpoint/1"
FILENAME = "best.pt"
REQUIRED = ("schema", "model_version", "architecture", "model_state_dict", "epoch", "best",
            "tokenizer", "dataset", "config")


class BestCheckpointError(RuntimeError):
    pass


class BestCheckpoint:

    def __init__(self, directory: str | Path, metric: str = "val_loss", mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be min or max")
        self.directory = Path(directory)
        self.metric = metric
        self.mode = mode
        self.best_value: Optional[float] = None
        self.directory.mkdir(parents=True, exist_ok=True)
        self.remove_temporaries()

    @property
    def path(self) -> Path:
        return self.directory / FILENAME

    def remove_temporaries(self) -> None:
        for stale in self.directory.glob(f"{FILENAME}.tmp-*"):
            stale.unlink(missing_ok=True)

    def is_better(self, value: float) -> bool:
        if value is None or not math.isfinite(value):
            return False
        if self.best_value is None:
            return True
        return value < self.best_value if self.mode == "min" else value > self.best_value

    def consider(self, value: float, make_payload: Callable[[], Dict[str, Any]]) -> bool:
        if not self.is_better(value):
            return False
        self.save(make_payload())
        self.best_value = float(value)
        return True

    def _write(self, payload: Dict[str, Any], path: Path) -> None:
        with open(path, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

    def _validate(self, path: Path, payload: Dict[str, Any]) -> None:
        loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        try:
            missing = [k for k in REQUIRED if k not in loaded]
            if missing:
                raise BestCheckpointError(f"written checkpoint lacks {missing}")
            if loaded["schema"] != CHECKPOINT_SCHEMA:
                raise BestCheckpointError("written checkpoint has the wrong schema")
            if set(loaded["model_state_dict"]) != set(payload["model_state_dict"]):
                raise BestCheckpointError("written checkpoint's weights do not match the model")
        finally:
            del loaded
            gc.collect()

    def save(self, payload: Dict[str, Any]) -> Path:
        tmp = self.directory / f"{FILENAME}.tmp-{os.getpid()}"
        try:
            self._write(payload, tmp)
            self._validate(tmp, payload)
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return self.path

    def load(self, map_location: Any = "cpu") -> Dict[str, Any]:
        if not self.path.is_file():
            raise BestCheckpointError(f"no checkpoint at {self.path}")
        payload = torch.load(self.path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise BestCheckpointError(
                f"{self.path} is not an {CHECKPOINT_SCHEMA} checkpoint (an older Ored model's "
                f"checkpoint cannot be loaded into this model)")
        best = payload.get("best") or {}
        if best.get("metric") == self.metric and best.get("mode") == self.mode:
            self.best_value = best.get("value")
        return payload


def check_compatible(payload: Dict[str, Any], *, model_version: str, architecture: Dict[str, Any],
                     tokenizer_sha256: str, dataset_sha256: Optional[str] = None,
                     path: str | Path = FILENAME) -> None:
    problems = []
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        problems.append(f"schema {payload.get('schema')!r}, expected {CHECKPOINT_SCHEMA}")
    if payload.get("model_version") != model_version:
        problems.append(f"model version {payload.get('model_version')!r}, this config is {model_version!r}")
    if payload.get("architecture") != architecture:
        problems.append(f"architecture {payload.get('architecture')}, this config builds {architecture}")
    recorded = (payload.get("tokenizer") or {}).get("sha256")
    if recorded != tokenizer_sha256:
        problems.append(f"tokenizer {str(recorded)[:16]}, this run uses {tokenizer_sha256[:16]}")
    if dataset_sha256 is not None:
        recorded = (payload.get("dataset") or {}).get("content_sha256")
        if recorded != dataset_sha256:
            problems.append(f"dataset {str(recorded)[:16]}, this run uses {dataset_sha256[:16]}")
    if problems:
        raise BestCheckpointError(f"{path} does not fit this run: " + "; ".join(problems))
