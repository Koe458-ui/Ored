from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ored.config import Config
from ored.models.registry import build_model

TASK_REGISTRY: Dict[str, Type["Task"]] = {}


def register_task(name: str) -> Callable[[Type["Task"]], Type["Task"]]:
    def decorator(cls: Type["Task"]) -> Type["Task"]:
        key = name.lower()
        if key in TASK_REGISTRY:
            raise ValueError(f"task {name!r} is already registered")
        TASK_REGISTRY[key] = cls
        cls.name = key
        return cls

    return decorator


class Task:

    name: str = "base"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.criterion: nn.Module = nn.Identity()

    def build_data(self, generator: torch.Generator | None = None
                   ) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
        raise NotImplementedError

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def compute_loss(
        self, model: nn.Module, batch: Tuple[torch.Tensor, ...], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float], int]:
        raise NotImplementedError

    @property
    def metric_names(self) -> List[str]:
        return []

    def describe_data(self, datasets: Dict[str, Any]) -> List[str]:
        return []

    def checkpoint_extra(self) -> Dict[str, Any]:
        return {}

    def next_steps(self) -> List[str]:
        return []


@register_task("bit_addition")
class BitAdditionTask(Task):

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        from ored.training.trainer import build_criterion
        self.criterion = build_criterion(cfg.training.loss)

    def build_data(self, generator=None):
        from ored.data.dataset import build_dataloaders
        loaders, datasets = build_dataloaders(self.cfg, generator=generator)
        return loaders, datasets

    def build_model(self) -> nn.Module:
        return build_model(self.cfg)

    def compute_loss(self, model, batch, device):
        from ored.training.metrics import bit_accuracy, exact_match_accuracy

        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = self.criterion(logits, targets)

        metrics = {
            "bit_acc": bit_accuracy(logits, targets),
            "exact_acc": exact_match_accuracy(logits, targets),
        }
        return loss, metrics, inputs.size(0)

    @property
    def metric_names(self) -> List[str]:
        return ["bit_acc", "exact_acc"]

    def describe_data(self, datasets):
        return [datasets[name].describe() for name in ("train", "val", "test")]

    def next_steps(self):
        return [
            f"python scripts/evaluate.py --checkpoint {self.cfg.checkpoint_dir}/best.pt",
            "python scripts/infer.py --a 9 --b 6",
        ]


@register_task("language_model")
class LanguageModelTask(Task):

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = None
        self.chars_per_token = 1.0
        self.dataset: Dict[str, Any] = {}

    def build_data(self, generator=None):
        from ored.data.text_dataset import build_text_dataloaders

        loaders, datasets, tokenizer = build_text_dataloaders(self.cfg, generator=generator)
        self.tokenizer = tokenizer
        self.chars_per_token = datasets["train"].chars_per_token
        if self.cfg.data.source == "supabase":
            self.dataset = self._snapshot_summary(tokenizer)
        else:
            from ored.data.snapshot import generated_corpus_summary
            self.dataset = generated_corpus_summary(self.cfg, tokenizer.to_dict())
        return loaders, datasets

    def _snapshot_summary(self, tokenizer) -> Dict[str, Any]:
        from ored.data.snapshot import SnapshotError, load_snapshot, tokenizer_digest

        snapshot = load_snapshot(self.cfg, verify=False)
        built = tokenizer_digest(tokenizer.to_dict())
        recorded = snapshot.manifest["tokenizer"]
        intended = snapshot.manifest.get("intended") or {}
        same_settings = recorded["name"] == self.cfg.data.tokenizer.lower() and (
            recorded["name"] == "char" or intended.get("vocab_size") == self.cfg.data.vocab_size)
        summary = snapshot.summary()
        if not same_settings:
            summary["tokenizer"] = {"name": tokenizer.name, "vocab_size": tokenizer.vocab_size,
                                    "sha256": built}
        elif built != recorded["sha256"]:
            raise SnapshotError(
                f"the tokenizer built from snapshot {snapshot.sha256[:16]} ({built[:16]}) differs from the "
                f"one its manifest records ({recorded['sha256'][:16]})")
        return summary

    def build_model(self) -> nn.Module:
        if self.tokenizer is None:
            raise RuntimeError("build_data() must run before build_model(): the model's "
                               "output width is the tokenizer's vocabulary size")
        return build_model(self.cfg, vocab_size=self.tokenizer.vocab_size)

    def compute_loss(self, model, batch, device):
        ids, targets = batch
        ids = ids.to(device)
        targets = targets.to(device)

        logits = model(ids)
        B, T, V = logits.shape

        loss = self.criterion(logits.reshape(B * T, V), targets.reshape(B * T))

        loss_value = loss.item()
        metrics = {
            "bpc": loss_value / math.log(2) / self.chars_per_token,
            "ppl": math.exp(min(loss_value, 20.0)),
        }
        return loss, metrics, B

    @property
    def metric_names(self) -> List[str]:
        return ["bpc"]

    def describe_data(self, datasets):
        lines = [f"{name:<5} split: {datasets[name].describe()}"
                 for name in ("train", "val", "test")]
        if self.tokenizer is not None:
            lines.append(f"tokenizer  : {self.tokenizer.describe()}")
        if self.dataset.get("source") == "supabase":
            lines.append(f"corpus     : Supabase snapshot {self.dataset['snapshot_sha256'][:16]} of "
                         f"{self.dataset['dataset_tag']!r}, {self.dataset['records']:,} records "
                         f"(ored_training_data)")
            fetched = self.dataset.get("rows_fetched")
            if fetched is not None:
                counts = self.dataset.get("split_counts") or {}
                lines.append(f"rows       : {fetched:,} fetched from ored_training_data -> "
                             f"{self.dataset['records']:,} trained on "
                             f"(train {counts.get('train', 0):,} / val {counts.get('val', 0):,} / "
                             f"test {counts.get('test', 0):,})")
        else:
            sha = str(self.dataset.get("corpus_sha256") or "")[:16]
            lines.append(f"corpus     : generated, {self.cfg.data.corpus.dir} (sha256 {sha})")
            lines.append("WARNING    : this run trains on the GENERATED corpus, not on the rows in "
                         "ored_training_data. Drop --generated-corpus / data.source=generated to "
                         "train on Supabase data.")
        return lines

    def checkpoint_extra(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"tokenizer": self.tokenizer.to_dict()} if self.tokenizer else {}
        if self.dataset:
            extra["dataset"] = dict(self.dataset)
        return extra

    def next_steps(self):
        return [
            f"python scripts/evaluate.py --checkpoint {self.cfg.checkpoint_dir}/best.pt",
            'python scripts/generate.py --prompt "the "',
            'python scripts/generate.py --arithmetic',
        ]


def build_task(cfg: Config) -> Task:
    key = cfg.task.lower()
    if key not in TASK_REGISTRY:
        raise ValueError(f"unknown task {cfg.task!r}. Registered tasks: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[key](cfg)
