from __future__ import annotations

import argparse
import inspect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from ored.config import Config, load_config
from ored.data.token_shards import (
    ShardError,
    TokenWindowDataset,
    dataset_identity,
    load_manifest,
    manifest_file_sha256,
    verify_shards,
)
from ored.data.tokenizer import Tokenizer
from ored.data.tokenizer_artifact import load_artifact
from ored.models.registry import build_model
from ored.training import gpu
from ored.training.best_checkpoint import CHECKPOINT_SCHEMA, BestCheckpoint, check_compatible
from ored.training.schedules import build_schedule
from ored.utils.logging_utils import get_logger, section
from ored.utils.seed import resolve_device, set_seed

logger = get_logger(__name__)

PARAMETER_TOLERANCE = 0.01


class PretrainError(RuntimeError):
    pass


class TrainingOOMError(PretrainError):
    pass


@dataclass
class PretrainData:
    directory: Path
    manifest: Dict[str, Any]
    identity: Dict[str, Any]
    tokenizer: Tokenizer
    tokenizer_info: Dict[str, Any]
    train: TokenWindowDataset
    val: TokenWindowDataset


def _dataset_directory(cfg: Config, tokenizer_info: Dict[str, Any]) -> Path:
    tokens = cfg.data.tokens
    if tokens.manifest:
        path = Path(tokens.manifest)
        return path.parent if path.name == "manifest.json" else path
    if tokens.dataset_name and tokens.dataset_version:
        label = f"{tokenizer_info['name']}-{tokenizer_info['version']}"
        directory = Path(tokens.cache_dir) / tokens.dataset_name / tokens.dataset_version / label
        if not (directory / "manifest.json").is_file():
            from ored.storage import ObjectLayout, R2Settings
            from ored.storage.objects import fetch_token_dataset
            settings = R2Settings.from_env()
            object_dir = ObjectLayout(settings.prefix).token_dir(tokens.dataset_name, tokens.dataset_version, label)
            logger.info(f"fetching r2://{settings.bucket}/{object_dir} into {directory}")
            fetch_token_dataset(settings.store(), object_dir, directory, tokens.verify, log=logger.info)
        return directory
    raise PretrainError(
        "no training dataset is configured: set data.tokens.manifest (a local shard directory) or "
        "data.tokens.dataset_name + data.tokens.dataset_version (fetched from R2). Build the shards "
        "first with scripts/corpus.py once the corpus is available.")


def prepare_token_data(cfg: Config) -> PretrainData:
    tokens = cfg.data.tokens
    if not tokens.tokenizer:
        raise PretrainError("data.tokens.tokenizer must name a tokenizer artifact (scripts/corpus.py "
                            "train-tokenizer writes one); the model never rebuilds its tokenizer")
    tokenizer, info = load_artifact(tokens.tokenizer)
    if tokenizer.vocab_size != cfg.data.vocab_size:
        raise PretrainError(f"tokenizer {info['name']} {info['version']} has {tokenizer.vocab_size} tokens, "
                            f"data.vocab_size is {cfg.data.vocab_size}")
    directory = _dataset_directory(cfg, info)
    manifest = load_manifest(directory)
    if manifest["tokenizer"]["sha256"] != info["sha256"]:
        raise PretrainError(f"{directory} was tokenized with tokenizer {manifest['tokenizer']['sha256'][:16]}, "
                            f"not {info['sha256'][:16]} ({tokens.tokenizer})")
    verify_shards(directory, manifest, tokens.verify, splits=("train", "val"))
    identity = dataset_identity(manifest, manifest_file_sha256(directory), tokens.dataset_id)
    block = cfg.data.block_size
    train = TokenWindowDataset(directory, manifest, "train", block, cfg.data.stride)
    val = TokenWindowDataset(directory, manifest, "val", block, block)
    if len(train) == 0 or len(val) == 0:
        raise PretrainError(f"{directory} has no full {block}-token window in train or val")
    return PretrainData(directory, manifest, identity, tokenizer, info, train, val)


def build_adamw(model: nn.Module, cfg: Config, device: torch.device) -> torch.optim.Optimizer:
    if cfg.training.optimizer.lower() != "adamw":
        raise PretrainError("the pretraining loop uses training.optimizer: adamw")
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.training.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    extra = {}
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        extra["fused"] = True
    return torch.optim.AdamW(groups, lr=cfg.training.learning_rate, betas=(0.9, 0.95), **extra)


def check_parameter_count(cfg: Config, actual: int) -> None:
    expected = cfg.model.expected_parameters
    if expected and abs(actual - expected) > PARAMETER_TOLERANCE * expected:
        raise PretrainError(f"the model has {actual:,} parameters, model.expected_parameters is {expected:,} "
                            f"(more than {PARAMETER_TOLERANCE:.0%} apart): the architecture changed")


def _code_version() -> Dict[str, Any]:
    from ored.data.snapshot import code_version
    return code_version()


class PretrainTrainer:

    def __init__(self, cfg: Config, data: Optional[PretrainData] = None, store: Any = None) -> None:
        if cfg.data.source != "tokens":
            raise PretrainError("PretrainTrainer trains on data.source: tokens")
        if (cfg.checkpoint.best_metric, cfg.checkpoint.best_mode) != ("val_loss", "min"):
            raise PretrainError("the pretraining loop keeps the epoch with the lowest val_loss")
        self.cfg = cfg
        set_seed(cfg.seed, cfg.deterministic)
        self.device = resolve_device(cfg.training.device)
        gpu.configure_backends(cfg.training.tf32, self.device)
        self.precision = gpu.resolve_precision(cfg.training.precision, self.device)

        self.data = data or prepare_token_data(cfg)
        self.model = build_model(cfg, vocab_size=self.data.tokenizer.vocab_size).to(self.device)
        self.n_parameters = self.model.num_parameters()
        check_parameter_count(cfg, self.n_parameters)
        self.model_version = cfg.model.version or cfg.run_name
        self.optimizer = build_adamw(self.model, cfg, self.device)
        self.scaler = gpu.make_scaler(self.precision.use_scaler)
        self.forward_model = torch.compile(self.model) if cfg.training.compile else self.model

        self.generator = torch.Generator()
        self.loaders = {"train": self._loader(self.data.train, shuffle=cfg.data.shuffle_train),
                        "val": self._loader(self.data.val, shuffle=False)}
        self.micro_batches_per_epoch = len(self.loaders["train"])
        self.accum = cfg.training.grad_accum_steps
        self.steps_per_epoch = math.ceil(self.micro_batches_per_epoch / self.accum)
        self.total_steps = self.steps_per_epoch * cfg.training.epochs
        self.schedule = build_schedule(cfg.training.scheduler)
        self.current_lr = cfg.training.learning_rate

        self.best = BestCheckpoint(cfg.checkpoint_dir, cfg.checkpoint.best_metric, cfg.checkpoint.best_mode)
        self.store = store
        self.r2_object: Optional[str] = None
        if cfg.checkpoint.r2_upload:
            from ored.storage import ObjectLayout, R2Settings
            settings = R2Settings.from_env()
            self.store = store or settings.store()
            self.r2_object = ObjectLayout(settings.prefix).checkpoint_best(cfg.run_name)
        self.upload_errors: List[str] = []

        self.start_epoch = 1
        self.global_step = 0
        self.history: List[Dict[str, Any]] = []
        self.epochs_without_improvement = 0
        if cfg.training.resume == "best":
            self._resume()
        elif cfg.training.resume:
            raise PretrainError("this model resumes only from its own best.pt: training.resume=best")


    def _loader(self, dataset: TokenWindowDataset, shuffle: bool) -> DataLoader:
        cfg = self.cfg.data
        workers = cfg.num_workers
        options: Dict[str, Any] = {"batch_size": cfg.batch_size, "num_workers": workers,
                                   "pin_memory": cfg.pin_memory and self.device.type == "cuda"}
        if workers > 0:
            options.update(persistent_workers=cfg.persistent_workers, prefetch_factor=cfg.prefetch_factor)
        sampler = RandomSampler(dataset, generator=self.generator) if shuffle else None
        return DataLoader(dataset, sampler=sampler, shuffle=False, drop_last=False, **options)

    def architecture(self) -> Dict[str, Any]:
        return dict(self.model.describe())

    def _resume(self) -> None:
        payload = self.best.load(map_location=self.device)
        check_compatible(payload, model_version=self.model_version, architecture=self.architecture(),
                         tokenizer_sha256=self.data.tokenizer_info["sha256"],
                         dataset_sha256=self.data.identity["content_sha256"], path=self.best.path)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        if payload.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scaler_state_dict") and self.precision.use_scaler:
            self.scaler.load_state_dict(payload["scaler_state_dict"])
        self.start_epoch = int(payload["epoch"]) + 1
        self.global_step = int(payload["global_step"])
        self.history = list(payload.get("history") or [])
        logger.info(f"resumed from {self.best.path} (epoch {payload['epoch']}, step {self.global_step:,}, "
                    f"best val_loss {self.best.best_value:.4f})")


    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with self.precision.autocast(self.device):
            logits = self.forward_model(x)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))

    def _apply_learning_rate(self) -> None:
        t = self.cfg.training
        self.current_lr = t.learning_rate * self.schedule(self.global_step, self.total_steps,
                                                          t.warmup_steps, t.min_lr_ratio)
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        cfg = self.cfg
        self.model.train()
        self.generator.manual_seed(cfg.seed + epoch)
        n_micro = self.micro_batches_per_epoch
        loss_sum = torch.zeros((), device=self.device)
        examples = 0
        window_sum = torch.zeros((), device=self.device)
        window_count = 0
        window_tokens = 0
        window_start = time.perf_counter()
        epoch_start = window_start
        epoch_tokens = 0
        self.optimizer.zero_grad(set_to_none=True)

        for index, (x, y) in enumerate(self.loaders["train"]):
            group_start = (index // self.accum) * self.accum
            group_size = min(self.accum, n_micro - group_start)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            loss = self._loss(x, y)
            self.scaler.scale(loss / group_size).backward()

            detached = loss.detach()
            loss_sum += detached * x.size(0)
            examples += x.size(0)
            window_sum += detached
            window_count += 1
            window_tokens += x.numel()
            epoch_tokens += x.numel()

            if index + 1 - group_start < group_size:
                continue
            self._apply_learning_rate()
            if cfg.training.grad_clip and cfg.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.training.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            if self.global_step % cfg.training.log_every_steps == 0:
                gpu.synchronize(self.device)
                elapsed = time.perf_counter() - window_start
                logger.info(f"epoch {epoch} step {self.global_step:,}/{self.total_steps:,} | "
                            f"loss {(window_sum / window_count).item():.4f} | lr {self.current_lr:.2e} | "
                            f"{window_tokens / max(elapsed, 1e-9):,.0f} tok/s | "
                            f"peak {gpu.peak_memory_gib(self.device):.2f} GiB")
                window_sum.zero_()
                window_count = window_tokens = 0
                window_start = time.perf_counter()

        gpu.synchronize(self.device)
        seconds = time.perf_counter() - epoch_start
        return {"train_loss": (loss_sum / max(examples, 1)).item(), "seconds": seconds,
                "tokens_per_second": epoch_tokens / max(seconds, 1e-9), "train_tokens": epoch_tokens}

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> float:
        self.model.eval()
        limit = self.cfg.training.eval_max_batches
        total = torch.zeros((), device=self.device)
        examples = 0
        for index, (x, y) in enumerate(self.loaders[split]):
            if limit and index >= limit:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            total += self._loss(x, y) * x.size(0)
            examples += x.size(0)
        return (total / max(examples, 1)).item()


    def payload(self, epoch: int, record: Dict[str, Any]) -> Dict[str, Any]:
        t = self.cfg.training
        return {
            "schema": CHECKPOINT_SCHEMA,
            "model_version": self.model_version,
            "run_name": self.cfg.run_name,
            "architecture": self.architecture(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler": {"name": t.scheduler, "current_lr": self.current_lr, "total_steps": self.total_steps,
                          "warmup_steps": t.warmup_steps, "min_lr_ratio": t.min_lr_ratio},
            "scaler_state_dict": self.scaler.state_dict() if self.precision.use_scaler else None,
            "precision": self.precision.name,
            "epoch": epoch,
            "global_step": self.global_step,
            "best": {"metric": "val_loss", "mode": "min", "value": record["val_loss"]},
            "metrics": dict(record),
            "history": [dict(r) for r in self.history] + [dict(record)],
            "tokenizer": dict(self.data.tokenizer_info),
            "dataset": dict(self.data.identity),
            "config": self.cfg.to_dict(),
            "code": _code_version(),
            "torch_version": str(torch.__version__),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _publish_best(self) -> None:
        if self.store is None or self.r2_object is None:
            return
        from ored.storage.objects import replace_object
        try:
            replace_object(self.store, self.best.path, self.r2_object)
            logger.info(f"  best.pt replaced in R2 at {self.r2_object}")
        except Exception as exc:
            self.upload_errors.append(f"{self.r2_object}: {exc}")
            logger.error(f"  R2 upload of best.pt failed (local best.pt kept): {exc}")


    def _oom(self, exc: BaseException) -> TrainingOOMError:
        cfg = self.cfg
        peak = gpu.peak_memory_gib(self.device)
        total = gpu.total_memory_gib(self.device)
        return TrainingOOMError(
            f"CUDA ran out of memory with micro-batch {cfg.data.batch_size} x {cfg.data.block_size} tokens "
            f"(grad_accum_steps {self.accum}, precision {self.precision.name}, gradient_checkpointing "
            f"{cfg.model.gradient_checkpointing}); peak allocated {peak:.2f} of {total:.2f} GiB. Lower "
            f"data.batch_size and raise training.grad_accum_steps to keep the effective batch, or set "
            f"model.gradient_checkpointing: true, or measure with scripts/pretrain.py --probe-batch-size. "
            f"The run stopped; best.pt is unchanged. ({exc})")

    def fit(self) -> Dict[str, Any]:
        cfg = self.cfg
        self._log_header()
        gpu.reset_peak_memory(self.device)
        patience = cfg.training.early_stopping_patience
        stopped_early = False
        started = time.time()
        for epoch in range(self.start_epoch, cfg.training.epochs + 1):
            if patience and self.epochs_without_improvement >= patience:
                stopped_early = True
                break
            try:
                record: Dict[str, Any] = {"epoch": epoch, **self._train_epoch(epoch), "lr": self.current_lr}
                evaluate = epoch % cfg.training.eval_every_epochs == 0 or epoch == cfg.training.epochs
                if evaluate:
                    record["val_loss"] = self.evaluate("val")
            except torch.cuda.OutOfMemoryError as exc:
                raise self._oom(exc) from exc
            record["global_step"] = self.global_step
            record["peak_vram_gib"] = gpu.peak_memory_gib(self.device)

            improved = False
            if "val_loss" in record:
                record["val_ppl"] = math.exp(min(record["val_loss"], 20.0))
                improved = self.best.consider(record["val_loss"], lambda: self.payload(epoch, record))
                if improved:
                    self._publish_best()
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
            record["best_val_loss"] = self.best.best_value
            self.history.append(record)
            self._log_epoch(record, improved)

        self._save_history()
        elapsed = time.time() - started
        logger.info(section("TRAINING COMPLETE"))
        logger.info(f"epochs run        : {len(self.history)}{' (stopped early)' if stopped_early else ''}")
        logger.info(f"wall clock        : {elapsed:,.1f}s")
        logger.info(f"best val_loss     : {self.best.best_value}")
        logger.info(f"checkpoint        : {self.best.path} (the only checkpoint kept)")
        for error in self.upload_errors:
            logger.info(f"NOT UPLOADED      : {error}")
        return {"history": self.history, "best_val_loss": self.best.best_value,
                "best_path": str(self.best.path), "elapsed_seconds": elapsed,
                "upload_errors": list(self.upload_errors), "parameters": self.n_parameters}

    def _save_history(self) -> None:
        path = self.best.directory / "history.json"
        path.write_text(json.dumps({"config": self.cfg.to_dict(), "history": self.history}, indent=2),
                        encoding="utf-8")

    def _log_epoch(self, record: Dict[str, Any], improved: bool) -> None:
        val = (f" | val loss {record['val_loss']:.4f} (ppl {record['val_ppl']:.1f})"
               if "val_loss" in record else " | no validation this epoch")
        best = record["best_val_loss"]
        logger.info(f"Epoch {record['epoch']:>3}/{self.cfg.training.epochs} | train loss "
                    f"{record['train_loss']:.4f}{val} | best {best if best is None else f'{best:.4f}'} | "
                    f"{record['tokens_per_second']:,.0f} tok/s | peak {record['peak_vram_gib']:.2f} GiB"
                    + ("  <- new best.pt" if improved else ""))

    def _log_header(self) -> None:
        cfg, data = self.cfg, self.data
        tokens_per_step = cfg.data.batch_size * self.accum * cfg.data.block_size
        logger.info(section(f"PRETRAINING RUN: {cfg.run_name} ({self.model_version})"))
        for line in gpu.device_report(self.device):
            logger.info(line)
        d = self.model.describe()
        logger.info(f"model         : {d['n_layer']} layers, d_model {d['d_model']}, {d['n_head']} heads, "
                    f"d_ff {d['d_ff']}, block {d['block_size']}, vocab {d['vocab_size']}")
        logger.info(f"parameters    : {self.n_parameters:,}")
        logger.info(f"attention     : {cfg.model.attention} | gradient checkpointing "
                    f"{cfg.model.gradient_checkpointing} | torch.compile {cfg.training.compile} | "
                    f"tf32 {cfg.training.tf32}")
        logger.info(f"precision     : {self.precision.name}"
                    + (" (GradScaler on)" if self.precision.use_scaler else ""))
        logger.info(f"batch         : micro-batch {cfg.data.batch_size} x {cfg.data.block_size} tokens, "
                    f"grad accumulation {self.accum} -> effective {cfg.data.batch_size * self.accum} sequences "
                    f"= {tokens_per_step:,} tokens per optimizer step")
        logger.info(f"schedule      : {self.steps_per_epoch:,} steps/epoch, {self.total_steps:,} total, "
                    f"{cfg.training.scheduler} lr {cfg.training.learning_rate} warmup {cfg.training.warmup_steps}")
        logger.info(f"loader        : {cfg.data.num_workers} workers, pin_memory "
                    f"{cfg.data.pin_memory and self.device.type == 'cuda'}")
        ident = data.identity
        logger.info(f"dataset       : {ident['name']} {ident['version']} (content {ident['content_sha256'][:16]}, "
                    f"registry id {ident.get('registry_id') or 'not set'})")
        logger.info(f"  train       : {data.train.describe()}")
        logger.info(f"  val         : {data.val.describe()}")
        info = data.tokenizer_info
        logger.info(f"tokenizer     : {info['name']} {info['version']} ({info['vocab_size']} tokens, "
                    f"sha256 {info['sha256'][:16]})")
        logger.info(f"checkpoint    : {self.best.path} (best only)"
                    + (f"; R2 {self.r2_object}" if self.r2_object else ""))
        logger.info("-" * 78)


def probe_batch_sizes(cfg: Config, candidates: Sequence[int] = (1, 2, 4, 6, 8, 12, 16, 24, 32),
                      steps: int = 2) -> List[Dict[str, Any]]:
    device = resolve_device(cfg.training.device)
    if device.type != "cuda":
        raise PretrainError("--probe-batch-size measures GPU memory; it needs a CUDA device")
    gpu.configure_backends(cfg.training.tf32, device)
    precision = gpu.resolve_precision(cfg.training.precision, device)
    model = build_model(cfg, vocab_size=cfg.data.vocab_size).to(device)
    optimizer = build_adamw(model, cfg, device)
    scaler = gpu.make_scaler(precision.use_scaler)
    forward = torch.compile(model) if cfg.training.compile else model
    model.train()
    results = []
    for size in candidates:
        gpu.reset_peak_memory(device)
        try:
            started = time.perf_counter()
            for _ in range(steps):
                x = torch.randint(0, cfg.data.vocab_size, (size, cfg.data.block_size), device=device)
                with precision.autocast(device):
                    logits = forward(x)
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), x.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            gpu.synchronize(device)
            seconds = (time.perf_counter() - started) / steps
            results.append({"batch_size": size, "fits": True, "peak_gib": gpu.peak_memory_gib(device),
                            "tokens_per_second": size * cfg.data.block_size / seconds})
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": size, "fits": False, "peak_gib": gpu.peak_memory_gib(device)})
            break
        finally:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain the ~50M Ored model on token shards.")
    parser.add_argument("--config", default="configs/ored_50m.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", action="store_true", help="continue from this run's best.pt")
    parser.add_argument("--describe", action="store_true",
                        help="build the model, print its exact parameter count and stop (no data needed)")
    parser.add_argument("--probe-batch-size", action="store_true",
                        help="measure which micro-batch sizes fit in GPU memory and stop")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    overrides = list(args.overrides) + (["training.resume=best"] if args.resume else [])
    cfg = load_config(args.config, overrides)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.describe:
        model = build_model(cfg, vocab_size=cfg.data.vocab_size)
        print(json.dumps({**model.describe(), "attention": cfg.model.attention,
                          "embeddings_tied": model.embeddings_tied,
                          "expected_parameters": cfg.model.expected_parameters}, indent=2))
        check_parameter_count(cfg, model.num_parameters())
        return 0
    if args.probe_batch_size:
        for row in probe_batch_sizes(cfg):
            print(json.dumps(row))
        return 0
    PretrainTrainer(cfg).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
