from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, List, Optional

import torch


@dataclass(frozen=True)
class Precision:
    name: str
    dtype: Optional[torch.dtype]
    use_scaler: bool

    def autocast(self, device: torch.device) -> Any:
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=self.dtype)


def bf16_supported(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_bf16_supported()


def resolve_precision(requested: str, device: torch.device) -> Precision:
    if requested == "fp32":
        return Precision("fp32", None, False)
    if device.type != "cuda":
        if requested == "auto":
            return Precision("fp32", None, False)
        if requested == "bf16":
            return Precision("bf16", torch.bfloat16, False)
        raise ValueError("training.precision fp16 needs a CUDA GPU (use fp32 or bf16 on CPU)")
    if requested == "auto":
        requested = "bf16" if bf16_supported(device) else "fp16"
    if requested == "bf16":
        if not bf16_supported(device):
            raise ValueError(f"{torch.cuda.get_device_name(device)} does not support bf16; "
                             f"use training.precision: fp16 or auto")
        return Precision("bf16", torch.bfloat16, False)
    if requested == "fp16":
        return Precision("fp16", torch.float16, True)
    raise ValueError(f"unknown precision {requested!r}")


def make_scaler(enabled: bool) -> Any:
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        return amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def configure_backends(tf32: bool, device: torch.device) -> None:
    if device.type == "cuda" and tf32:
        torch.set_float32_matmul_precision("high")


def device_report(device: torch.device) -> List[str]:
    lines = [f"torch {torch.__version__}" + (f", CUDA {torch.version.cuda}" if torch.version.cuda else "")]
    if device.type != "cuda":
        lines.append(f"device: {device} (no CUDA GPU in use)")
        return lines
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    lines.append(f"GPU: {props.name} (compute capability {props.major}.{props.minor}, "
                 f"{props.multi_processor_count} SMs)")
    lines.append(f"VRAM: {total / 2**30:.2f} GiB total, {free / 2**30:.2f} GiB free at start")
    lines.append(f"bf16 supported: {bf16_supported(device)}")
    backends = []
    for name in ("flash", "mem_efficient", "math"):
        check = getattr(torch.backends.cuda, f"{name}_sdp_enabled", None)
        if check is not None:
            backends.append(f"{name}={'on' if check() else 'off'}")
    if backends:
        lines.append("SDPA backends allowed: " + ", ".join(backends))
    return lines


def total_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / 2**30


def peak_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 2**30


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
