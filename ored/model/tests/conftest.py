from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ored.config import load_config
from ored.data.corpus import generate_corpus

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
LM_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "language_model.yaml"


@pytest.fixture()
def tiny_lm_cfg(tmp_path):
    cfg = load_config(
        LM_CONFIG_PATH,
        overrides=[
            "data.source=generated",
            "training.epochs=2",
            "training.log_every=100",
            "training.early_stopping_patience=0",
            "training.device=cpu",
            "data.block_size=32",
            "data.stride=16",
            "data.batch_size=8",
            "data.corpus.sentence_lines=300",
            "data.corpus.max_operand=9",
            "data.corpus.arithmetic_repeats=3",
            "model.d_model=32",
            "model.n_layer=2",
            "model.n_head=2",
            "model.d_ff=64",
            "model.dropout=0.0",
        ],
    )
    cfg.data.corpus.dir = str(tmp_path / "corpus")
    cfg.paths.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.run_name = "pytest_lm"
    return cfg


@pytest.fixture()
def tiny_corpus(tiny_lm_cfg):
    generate_corpus(tiny_lm_cfg)
    return tiny_lm_cfg
