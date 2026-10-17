from __future__ import annotations

from pathlib import Path

import pytest

from ored.config import Config, config_from_dict, load_config

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "language_model.yaml"


def test_loads_the_real_config():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, Config)
    assert cfg.data.split.train > 0
    assert cfg.model.name == "transformer" and cfg.task == "language_model"


def test_overrides_are_typed():
    cfg = load_config(CONFIG_PATH, ["training.epochs=3", "model.d_model=64",
                                    "deterministic=false", "model.attention=sdpa"])
    assert cfg.training.epochs == 3 and isinstance(cfg.training.epochs, int)
    assert cfg.model.d_model == 64
    assert cfg.deterministic is False
    assert cfg.model.attention == "sdpa"


def test_typo_in_override_is_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(CONFIG_PATH, ["training.lerning_rate=0.1"])


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        load_config(CONFIG_PATH, ["training.epochs=0"])
    with pytest.raises(ValueError):
        load_config(CONFIG_PATH, ["data.split.train=0.9"])
    with pytest.raises(ValueError):
        load_config(CONFIG_PATH, ["model.n_head=3"])
    with pytest.raises(ValueError, match="model.name must be transformer"):
        load_config(CONFIG_PATH, ["model.name=mlp"])
    with pytest.raises(ValueError, match="task must be language_model"):
        load_config(CONFIG_PATH, ["task=bit_addition"])


def test_round_trip_through_dict():
    cfg = load_config(CONFIG_PATH)
    rebuilt = config_from_dict(cfg.to_dict())
    assert rebuilt.to_dict() == cfg.to_dict()


def test_checkpoints_saved_with_the_retired_mlp_keys_still_load():
    cfg = load_config(CONFIG_PATH)
    saved = cfg.to_dict()
    saved["data"].update(n_bits=4, raw_path="data/raw/bit_addition.csv")
    saved["model"].update(hidden_sizes=[32, 32], activation="relu")
    assert config_from_dict(saved).to_dict() == cfg.to_dict()


def test_text_settings_stay_text_when_they_look_like_numbers():
    from pathlib import Path

    from ored.config import load_config

    path = Path(__file__).resolve().parent / "configs" / "language_model.yaml"
    cfg = load_config(path, ["data.source=supabase", "data.supabase.dataset_tag=2024",
                             "data.supabase.snapshot=" + "1e5" + "0" * 61, "run_name=007",
                             "training.epochs=3", "training.learning_rate=1e-4", "data.shuffle_train=false"])
    assert cfg.data.supabase.dataset_tag == "2024" and cfg.run_name == "007"
    assert cfg.data.supabase.snapshot == "1e5" + "0" * 61
    assert cfg.training.epochs == 3 and cfg.training.learning_rate == 1e-4 and cfg.data.shuffle_train is False


def test_subword_vocab_must_hold_every_byte():
    from ored.config import Config

    cfg = Config()
    cfg.data.tokenizer = "subword"
    cfg.data.vocab_size = 100
    with pytest.raises(ValueError, match="vocab_size must be >= 256"):
        cfg.data.validate()
