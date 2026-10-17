from __future__ import annotations

import json

import torch

from ored.training.trainer import Trainer, build_optimizer
from ored.utils.checkpoint import load_checkpoint
from ored.utils.weight_stats import snapshot_parameters, summarise_weight_change


def test_loss_decreases(tiny_corpus):
    cfg = tiny_corpus
    cfg.training.epochs = 3
    trainer = Trainer(cfg)
    result = trainer.fit()

    first = result["history"][0]["train_loss"]
    last = result["history"][-1]["train_loss"]
    assert last < first, f"training loss did not fall: {first} -> {last}"


def test_weights_actually_change(tiny_corpus):
    trainer = Trainer(tiny_corpus)
    before = snapshot_parameters(trainer.model)
    trainer.fit()

    rows = summarise_weight_change(before, trainer.model)
    assert rows, "no parameters found"
    for row in rows:
        assert row["delta_norm"] > 0, f"{row['name']} never changed"


def test_one_step_lowers_the_loss_for_that_batch(tiny_corpus):
    trainer = Trainer(tiny_corpus)
    batch = next(iter(trainer.loaders["train"]))

    trainer.model.train()
    loss_before, _, _ = trainer.task.compute_loss(trainer.model, batch, trainer.device)

    trainer.optimizer.zero_grad(set_to_none=True)
    loss_before.backward()
    trainer.optimizer.step()

    with torch.no_grad():
        loss_after, _, _ = trainer.task.compute_loss(trainer.model, batch, trainer.device)

    assert loss_after.item() < loss_before.item()


def test_checkpoints_are_written_and_reloadable(tiny_corpus):
    trainer = Trainer(tiny_corpus)
    trainer.fit()

    for name in ("base.pt", "best.pt", "live.pt"):
        path = tiny_corpus.checkpoint_dir / name
        assert path.exists(), f"{name} was not written"

        payload = load_checkpoint(path)
        assert payload["checkpoint_kind"] == name[:-3]
        assert "model_state_dict" in payload
        assert payload["config"]["model"]["d_model"] == tiny_corpus.model.d_model

        if name == "live.pt":
            for key, tensor in trainer.model.state_dict().items():
                assert torch.allclose(payload["model_state_dict"][key], tensor)


def test_history_json_is_written(tiny_corpus):
    trainer = Trainer(tiny_corpus)
    trainer.fit()
    path = tiny_corpus.checkpoint_dir / "history.json"
    data = json.loads(path.read_text())
    assert len(data["history"]) == len(trainer.history)
    assert "train_loss" in data["history"][0]


def test_same_seed_gives_identical_results(tiny_corpus):
    first = Trainer(tiny_corpus).fit()["history"]
    second = Trainer(tiny_corpus).fit()["history"]
    assert [round(r["train_loss"], 10) for r in first] == \
           [round(r["train_loss"], 10) for r in second]


def test_different_seeds_give_different_results(tiny_corpus):
    first = Trainer(tiny_corpus).fit()["history"]
    tiny_corpus.seed += 1
    second = Trainer(tiny_corpus).fit()["history"]
    assert first[-1]["train_loss"] != second[-1]["train_loss"]


def test_validation_does_not_change_weights(tiny_corpus):
    trainer = Trainer(tiny_corpus)
    before = snapshot_parameters(trainer.model)
    trainer.evaluate("val")
    for name, param in trainer.model.named_parameters():
        assert torch.equal(before[name], param.detach())


def test_builders_reject_unknown_names(tiny_corpus):
    import pytest
    with pytest.raises(ValueError):
        build_optimizer("not_an_optimizer", torch.nn.Linear(2, 2), 0.1, 0.0)
