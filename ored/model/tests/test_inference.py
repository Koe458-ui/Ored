from __future__ import annotations

import pytest
import torch

from ored.inference.predictor import (
    BasePredictor,
    LanguageModelPredictor,
    load_predictor,
    resolve_task,
)
from ored.inference.predictor import main as predictor_main
from ored.training.trainer import Trainer
from ored.utils.checkpoint import CheckpointError


def test_missing_checkpoint_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train a model first"):
        BasePredictor.from_checkpoint(tmp_path / "nope.pt")


@pytest.fixture()
def trained_lm_checkpoint(tiny_corpus):
    Trainer(tiny_corpus).fit()
    return tiny_corpus.checkpoint_dir / "best.pt"


def test_language_model_checkpoint_loads_with_its_tokenizer(trained_lm_checkpoint):
    predictor = load_predictor(trained_lm_checkpoint, device="cpu")

    assert isinstance(predictor, LanguageModelPredictor)
    assert predictor.task == "language_model"
    assert predictor.vocab_size == predictor.tokenizer.vocab_size
    assert predictor.model.describe()["vocab_size"] == predictor.tokenizer.vocab_size
    assert predictor.model.training is False


def test_language_model_predicts_from_text(trained_lm_checkpoint):
    predictor = load_predictor(trained_lm_checkpoint, device="cpu")
    prediction = predictor.predict("3 + 4 = ")

    assert prediction.prompt == "3 + 4 = "
    assert isinstance(prediction.completion, str)
    assert prediction.text.startswith("3 + 4 = ")
    assert "\n" not in prediction.completion


def test_language_model_text_predictions_are_deterministic(trained_lm_checkpoint):
    predictor = load_predictor(trained_lm_checkpoint, device="cpu")
    assert predictor.predict("2 + 3 = ").completion == predictor.predict("2 + 3 = ").completion


def test_language_model_generates_text(trained_lm_checkpoint):
    predictor = load_predictor(trained_lm_checkpoint, device="cpu")
    text = predictor.generate("the ", max_new_tokens=20, greedy=True)

    assert text.startswith("the ")
    assert len(text) > len("the ")


def test_language_model_predict_sum_scores_against_the_right_answer(trained_lm_checkpoint):
    predictor = load_predictor(trained_lm_checkpoint, device="cpu")
    prediction = predictor.predict_sum(3, 4)

    assert prediction.prompt == "3 + 4 = "
    assert prediction.expected == "7"
    assert prediction.correct in (True, False)


def test_language_model_weights_match_the_checkpoint(trained_lm_checkpoint):
    from ored.utils.checkpoint import load_checkpoint

    predictor = load_predictor(trained_lm_checkpoint, device="cpu")
    payload = load_checkpoint(trained_lm_checkpoint)
    state = predictor.model.state_dict()

    assert set(state) == set(payload["model_state_dict"])
    for key, tensor in state.items():
        assert torch.allclose(payload["model_state_dict"][key], tensor)


def test_predictor_class_dispatches_on_the_checkpoint_task(trained_lm_checkpoint):
    predictor = BasePredictor.from_checkpoint(trained_lm_checkpoint, device="cpu")
    assert isinstance(predictor, LanguageModelPredictor)


def test_language_model_checkpoint_without_a_tokenizer_is_rejected(
    trained_lm_checkpoint, tmp_path
):
    payload = torch.load(trained_lm_checkpoint, map_location="cpu", weights_only=True)
    payload["tokenizer"] = None
    stripped = tmp_path / "no_tokenizer.pt"
    torch.save(payload, stripped)

    with pytest.raises(ValueError, match="no tokenizer"):
        load_predictor(stripped, device="cpu")


def test_language_model_cli_completes_text(trained_lm_checkpoint):
    import io
    import logging

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    ored_logger = logging.getLogger("ored")
    ored_logger.addHandler(handler)
    try:
        exit_code = predictor_main([
            "--checkpoint", str(trained_lm_checkpoint),
            "--device", "cpu",
            "--text", "5 + 6 = ",
            "--quiet",
        ])
    finally:
        ored_logger.removeHandler(handler)

    assert exit_code == 0
    assert "5 + 6 = " in stream.getvalue()


def test_the_default_checkpoint_is_the_transformer():
    from ored.inference.predictor import DEFAULT_CHECKPOINT, resolve_checkpoint

    assert resolve_checkpoint() == DEFAULT_CHECKPOINT == "checkpoints/char_transformer/best.pt"
    assert resolve_checkpoint("somewhere/else.pt") == "somewhere/else.pt"


def _payload(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_checkpoint_task_is_read_from_the_payload_not_just_the_label(trained_lm_checkpoint):
    from ored.config import config_from_dict

    payload = _payload(trained_lm_checkpoint)
    cfg = config_from_dict(payload["config"])

    assert resolve_task(payload, cfg, trained_lm_checkpoint) == "language_model"

    payload["config"]["task"] = "bit_addition"
    stale = config_from_dict(payload["config"])
    assert resolve_task(payload, stale, trained_lm_checkpoint) == "language_model"


def test_stale_task_label_still_loads_the_char_transformer(trained_lm_checkpoint, tmp_path):
    payload = _payload(trained_lm_checkpoint)
    payload["config"]["task"] = "bit_addition"
    mislabelled = tmp_path / "mislabelled.pt"
    torch.save(payload, mislabelled)

    predictor = load_predictor(mislabelled, device="cpu")

    assert isinstance(predictor, LanguageModelPredictor)
    assert predictor.model.describe()["vocab_size"] == predictor.tokenizer.vocab_size
    state = predictor.model.state_dict()
    assert set(state) == set(payload["model_state_dict"])


def test_tokenizer_saved_under_extra_is_accepted(trained_lm_checkpoint, tmp_path):
    payload = _payload(trained_lm_checkpoint)
    payload["extra"] = {"tokenizer": payload.pop("tokenizer")}
    top_level = tmp_path / "extra_tokenizer.pt"
    torch.save(payload, top_level)

    predictor = load_predictor(top_level, device="cpu")
    assert predictor.vocab_size == predictor.tokenizer.vocab_size


def test_state_dict_loading_stays_strict(trained_lm_checkpoint, tmp_path):
    payload = _payload(trained_lm_checkpoint)
    payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
    incomplete = tmp_path / "incomplete.pt"
    torch.save(payload, incomplete)

    with pytest.raises(CheckpointError, match="missing from the checkpoint"):
        load_predictor(incomplete, device="cpu")


def test_char_transformer_checkpoint_runs_through_the_cli(trained_lm_checkpoint, tmp_path):
    import io
    import logging
    import shutil

    checkpoint = tmp_path / "checkpoints" / "char_transformer" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    shutil.copy(trained_lm_checkpoint, checkpoint)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    ored_logger = logging.getLogger("ored")
    ored_logger.addHandler(handler)
    try:
        exit_code = predictor_main(["--checkpoint", str(checkpoint), "--device", "cpu"])
    finally:
        ored_logger.removeHandler(handler)

    output = stream.getvalue()
    assert exit_code == 0
    assert "task       : language_model" in output
    assert "generated text:" in output
    assert "3 + 4 = " in output
    assert "bits " not in output

