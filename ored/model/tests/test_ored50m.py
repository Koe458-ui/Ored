from __future__ import annotations

import gzip
import json
import pickle
import random
import re
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from ored.config import load_config
from ored.data.dataset_registry import DatasetEntry, RegistryError
from ored.data.readers import JsonlReader, ReaderError, detect_format, inspect_file, open_reader
from ored.data.token_shards import (
    BuildSettings,
    ShardError,
    TokenWindowDataset,
    build_token_shards,
    load_manifest,
    verify_shards,
)
from ored.data.tokenizer import END_OF_TEXT, SubwordTokenizer, count_words
from ored.data.tokenizer_artifact import TokenizerArtifactError, load_artifact, save_artifact, tokenizer_sha256
from ored.learning.r2 import R2Store
from ored.models.registry import build_model
from ored.models.transformer import Transformer
from ored.storage import ObjectLayout, R2Settings, StorageConfigError
from ored.storage.layout import LayoutError
from ored.storage.objects import fetch_token_dataset, publish_token_dataset, replace_object
from ored.training.best_checkpoint import (
    CHECKPOINT_SCHEMA,
    BestCheckpoint,
    BestCheckpointError,
    check_compatible,
)
from ored.training.pretrain import PretrainError, PretrainTrainer, check_parameter_count
from test_r2_store import FakeS3

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
CONFIG_50M = CONFIGS / "ored50m.yaml"
MIGRATIONS = Path(__file__).resolve().parents[2] / "supabase" / "migrations"
EXPECTED_50M = 49_894_912

WORDS = ("the quick brown fox jumps over a lazy dog while seven small birds sing about rivers "
         "mountains energy light atoms numbers tables gardens").split()


def synthetic_docs(n: int, seed: int = 0, prefix: str = "doc"):
    rng = random.Random(seed)
    return [{"meta": {"uid": f"{prefix}-{i}"}, "body": " ".join(rng.choice(WORDS) for _ in range(rng.randint(30, 90))),
             "extra": rng.random()} for i in range(n)]


def write_jsonl(path: Path, records, compress: bool = False) -> Path:
    data = "".join(json.dumps(r) + "\n" for r in records).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if compress else data)
    return path


@pytest.fixture(scope="module")
def tokenizer():
    words, seen = count_words(d["body"] for d in synthetic_docs(200, seed=99))
    return SubwordTokenizer.from_word_counts(words, seen, 320, [END_OF_TEXT])


@pytest.fixture()
def artifact(tmp_path, tokenizer):
    path = tmp_path / "tok" / "tokenizer.json"
    save_artifact(tokenizer, path, "ored-bpe-test", "v1", training={"note": "pytest"})
    return path


@pytest.fixture()
def shards(tmp_path, tokenizer, artifact):
    _, info = load_artifact(artifact)
    sources = [write_jsonl(tmp_path / "a.jsonl.gz", synthetic_docs(120, 1, "a"), compress=True),
               write_jsonl(tmp_path / "b.jsonl", synthetic_docs(120, 2, "b"))]
    settings = BuildSettings(text_field="body", id_field="meta.uid", split_seed=7,
                             fractions={"train": 0.8, "val": 0.1, "test": 0.1}, max_tokens_per_shard=2048)
    out = tmp_path / "shards"
    manifest = build_token_shards(sources, tokenizer, info, out, settings, "pytest-corpus", "v1", log=lambda m: None)
    return out, manifest, sources, settings, info


@pytest.fixture()
def r2():
    FakeS3.objects = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeS3)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield R2Store("acct", "key-id", "secret", bucket="ored-test", endpoint=f"http://127.0.0.1:{server.server_port}")
    server.shutdown()


def test_50m_parameter_count_is_exact_and_embeddings_tied():
    cfg = load_config(CONFIG_50M)
    model = build_model(cfg, vocab_size=cfg.data.vocab_size)
    assert model.num_parameters() == EXPECTED_50M == cfg.model.expected_parameters
    assert abs(model.num_parameters() - 49.9e6) / 49.9e6 < 0.01
    assert model.embeddings_tied and model.head.bias is None
    assert model.head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    d = model.describe()
    assert (d["n_layer"], d["d_model"], d["n_head"], d["d_ff"], d["block_size"], d["vocab_size"]) == \
        (13, 512, 8, 2048, 1024, 16384)
    assert d["d_model"] // d["n_head"] == 64
    model.eval()
    with torch.no_grad():
        out = model(torch.randint(0, 16384, (2, 64)))
    assert out.shape == (2, 64, 16384)


def test_parameter_drift_is_detected():
    cfg = load_config(CONFIG_50M, ["model.n_layer=12"])
    model = build_model(cfg, vocab_size=cfg.data.vocab_size)
    with pytest.raises(PretrainError, match="architecture changed"):
        check_parameter_count(cfg, model.num_parameters())


def test_char_transformer_config_is_gone_and_defaults_stay_manual():
    assert not (CONFIGS / "char_transformer.yaml").exists() and not (CONFIGS / "ored_50m.yaml").exists()
    cfg = load_config(Path(__file__).resolve().parent / "configs" / "language_model.yaml")
    assert cfg.model.attention == "manual" and cfg.checkpoint.policy == "roles"
    model = build_model(cfg, vocab_size=1024)
    assert model.num_parameters() == 5_132_288
    assert set(model.describe()) == {"type", "vocab_size", "block_size", "d_model", "n_layer", "n_head",
                                     "d_ff", "parameters"}


def _pair(**kw):
    torch.manual_seed(0)
    manual = Transformer(vocab_size=50, block_size=32, d_model=32, n_layer=2, n_head=4, d_ff=64,
                         dropout=0.0, attention="manual", **kw)
    sdpa = Transformer(vocab_size=50, block_size=32, d_model=32, n_layer=2, n_head=4, d_ff=64,
                       dropout=0.0, attention="sdpa", **kw)
    sdpa.load_state_dict(manual.state_dict())
    return manual, sdpa


def test_sdpa_matches_manual_attention_and_is_causal():
    manual, sdpa = _pair()
    ids = torch.randint(0, 50, (3, 20))
    manual.eval(), sdpa.eval()
    with torch.no_grad():
        a, b = manual(ids), sdpa(ids)
        assert torch.allclose(a, b, atol=1e-5)
        changed = ids.clone()
        changed[:, 12] = (changed[:, 12] + 1) % 50
        c = sdpa(changed)
    assert torch.allclose(b[:, :12], c[:, :12], atol=1e-6)
    assert not torch.allclose(b[:, 12], c[:, 12], atol=1e-6)


def test_gradient_checkpointing_gives_the_same_gradients():
    plain, _ = _pair()
    torch.manual_seed(0)
    ckpt = Transformer(vocab_size=50, block_size=32, d_model=32, n_layer=2, n_head=4, d_ff=64, dropout=0.0,
                       attention="sdpa", gradient_checkpointing=True)
    ckpt.load_state_dict(plain.state_dict())
    ids = torch.randint(0, 50, (2, 16))
    for model in (plain, ckpt):
        model.train()
        torch.nn.functional.cross_entropy(model(ids).view(-1, 50), ids.view(-1)).backward()
    for (name, p), q in zip(plain.named_parameters(), ckpt.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-5), name


def test_special_token_is_never_produced_from_text(tokenizer):
    eos = tokenizer.special_id(END_OF_TEXT)
    assert eos == tokenizer.vocab_size - 1 == 319
    ids = tokenizer.encode("a dog <|endoftext|> sings")
    assert eos not in ids
    assert tokenizer.decode(ids) == "a dog <|endoftext|> sings"
    assert tokenizer.decode(ids + [eos]) == "a dog <|endoftext|> sings"


def test_tokenizer_without_special_tokens_serialises_as_before():
    old = SubwordTokenizer.from_text("the cat sat on the mat " * 20, vocab_size=280)
    assert "special_tokens" not in old.to_dict()
    assert SubwordTokenizer._from_dict(old.to_dict()).to_dict() == old.to_dict()


def test_artifact_round_trip_and_deterministic_hash(tmp_path, tokenizer, artifact):
    loaded, info = load_artifact(artifact)
    assert loaded.to_dict() == tokenizer.to_dict()
    assert info["sha256"] == tokenizer_sha256(tokenizer) and info["name"] == "ored-bpe-test"
    words, seen = count_words(d["body"] for d in synthetic_docs(200, seed=99))
    again = SubwordTokenizer.from_word_counts(words, seen, 320, [END_OF_TEXT])
    assert tokenizer_sha256(again) == info["sha256"]
    with pytest.raises(TokenizerArtifactError, match="expected"):
        load_artifact(artifact, expected_sha256="0" * 64)


def test_tampered_artifact_is_refused(artifact):
    data = json.loads(artifact.read_text())
    data["tokenizer"]["merges"][0] = ["x", "y"]
    artifact.write_text(json.dumps(data))
    with pytest.raises(TokenizerArtifactError, match="damaged"):
        load_artifact(artifact)


def test_format_detection_and_inspection_show_fields_not_content(tmp_path):
    gz = write_jsonl(tmp_path / "x.jsonl.gz", synthetic_docs(3), compress=True)
    plain = write_jsonl(tmp_path / "unknown.bin", synthetic_docs(3))
    assert (detect_format(gz).format, detect_format(gz).compression) == ("jsonl", "gzip")
    assert detect_format(plain).format is None
    report = inspect_file(gz)
    assert report["fields"] == {"body": ["str"], "extra": ["float"], "meta.uid": ["str"]}
    assert "fox" not in json.dumps(report) and report["records_sampled"] == 3
    with pytest.raises(ReaderError, match="pass the format"):
        open_reader(plain, "body")
    assert len(list(open_reader(plain, "body", file_format="jsonl"))) == 3


def test_readers_need_explicit_fields(tmp_path):
    path = write_jsonl(tmp_path / "x.jsonl", [{"body": "hi"}, {"body": ""}, {"other": 1}])
    with pytest.raises(ReaderError, match="text_field"):
        JsonlReader(path, "")
    reader = JsonlReader(path, "body")
    assert [d.text for d in reader] == ["hi"] and reader.skipped == 2
    with pytest.raises(ReaderError, match="has no 'uid'"):
        list(JsonlReader(path, "body", id_field="uid"))


def test_optional_formats_fail_clearly_without_their_package(tmp_path):
    zst = tmp_path / "x.jsonl.zst"
    zst.write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 16)
    pq = tmp_path / "x.parquet"
    pq.write_bytes(b"PAR1" + b"\x00" * 16)
    try:
        import zstandard
    except ImportError:
        with pytest.raises(ReaderError, match="pip install zstandard"):
            list(open_reader(zst, "body"))
    try:
        import pyarrow
    except ImportError:
        with pytest.raises(ReaderError, match="pip install pyarrow"):
            list(open_reader(pq, "body"))


def test_shards_manifest_counts_and_windows(shards, tokenizer):
    out, manifest, sources, _, info = shards
    assert manifest["tokenizer"]["sha256"] == info["sha256"] and manifest["dtype"] == "uint16"
    assert sum(manifest["counts"]["documents"].values()) == 240
    assert all(manifest["counts"]["documents"][s] > 0 for s in ("train", "val", "test"))
    assert len([s for s in manifest["shards"] if s["split"] == "train"]) > 2
    assert not (out / ".progress.json").exists()
    assert load_manifest(out)["content_sha256"] == manifest["content_sha256"]

    dataset = TokenWindowDataset(out, manifest, "train", block_size=16, stride=16)
    x, y = dataset[len(dataset) - 1]
    assert x.dtype == torch.int64 and x.shape == y.shape == (16,) and torch.equal(x[1:], y[:-1])
    first = manifest["shards"][0]
    import numpy as np
    raw = np.fromfile(out / first["path"], dtype="<u2")
    x0, y0 = TokenWindowDataset(out, manifest, first["split"], 16, 16)[0]
    assert x0.tolist() == raw[:16].tolist() and y0.tolist() == raw[1:17].tolist()
    assert raw.tolist().count(tokenizer.special_id()) == first["documents"]
    clone = pickle.loads(pickle.dumps(dataset))
    assert torch.equal(clone[3][0], dataset[3][0])


def test_shard_build_is_deterministic_and_resumable(shards, tmp_path, tokenizer, monkeypatch):
    out, manifest, sources, settings, info = shards
    import ored.data.token_shards as ts
    real = ts.tokenize_unit
    calls = {"n": 0}

    def flaky(unit, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated crash in unit 1")
        return real(unit, *args, **kwargs)

    monkeypatch.setattr(ts, "tokenize_unit", flaky)
    again = tmp_path / "again"
    with pytest.raises(KeyboardInterrupt):
        build_token_shards(sources, tokenizer, info, again, settings, "pytest-corpus", "v1", log=lambda m: None)
    assert (again / ".progress.json").exists() and not (again / "manifest.json").exists()
    resumed = build_token_shards(sources, tokenizer, info, again, settings, "pytest-corpus", "v1",
                                 log=lambda m: None)
    assert calls["n"] == 3
    assert resumed["content_sha256"] == manifest["content_sha256"]
    with pytest.raises(ShardError, match="immutable"):
        build_token_shards(sources, tokenizer, info, again, settings, "pytest-corpus", "v1", log=lambda m: None)


def test_split_is_by_document_id(shards, tokenizer, tmp_path):
    out, manifest, sources, settings, info = shards
    reordered = [write_jsonl(tmp_path / "c.jsonl", list(reversed(synthetic_docs(120, 1, "a"))))]
    m2 = build_token_shards(reordered, tokenizer, info, tmp_path / "c", settings, "x", "v1", log=lambda m: None)
    a_only = [write_jsonl(tmp_path / "d.jsonl", synthetic_docs(120, 1, "a"))]
    m3 = build_token_shards(a_only, tokenizer, info, tmp_path / "d", settings, "x", "v1", log=lambda m: None)
    assert m2["counts"]["documents"] == m3["counts"]["documents"]


def test_corrupted_shard_is_detected(shards):
    out, manifest, *_ = shards
    shard = out / manifest["shards"][0]["path"]
    data = bytearray(shard.read_bytes())
    data[5] ^= 0xFF
    shard.write_bytes(bytes(data))
    verify_shards(out, manifest, "size")
    with pytest.raises(ShardError, match="sha256"):
        verify_shards(out, manifest, "full")


ENV = {"ORED_R2_ACCOUNT_ID": "acct123", "ORED_R2_ACCESS_KEY_ID": "AKIA-SECRET-ID",
       "ORED_R2_SECRET_ACCESS_KEY": "very-secret-value", "ORED_R2_BUCKET": "ored-ai-data"}


def test_r2_settings_never_show_secrets():
    settings = R2Settings.from_env(ENV)
    shown = repr(settings) + json.dumps(settings.describe())
    for secret in ("AKIA-SECRET-ID", "very-secret-value", "acct123"):
        assert secret not in shown
    assert settings.prefix == "ored-ai" and settings.store().bucket == "ored-ai-data"
    assert settings.checkpoint_store().bucket == "ored-ai-data"
    split = R2Settings.from_env({**ENV, "ORED_R2_CHECKPOINT_BUCKET": "ored-checkpoints"})
    assert split.checkpoint_store().bucket == "ored-checkpoints" and split.store().bucket == "ored-ai-data"
    with pytest.raises(StorageConfigError) as missing:
        R2Settings.from_env({"ORED_R2_SECRET_ACCESS_KEY": "very-secret-value"})
    assert "ORED_R2_BUCKET" in str(missing.value) and "very-secret-value" not in str(missing.value)
    assert not R2Settings.configured({})


def test_object_layout():
    layout = ObjectLayout()
    assert layout.dataset_manifest("common-pile", "v0.1-1gb") == "ored-ai/datasets/common-pile/v0.1-1gb/manifest.json"
    assert layout.dataset_file("common-pile", "v0.1-1gb", "shard-000.parquet") == \
        "ored-ai/datasets/common-pile/v0.1-1gb/shard-000.parquet"
    assert layout.checkpoint_best("ored50m") == "ored-ai/checkpoints/ored50m/best.pt"
    assert layout.tokenizer_artifact("ored-bpe-16k", "v1") == "ored-ai/tokenizers/ored-bpe-16k/v1/tokenizer.json"
    for bad in ("..", "a/b", "", "../x"):
        with pytest.raises(LayoutError):
            layout.dataset_dir(bad, "v1")
    with pytest.raises(LayoutError):
        layout.dataset_file("x", "v1", "../../secrets")


def test_token_dataset_round_trip_through_fake_r2(shards, r2, tmp_path):
    out, manifest, *_ = shards
    object_dir = ObjectLayout().token_dir("pytest-corpus", "v1", "ored-bpe-test-v1")
    publish_token_dataset(r2, out, object_dir, log=lambda m: None)
    assert f"{object_dir}/manifest.json" in FakeS3.objects
    fetched = fetch_token_dataset(r2, object_dir, tmp_path / "cache", log=lambda m: None)
    assert fetched["content_sha256"] == manifest["content_sha256"]


def test_replace_object_keeps_one_best(r2, tmp_path):
    key = ObjectLayout().checkpoint_best("ored50m")
    for size in (100, 250):
        path = tmp_path / f"best-{size}.pt"
        path.write_bytes(b"x" * size)
        replace_object(r2, path, key)
    assert list(FakeS3.objects) == [key] and len(FakeS3.objects[key]) == 250


def test_unknown_format_dataset_entry_is_valid_with_nulls():
    entry = DatasetEntry(name="common-pile", version_label="v0.1-1gb", dataset_type="pretraining_corpus",
                         storage_path=ObjectLayout().dataset_dir("common-pile", "v0.1-1gb"),
                         metadata={"licence": "pending", "nested": {"a": [1, 2]}})
    row = entry.to_row()
    for name in ("file_format", "compression", "token_count", "document_count", "sha256", "external_id",
                 "source_id", "size_bytes", "file_name"):
        assert row[name] is None
    assert row["status"] == "registered" and row["manifest"] == {} and row["source"] == "external"
    sql = (MIGRATIONS / "20261005120000_ored_dataset_registry.sql").read_text() + \
        (MIGRATIONS / "20260926120000_ored_training_data.sql").read_text()
    for column in row:
        assert re.search(rf"\b{column}\b", sql), column


def test_registry_entry_refuses_data_and_bad_states():
    with pytest.raises(RegistryError, match="metadata"):
        DatasetEntry("x", "v1", manifest={"text": "x" * (1024 * 1024 + 1)}).to_row()
    with pytest.raises(RegistryError, match="ready dataset needs its sha256"):
        DatasetEntry("x", "v1", status="ready").to_row()
    with pytest.raises(RegistryError, match="storage_path"):
        DatasetEntry("x", "v1", storage_path="../other").to_row()
    with pytest.raises(RegistryError, match="non-negative"):
        DatasetEntry("x", "v1", token_count=-1).to_row()


def test_config_pairs_token_data_with_best_only():
    cfg = load_config(CONFIG_50M)
    assert cfg.data.source == "tokens" and cfg.checkpoint.policy == "best_only"
    assert (cfg.data.batch_size, cfg.training.grad_accum_steps, cfg.data.block_size) == (4, 8, 1024)
    with pytest.raises(ValueError, match="best_only"):
        load_config(CONFIG_50M, ["checkpoint.policy=roles"])
    with pytest.raises(ValueError, match="keep_history"):
        load_config(CONFIG_50M, ["checkpoint.keep_history=true"])
    with pytest.raises(ValueError, match="precision"):
        load_config(CONFIG_50M, ["training.precision=int8"])


def _payload(epoch, value, tokenizer_sha="t" * 64):
    return {"schema": CHECKPOINT_SCHEMA, "model_version": "ored50m", "architecture": {"type": "T"},
            "model_state_dict": {"w": torch.full((3,), float(epoch))}, "epoch": epoch,
            "best": {"metric": "val_loss", "mode": "min", "value": value},
            "tokenizer": {"sha256": tokenizer_sha}, "dataset": {"content_sha256": "d" * 64}, "config": {}}


def test_best_only_keeps_one_file_and_only_improvements(tmp_path):
    best = BestCheckpoint(tmp_path)
    assert best.consider(2.0, lambda: _payload(1, 2.0))
    assert not best.consider(2.5, lambda: pytest.fail("a worse epoch must not even build a payload"))
    assert best.load()["epoch"] == 1
    assert best.consider(1.5, lambda: _payload(3, 1.5))
    assert best.load()["epoch"] == 3 and best.best_value == 1.5
    assert not best.consider(float("nan"), lambda: _payload(4, 0.0))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt"]


def test_failed_save_leaves_the_previous_best(tmp_path, monkeypatch):
    best = BestCheckpoint(tmp_path)
    best.consider(2.0, lambda: _payload(1, 2.0))
    before = best.path.read_bytes()

    def corrupt(payload, path):
        path.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(best, "_write", corrupt)
    with pytest.raises(Exception):
        best.consider(1.0, lambda: _payload(2, 1.0))
    monkeypatch.undo()
    with pytest.raises(BestCheckpointError, match="lacks"):
        best.consider(0.5, lambda: {k: v for k, v in _payload(3, 0.5).items() if k != "tokenizer"})
    assert best.path.read_bytes() == before and best.best_value == 2.0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt"]
    (tmp_path / "best.pt.tmp-999").write_bytes(b"left by a crash")
    BestCheckpoint(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt"]


def test_checkpoint_compatibility_checks(tmp_path):
    payload = _payload(1, 1.0)
    check_compatible(payload, model_version="ored50m", architecture={"type": "T"}, tokenizer_sha256="t" * 64)
    with pytest.raises(BestCheckpointError, match="tokenizer"):
        check_compatible(payload, model_version="ored50m", architecture={"type": "T"}, tokenizer_sha256="u" * 64)
    with pytest.raises(BestCheckpointError, match="model version"):
        check_compatible(payload, model_version="char_transformer", architecture={"type": "T"},
                         tokenizer_sha256="t" * 64)
    with pytest.raises(BestCheckpointError, match="dataset"):
        check_compatible(payload, model_version="ored50m", architecture={"type": "T"},
                         tokenizer_sha256="t" * 64, dataset_sha256="e" * 64)
    torch.save({"format_version": 2, "model_state_dict": {}}, tmp_path / "best.pt")
    with pytest.raises(BestCheckpointError, match="older Ored model"):
        BestCheckpoint(tmp_path).load()


def tiny_cfg(tmp_path, out, artifact, **extra):
    overrides = [
        f"data.tokens.manifest={out}", f"data.tokens.tokenizer={artifact}", "data.vocab_size=320",
        "data.block_size=32", "data.stride=32", "data.batch_size=4", "data.num_workers=0",
        "data.pin_memory=false", "data.persistent_workers=false", "model.d_model=32", "model.n_layer=2",
        "model.n_head=4", "model.d_ff=64", "model.expected_parameters=0", "model.gradient_checkpointing=true",
        "training.epochs=3", "training.grad_accum_steps=2", "training.warmup_steps=2",
        "training.device=cpu", "training.log_every_steps=5", "training.early_stopping_patience=0",
        "training.eval_every_steps=0", "training.eval_windows=0",
        f"paths.checkpoint_dir={tmp_path / 'checkpoints'}",
    ] + [f"{k}={v}" for k, v in extra.items()]
    return load_config(CONFIG_50M, overrides)


def test_pretrain_saves_only_best_and_replaces_it(shards, tmp_path, artifact, r2, monkeypatch):
    out, manifest, *_ = shards
    cfg = tiny_cfg(tmp_path, out, artifact, **{"checkpoint.r2_upload": "true"})
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    trainer = PretrainTrainer(cfg, store=r2)
    assert trainer.precision.name == "fp32" and trainer.steps_per_epoch == -(-len(trainer.loaders["train"]) // 2)
    losses = iter([2.0, 3.0, 1.0])
    monkeypatch.setattr(trainer, "evaluate", lambda split="val": next(losses))
    saved = []
    real_save = trainer.best.save
    monkeypatch.setattr(trainer.best, "save", lambda payload: (saved.append(payload["epoch"]), real_save(payload))[1])
    result = trainer.fit()

    assert saved == [1, 3] and result["best_val_loss"] == 1.0
    run_dir = Path(cfg.checkpoint_dir)
    assert sorted(p.name for p in run_dir.iterdir()) == ["best.pt", "history.json"]
    payload = torch.load(run_dir / "best.pt", weights_only=True)
    assert payload["epoch"] == 3 and payload["model_version"] == "ored50m"
    assert payload["tokenizer"]["sha256"] == manifest["tokenizer"]["sha256"]
    assert payload["dataset"]["content_sha256"] == manifest["content_sha256"]
    assert payload["dataset"]["manifest_sha256"] and payload["dataset"]["sources"][0]["sha256"]
    for key in ("optimizer_state_dict", "scheduler", "config", "code", "architecture", "global_step"):
        assert key in payload
    assert list(FakeS3.objects) == ["ored-ai/checkpoints/ored50m/best.pt"]
    assert FakeS3.objects["ored-ai/checkpoints/ored50m/best.pt"] == (run_dir / "best.pt").read_bytes()

    resumed = PretrainTrainer(tiny_cfg(tmp_path, out, artifact, **{"training.resume": "best",
                                                                   "training.epochs": 4}))
    assert resumed.start_epoch == 4 and resumed.global_step == payload["global_step"]


def test_pretrain_really_trains_with_bf16_autocast(shards, tmp_path, artifact):
    out, *_ = shards
    cfg = tiny_cfg(tmp_path, out, artifact, **{"training.precision": "bf16", "training.epochs": 2,
                                               "training.learning_rate": 0.003, "model.attention": "sdpa"})
    trainer = PretrainTrainer(cfg)
    assert trainer.precision.name == "bf16" and not trainer.precision.use_scaler
    result = trainer.fit()
    history = result["history"]
    assert history[-1]["val_loss"] < history[0]["val_loss"] and Path(result["best_path"]).is_file()


def test_pretrain_refuses_a_mismatched_tokenizer(shards, tmp_path, tokenizer):
    out, *_ = shards
    other = SubwordTokenizer(list(reversed(tokenizer.merges)), tokenizer.seen_bytes, [END_OF_TEXT])
    assert other.vocab_size == tokenizer.vocab_size and tokenizer_sha256(other) != tokenizer_sha256(tokenizer)
    path = tmp_path / "other.json"
    save_artifact(other, path, "other", "v1")
    with pytest.raises(PretrainError, match="tokenized with tokenizer"):
        PretrainTrainer(tiny_cfg(tmp_path, out, path))


def test_pretrain_without_a_dataset_says_so(tmp_path, artifact):
    cfg = load_config(CONFIG_50M, [f"data.tokens.tokenizer={artifact}", "data.vocab_size=320",
                                   "training.device=cpu"])
    with pytest.raises(PretrainError, match="no training dataset is configured"):
        PretrainTrainer(cfg)


def test_corpus_cli_inspect_layout_and_registry_row(tmp_path, capsys, monkeypatch):
    from ored.data.corpus_cli import main

    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    path = write_jsonl(tmp_path / "unknown.jsonl.gz", synthetic_docs(4), compress=True)
    assert main(["inspect", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["file_format"] == "jsonl" and "body" in report["fields"]
    assert main(["layout", "--name", "common-pile", "--version", "v0.1-1gb"]) == 0
    assert json.loads(capsys.readouterr().out)["manifest"] == "ored-ai/datasets/common-pile/v0.1-1gb/manifest.json"
    assert main(["registry-row", "--name", "common-pile", "--version-label", "v0.1-1gb", "--file", str(path)]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["file_format"] == "jsonl" and row["compression"] == "gzip" and row["token_count"] is None
    assert row["size_bytes"] == path.stat().st_size and len(row["sha256"]) == 64 and row["status"] == "registered"
    assert row["storage_path"] == "ored-ai/datasets/common-pile/v0.1-1gb"


def test_chunked_jsonl_matches_one_unit_and_keys_follow_the_file(tmp_path, tokenizer, artifact):
    _, info = load_artifact(artifact)
    source = write_jsonl(tmp_path / "big.jsonl", synthetic_docs(300, 5, "big"))
    whole = BuildSettings(text_field="body", id_field="meta.uid", chunk_bytes=0, max_tokens_per_shard=4096)
    chunked = BuildSettings(text_field="body", id_field="meta.uid", chunk_bytes=8192, max_tokens_per_shard=4096)
    m1 = build_token_shards([source], tokenizer, info, tmp_path / "w", whole, "x", "v1", log=lambda m: None)
    m2 = build_token_shards([source], tokenizer, info, tmp_path / "c", chunked, "x", "v1", workers=2,
                            log=lambda m: None)
    assert len(m1["units"]) == 1 and len(m2["units"]) > 3
    assert m1["counts"] == m2["counts"]
    sha16 = m2["dataset"]["sources"][0]["sha256"][:16]
    assert all(key.startswith(sha16 + "-c") for key in m2["units"])


def test_adding_a_file_reuses_the_tokenized_ones(tmp_path, tokenizer, artifact, monkeypatch):
    _, info = load_artifact(artifact)
    a = write_jsonl(tmp_path / "a.jsonl", synthetic_docs(100, 1, "a"))
    b = write_jsonl(tmp_path / "b.jsonl.gz", synthetic_docs(100, 2, "b"), compress=True)
    settings = BuildSettings(text_field="body", id_field="meta.uid", chunk_bytes=0, max_tokens_per_shard=4096)
    v1 = build_token_shards([a], tokenizer, info, tmp_path / "v1", settings, "x", "v1", log=lambda m: None)
    import ored.data.token_shards as ts
    real, seen = ts.tokenize_unit, []
    monkeypatch.setattr(ts, "tokenize_unit", lambda unit, *rest, **kw: (seen.append(unit), real(unit, *rest, **kw))[1])
    v2 = build_token_shards([a, b], tokenizer, info, tmp_path / "v2", settings, "x", "v2",
                            reuse_from=[tmp_path / "v1"], log=lambda m: None)
    fresh = build_token_shards([a, b], tokenizer, info, tmp_path / "fresh", settings, "x", "v2", log=lambda m: None)
    assert len(seen) == 3 and seen[0].startswith(v2["dataset"]["sources"][1]["sha256"][:16])
    assert v2["content_sha256"] == fresh["content_sha256"] != v1["content_sha256"]
    v3 = build_token_shards([b], tokenizer, info, tmp_path / "v3", settings, "x", "v3",
                            reuse_from=[tmp_path / "v2"], log=lambda m: None)
    assert len(seen) == 3 and sum(v3["counts"]["documents"].values()) == 100


def test_raw_files_in_r2_can_be_listed_inspected_and_fetched(tmp_path, r2):
    from ored.storage.raw import fetch_raw, inspect_object, list_raw

    gz = write_jsonl(tmp_path / "part 1.jsonl.gz", synthetic_docs(50, 3, "p"), compress=True)
    r2.upload(gz, "training data/part 1.jsonl.gz")
    r2.upload(write_jsonl(tmp_path / "other.jsonl", synthetic_docs(2)), "elsewhere/other.jsonl")
    assert [i["key"] for i in list_raw(r2, "training data")] == ["training data/part 1.jsonl.gz"]
    report = inspect_object(r2, "training data/part 1.jsonl.gz", head_bytes=4096)
    assert report["file_format"] == "jsonl" and report["compression"] == "gzip" and "body" in report["fields"]
    fetched = fetch_raw(r2, "training data/", tmp_path / "raw", decompress=True, log=lambda m: None)
    assert [Path(f["path"]).name for f in fetched] == ["part 1.jsonl"]
    assert len(list(open_reader(fetched[0]["path"], "body"))) == 50
    again = fetch_raw(r2, "training data/", tmp_path / "raw", decompress=True, log=lambda m: None)
    assert again[0]["sha256"] == fetched[0]["sha256"]


def test_epoch_sampler_skips_exactly_what_was_trained():
    from ored.training.pretrain import EpochSampler

    sampler = EpochSampler(50, seed=3, shuffle=True)
    sampler.set_epoch(2)
    full = list(sampler)
    sampler.set_epoch(2, skip=20)
    assert list(sampler) == full[20:] and len(sampler) == 30 and sorted(full) == list(range(50))
    sampler.set_epoch(3)
    assert list(sampler) != full


def test_step_evaluation_saves_mid_epoch_and_resumes_there(shards, tmp_path, artifact, monkeypatch):
    out, *_ = shards
    cfg = tiny_cfg(tmp_path, out, artifact, **{"training.eval_every_steps": 7, "training.eval_windows": 20,
                                               "training.epochs": 1})
    trainer = PretrainTrainer(cfg)
    assert len(trainer.val_set) == 20
    losses = iter([3.0, 2.0, 2.5, 2.6] + [9.0] * 100)
    monkeypatch.setattr(trainer, "evaluate", lambda split="val": next(losses))
    trainer.fit()
    steps = [r["global_step"] for r in trainer.history]
    assert steps[:4] == [7, 14, 21, 28] and steps[-1] == trainer.global_step
    payload = torch.load(Path(cfg.checkpoint_dir) / "best.pt", weights_only=True)
    assert payload["global_step"] == 14 and payload["position"] == {
        "epoch": 1, "micro_batch": 28, "epoch_complete": False}
    resumed = PretrainTrainer(tiny_cfg(tmp_path, out, artifact, **{"training.resume": "best",
                                                                   "training.eval_every_steps": 7,
                                                                   "training.epochs": 1}))
    assert (resumed.start_epoch, resumed.skip_micro_batches, resumed.global_step) == (1, 28, 14)
    batches = []
    real = resumed._loss
    monkeypatch.setattr(resumed, "_loss", lambda x, y: (batches.append(1), real(x, y))[1])
    monkeypatch.setattr(resumed, "evaluate", lambda split="val": 5.0)
    resumed.fit()
    assert len(batches) == resumed.micro_batches_per_epoch - 28


def test_init_from_starts_a_new_run_on_another_dataset_version(shards, tmp_path, artifact, monkeypatch):
    out, *_ = shards
    first = PretrainTrainer(tiny_cfg(tmp_path, out, artifact, **{"training.epochs": 1}))
    first.fit()
    best = Path(first.cfg.checkpoint_dir) / "best.pt"
    other = tmp_path / "other"
    import shutil
    shutil.copytree(out, other)
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["dataset"]["version"] = "v2"
    from ored.data.token_shards import content_sha256
    manifest["content_sha256"] = content_sha256(manifest)
    (other / "manifest.json").write_text(json.dumps(manifest))
    cfg = tiny_cfg(tmp_path, other, artifact, **{"training.epochs": 1, "run_name": "ored50m-v2"})
    with pytest.raises(BestCheckpointError, match="dataset"):
        PretrainTrainer(tiny_cfg(tmp_path, other, artifact, **{"training.resume": "best", "training.epochs": 1}))
    cfg.training.init_from = str(best)
    second = PretrainTrainer(cfg)
    assert second.global_step == 0 and second.start_epoch == 1
    for name, tensor in torch.load(best, weights_only=True)["model_state_dict"].items():
        assert torch.equal(second.model.state_dict()[name].cpu(), tensor.cpu())
    second.fit()
    payload = torch.load(Path(cfg.checkpoint_dir) / "best.pt", weights_only=True)
    assert payload["initialised_from"]["path"] == str(best) and payload["dataset"]["version"] == "v2"
