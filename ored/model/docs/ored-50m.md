# Ored AI ~50M: model, data pipeline, storage and checkpoints

`ored_50m` is a new model generation. It does not load, extend or replace the existing
~5.13M `char_transformer` model: that model keeps its config, its Supabase snapshot
pipeline, its tokenizer-in-checkpoint behaviour and its base/live/best/history
checkpoints, unchanged.

| | char_transformer (existing) | ored_50m (new) |
|---|---|---|
| config | `configs/char_transformer.yaml` | `configs/ored_50m.yaml` |
| entry point | `scripts/train.py` | `scripts/pretrain.py` (`scripts/train.py --config configs/ored_50m.yaml` dispatches to it too) |
| data | Supabase `ored_training_data` snapshot, loaded into RAM | pre-tokenized shards (memory-mapped), files in R2 |
| tokenizer | rebuilt from `train.txt` each run, stored in the checkpoint | trained once, a versioned artifact; checkpoints record its sha256 |
| checkpoints | base / live / best / history | `best.pt` only |

## Model

Decoder-only GPT-style transformer, the same `Transformer` class as before
(`src/ored/models/transformer.py`) with two new switches that default to the old behaviour:

* 13 layers, d_model 512, 8 heads x 64, FFN 2048, GELU, pre-LayerNorm
* 1024-token context, learned absolute position embeddings
* 16,384-token vocabulary, token embedding tied to the LM head, no head bias
* dropout 0.1, normal(0, 0.02) initialisation (unchanged from the existing model)
* `model.attention: sdpa` -- `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`,
  which lets PyTorch pick flash / memory-efficient kernels. `manual` (the default, used by
  `char_transformer`) keeps the explicit T x T score matrix. Same parameters, same function
  (a test checks they agree).
* `model.gradient_checkpointing` -- recompute each block in the backward pass.

**Parameters: 49,894,912**, computed by the model itself
(`python scripts/pretrain.py --describe`). `model.expected_parameters` makes training refuse
a config that drifts more than 1% from that, and `tests/test_ored_50m.py` asserts the exact
number.

## Training on an 8 GB GPU (RTX 5060 Laptop)

Defaults in `configs/ored_50m.yaml`:

* micro-batch 4 x 1024 tokens, `grad_accum_steps: 8` -> 32 sequences = 32,768 tokens per
  optimizer step
* `precision: auto` -> bf16 autocast on GPUs that support it (the RTX 50 series does), fp16
  + GradScaler on older GPUs, fp32 on CPU. The GradScaler exists only for fp16.
* `attention: sdpa`, `tf32: true` (fp32 matmuls that remain may use tensor cores)
* fused AdamW on CUDA, betas (0.9, 0.95), weight decay 0.1 on weight matrices only
* DataLoader: 2 workers, pinned memory, `non_blocking` copies, persistent workers
* loss read back from the GPU only every `log_every_steps` optimizer steps
* `compile: false`: `torch.compile` needs Triton, which is not officially available on
  Windows; turn it on and compare tokens/s yourself on Linux.
* `gradient_checkpointing: false`: turn it on if the probe shows the micro-batch you want
  does not fit.

Measured in this repository's CI-like sandbox (CPU only, 4 threads, fp32, the 50M model):

| batch 4 x 1024, forward + backward | peak memory added |
|---|---|
| `attention: manual` | 4.32 GiB |
| `attention: sdpa` | 2.60 GiB |
| `attention: sdpa` + gradient checkpointing | 1.23 GiB |

At batch 1 x 1024 on that CPU, sdpa ran 2.19 s per forward+backward against 2.72 s for manual;
gradient checkpointing added about 6%. **None of this was measured on a GPU.** Measure the real
headroom on the RTX 5060 before the first run:

```
python scripts/pretrain.py --probe-batch-size      # peak VRAM and tok/s per micro-batch size
```

If a run hits CUDA out-of-memory it stops with the batch, accumulation, precision and peak
memory in the message; `best.pt` is not touched. Lower `data.batch_size`, raise
`training.grad_accum_steps` to keep the effective batch, or enable gradient checkpointing.

The RTX 50 series (Blackwell) needs a CUDA 12.8+ build of PyTorch (2.7 or newer), e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu128`.

Every run logs: GPU, VRAM total/free, parameters, micro-batch, sequence length, accumulation,
effective batch, precision, tokens/s, validation loss, best validation loss, epoch and peak
allocated VRAM.

## Data: from an unknown raw file to token shards

The corpus has not been supplied yet, so nothing assumes its format or fields.

```
# 1. look at the file: format, compression, size, field names (no content printed)
python scripts/corpus.py inspect /path/to/corpus.jsonl.zst

# 2. train the tokenizer once (a sample is enough), producing a versioned artifact
python scripts/corpus.py train-tokenizer --input FILE --text-field <field> \
    --vocab-size 16384 --max-bytes 300000000 --name ored-bpe-16k --version v1 \
    --out data/tokenizers/ored-bpe-16k-v1.json

# 3. tokenize into shards (resumable; one unit per input file; --workers N in parallel)
python scripts/corpus.py build-shards --input FILE [FILE ...] --text-field <field> \
    [--id-field <field>] --tokenizer data/tokenizers/ored-bpe-16k-v1.json \
    --name common-pile --version v0.1-1gb --out data/tokens/common-pile/v0.1-1gb/ored-bpe-16k-v1
```

* Readers (`src/ored/data/readers.py`): `DatasetReader` -> `JsonlReader` (`.jsonl`,
  `.jsonl.gz`; `.jsonl.zst` with `zstandard`) and `ParquetReader` (with `pyarrow`). The
  two optional packages are listed under the `corpus` extra in `pyproject.toml` and are not
  installed by default. Format is detected from magic bytes, then the file name.
* Shards (`src/ored/data/token_shards.py`): flat little-endian uint16 token ids, each
  document followed by `<|endoftext|>`. Documents are split train/val/test by a hash of
  their id (or of their text when there is no id field), so the split does not depend on
  file order. Memory stays bounded; an interrupted build resumes from the last finished
  input file.
* `manifest.json` records every shard's sha256 and size, every source file's sha256, the
  tokenizer identity, the split rule and counts, and a `content_sha256` over all of it: the
  dataset version's identity. Training verifies the shards (sha256 by default) before using
  them and reads them through `numpy.memmap`.
* No cleaning, filtering or deduplication is applied yet: those depend on the real corpus.

## Storage: R2 for files, Supabase for metadata

Object layout (`src/ored/storage/layout.py`), under the prefix `ored-ai/`:

```
ored-ai/datasets/<name>/<version>/manifest.json
ored-ai/datasets/<name>/<version>/<raw file(s)>
ored-ai/datasets/<name>/<version>/tokens/<tokenizer-name>-<tokenizer-version>/{manifest.json, train/, val/, test/}
ored-ai/tokenizers/<name>/<version>/tokenizer.json
ored-ai/checkpoints/<run_name>/best.pt
```

Environment variables (never commit values):
`ORED_R2_ACCOUNT_ID` (or `ORED_R2_ENDPOINT`), `ORED_R2_ACCESS_KEY_ID`,
`ORED_R2_SECRET_ACCESS_KEY`, `ORED_R2_BUCKET`, optional `ORED_R2_PREFIX`. The client is the
repository's existing signed S3 client (`ored.learning.r2.R2Store`, no new dependency);
downloads now stream to disk in 4 MiB chunks.

```
python scripts/corpus.py layout --name common-pile --version v0.1-1gb
python scripts/corpus.py publish --dir <shard dir> --name common-pile --version v0.1-1gb
python scripts/corpus.py fetch --name common-pile --version v0.1-1gb --tokenizer-label ored-bpe-16k-v1 --out <dir>
```

Uploading the raw file itself (`scripts/corpus.py upload-raw`) is one object at
`ored-ai/datasets/<name>/<version>/<file>`, never overwritten; R2Store takes up to 5 GiB per object (a 1 GB file
fits; larger files need multipart upload, not implemented yet).

### Dataset registry (Supabase, metadata only)

`public.ored_datasets` already was the registry; migration
`ored/supabase/migrations/20261005120000_ored_dataset_registry.sql` extends it for
`source = 'external'` datasets with nullable columns: `version_label, dataset_type, status,
storage_provider, storage_bucket, file_name, file_format, compression, external_id,
source_id, size_bytes, document_count, token_count`, plus `manifest` and `metadata` (jsonb,
at most 1 MiB / 256 KiB, objects only: metadata, never the data). `sha256`, `storage_path`,
`version`, `created_at`, `updated_at` already existed.

* Lifecycle: `registered -> uploading -> uploaded -> processing -> ready` (or `failed`,
  `deprecated`). `ready` requires a sha256; from then on the row's identity is frozen by a
  trigger, and it can only be deprecated.
* `ored_dataset_register_external(jsonb)` registers by `(name, version_label)` and is
  idempotent.
* Security is unchanged: RLS enabled and forced, no policies, `anon` / `authenticated`
  without privileges, functions executable by `service_role` only.

```
python scripts/corpus.py registry-row --name common-pile --version-label v0.1-1gb [--file FILE]   # prints the row
python scripts/corpus.py register     --name common-pile --version-label v0.1-1gb [--file FILE]   # sends it (ORED_SB_*)
```

## Checkpoints: best.pt only

`src/ored/training/best_checkpoint.py`. After each validated epoch:

* better `val_loss` than the best so far -> write `best.pt.tmp-<pid>`, fsync, load it back
  and check it, then atomically rename over `checkpoints/<run_name>/best.pt`; with
  `checkpoint.r2_upload: true`, replace `ored-ai/checkpoints/<run_name>/best.pt` in R2
* not better -> nothing is written

No per-epoch files, no history, no backups; stale temporaries from a crash are removed at
start. `history.json` (metrics only) sits next to `best.pt`.

`best.pt` holds: schema `ored-best-checkpoint/1`, model version, architecture, weights,
optimizer, scheduler and (fp16 only) scaler state, epoch, global step, best `val_loss`,
metrics history, tokenizer identity (name, version, sha256, vocab), dataset identity (name,
version, registry id, content and manifest sha256, source file sha256s), the full config,
git commit and dirty flag, precision and torch version. Loading checks schema, model version,
architecture, tokenizer and dataset, so an old `char_transformer` checkpoint is refused.
`python scripts/pretrain.py --resume` continues from `best.pt`.

## When the corpus arrives

1. `scripts/corpus.py inspect FILE` and choose `--text-field` / `--id-field`.
2. `scripts/corpus.py register --file FILE ...` (status `registered`), then
   `scripts/corpus.py upload-raw --file FILE --name ... --version ...`, then
   `scripts/corpus.py update-registry ... --set status=ready --set document_count=...` as
   facts become known.
3. Train the tokenizer artifact, build the shards, `publish` them.
4. Set `data.tokens.*` in `configs/ored_50m.yaml` (dataset name / version / registry id,
   tokenizer artifact path), run `--probe-batch-size` on the GPU, then train.
