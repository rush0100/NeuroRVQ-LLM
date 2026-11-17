# Neuralink -> SSL Encoder -> RUSH Representation (experimental)

**Status:** research prototype, no results claimed. This branch does not modify the ZuCo
pipeline or RUSH Stage-1/Stage-2 weights. Neuralink Compression Challenge is an
unlabeled intracortical recording corpus from a **non-human primate**. It is not scalp
EEG, and it does not provide paired text or verified multichannel topology.

## Sources

- Data mirror: https://github.com/DovRine/neuralink-compression-challenge/blob/main/data.zip
- Challenge details: https://github.com/DovRine/neuralink-compression-challenge
- Primary RUSH pipeline: ../README.md

Inspect dataset permissions before redistributing data or checkpoints. The upstream
repository has no declared license for these signals. This branch does not bundle data.

## Run on DGX Spark / Linux

From the RUSH-EEG2Text repository root:

```bash
# Download data.zip from the linked source after checking permissions.
mkdir -p datasets/neuralink_raw
unzip /path/to/data.zip -d datasets/neuralink_raw
# Expected extracted folder: datasets/neuralink_raw/data/*.wav

python -m src.data.neuralink_wav --root datasets/neuralink_raw/data
python -m unittest discover -s tests -p 'test_neuralink_ssl.py'

python -m src.train_neuralink_ssl \
  --data-root datasets/neuralink_raw/data \
  --output-dir checkpoints/neuralink_ssl \
  --epochs 20 --batch-size 16 \
  --window-samples 16384 --patch-size 256 \
  --embed-dim 256 --depth 4 --mask-ratio 0.5
```

Dependencies: Python, PyTorch, NumPy. CPU smoke tests do not require the archive.

## Model and boundaries

1. Individual 16-bit mono WAV files are read lazily and validated. Files are never
   concatenated into fictitious 1024-channel samples.
2. Fixed windows are split into 256-sample patches.
3. Masked patch reconstruction trains a small transformer with Huber loss on
   **per-patch-normalized** targets.
4. `model.encode_rush_tokens(waveform)` exports a **[B,24,768] shape-compatible
   interface only**. Its projection parameters are **not trained by the SSL loss**,
   and tokens are **not semantically aligned with RUSH EEG or Qwen**. Do not pass
   these directly to the RUSH text decoder or claim EEG-to-text performance.

No text labels, motor labels, electrode coordinates, verified simultaneous-channel
grouping, or session/subject identifiers are assumed.

### Evaluation caveat

Deterministic hash-based **file** splitting prevents a filename from occurring in
both train and validation. It does NOT guarantee independent recording sessions or
subjects. Report this as an *engineering validation*, not cross-subject generalization.

## Next scientifically valid alignment study

To establish a transfer claim, obtain properly labeled **intracortical-to-text**
data, or a dataset containing time-aligned scalp and invasive neural recordings.
Train and evaluate a *separately supervised* adapter using appropriate paired data.

Keep the existing ZuCo-only RUSH results untouched and compare:
- A: ZuCo-only RUSH Stage 1/Stage 2 baseline
- B: ZuCo-only EEG SSL pretraining (compute/data matched)
- C: Neuralink SSL encoder + explicitly trained cross-modal adapter on valid pairs
- D: Randomly initialized Neuralink encoder + same adapter

Use held-out subjects, matched training budgets, repeated seeds, noise sanity controls,
BLEU/ROUGE/WER, and retrieval metrics where applicable. Any proposed cross-modal
alignment without paired samples must be framed as exploratory domain adaptation,
not validated transfer of meaning.
