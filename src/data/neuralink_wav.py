"""Lazy WAV reader for the *unlabeled* Neuralink Compression Challenge archive.

Every WAV is treated as an independent recording. The archive does not supply
reliable channel coordinates, cross-file timing, text, or behavioral labels.
File-wise splits are diagnostic only (NOT independent subjects or sessions).
"""
from __future__ import annotations

import argparse
import hashlib
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def inspect_wavs(root: str | Path) -> list[dict]:
    paths = sorted(Path(root).glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No .wav files in {root}. Extract data.zip into this directory.")
    manifest = []
    for path in paths:
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise ValueError(f"Expected mono, uncompressed 16-bit PCM: {path}")
            manifest.append({
                "path": str(path),
                "name": path.name,
                "rate": wav.getframerate(),
                "frames": wav.getnframes(),
            })
    return manifest


def split_recording(name: str, seed: int = 42) -> str:
    # Deterministic *file* split, without assuming a separate participant/session.
    fraction = int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16) / 2**32
    return "train" if fraction < 0.8 else ("val" if fraction < 0.9 else "test")


class NeuralinkWavDataset(Dataset):
    """Yields fixed-length windows [samples] while preserving the original sample rate."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        window_samples: int = 16384,
        patch_size: int = 256,
        seed: int = 42,
        manifest: list[dict] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if window_samples <= 0 or patch_size <= 0 or window_samples % patch_size:
            raise ValueError("window_samples must be positive and divisible by patch_size")
        self.seed = seed
        self.epoch = 0
        self.split = split
        self.window_samples = window_samples
        self.manifest = [
            row for row in (manifest if manifest is not None else inspect_wavs(root))
            if row["frames"] >= window_samples and split_recording(row["name"], seed) == split
        ]
        if not self.manifest:
            raise RuntimeError(f"No usable files in {split} split: {root}")

    def __len__(self) -> int:
        return len(self.manifest)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        entry = self.manifest[index]
        with wave.open(entry["path"], "rb") as wav:
            remaining = wav.getnframes() - self.window_samples
            if self.split == "train":
                digest = hashlib.sha256(
                    f"{self.seed}:{self.epoch}:{entry['name']}".encode()
                ).digest()
                start = int.from_bytes(digest[:8], "little") % (remaining + 1)
            else:
                start = remaining // 2
            wav.setpos(start)
            raw = wav.readframes(self.window_samples)
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if len(samples) != self.window_samples:
            raise IOError(f"Short WAV read for {entry['path']}")
        return {
            "waveform": torch.from_numpy(samples),
            "sample_rate": int(entry["rate"]),
            "file": entry["name"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect extracted Neuralink WAV headers")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    rows = inspect_wavs(args.root)
    print(f"Files: {len(rows)}")
    print(f"Sample rates: {dict(sorted(Counter(r['rate'] for r in rows).items()))}")
    print(f"Min/max seconds: {min(r['frames']/r['rate'] for r in rows):.3f} / "
          f"{max(r['frames']/r['rate'] for r in rows):.3f}")
    print("WARNING: UUID-named WAVs are not documented as separate subjects/sessions.")


if __name__ == "__main__":
    main()
