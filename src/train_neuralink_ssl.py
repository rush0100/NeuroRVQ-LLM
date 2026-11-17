"""Train and evaluate masked reconstruction on unlabeled Neuralink WAV recordings.

This diagnostic file-level validation does NOT establish cross-subject transfer.
Do not publish Neuralink waveforms/weights without checking data permissions.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.neuralink_wav import NeuralinkWavDataset, inspect_wavs, split_recording
from src.models.neuralink_ssl import NeuralinkMaskedEncoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="datasets/neuralink_raw/data")
    p.add_argument("--output-dir", type=str, default="checkpoints/neuralink_ssl")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--window-samples", type=int, default=16384)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    manifest = inspect_wavs(args.data_root)
    rates = {r["rate"] for r in manifest}
    if len(rates) != 1:
        raise ValueError(f"Mixed sample rates {sorted(rates)}. Revisit resampling/feature design.")
    train = NeuralinkWavDataset(
        args.data_root, split="train", window_samples=args.window_samples,
        patch_size=args.patch_size, seed=args.seed, manifest=manifest,
    )
    val = NeuralinkWavDataset(
        args.data_root, split="val", window_samples=args.window_samples,
        patch_size=args.patch_size, seed=args.seed, manifest=manifest,
    )
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model_kwargs = {
        "window_samples": args.window_samples, "patch_size": args.patch_size,
        "embed_dim": args.embed_dim, "depth": args.depth, "mask_ratio": args.mask_ratio,
    }
    model = NeuralinkMaskedEncoder(**model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")

    for epoch in range(args.epochs):
        train.set_epoch(epoch)
        model.train()
        train_loss = []
        for batch in train_loader:
            signals = batch["waveform"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = model(signals)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss.append(float(loss.detach()))
        model.eval()
        val_loss = []
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                signals = batch["waveform"].to(device)
                # Fixed masks and file windows for consistent validation.
                val_loss.append(float(model(signals, mask_seed=12345 + bi)["loss"]))
        report = {
            "epoch": epoch + 1, "train_masked_huber": float(np.mean(train_loss)),
            "val_masked_huber": float(np.mean(val_loss)),
        }
        print(json.dumps(report))
        history.append(report)
        if report["val_masked_huber"] < best:
            best = report["val_masked_huber"]
            torch.save({
                "state_dict": model.state_dict(), "model_kwargs": model_kwargs,
                "epoch": epoch + 1, "val_masked_huber": best,
                "warning": "RUSH projection head NOT trained for cross-modal semantic alignment",
            }, out / "best.pt")

    (out / "history.json").write_text(json.dumps({
        "history": history, "sample_rate": list(rates)[0],
        "files_by_split": {
            split: sum(split_recording(r["name"], args.seed) == split for r in manifest)
            for split in ("train", "val", "test")
        },
        "limitations": [
            "This is a file-wise split; recording/session/subject overlap is unknown.",
            "No text/behavior labels exist in this dataset.",
            "RUSH-shaped latent interface is NOT aligned to ZuCo EEG embeddings.",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
