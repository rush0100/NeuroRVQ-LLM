"""Fast CPU smoke checks; no Neuralink archive needed."""
from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch

from src.data.neuralink_wav import NeuralinkWavDataset, inspect_wavs, split_recording
from src.models.neuralink_ssl import NeuralinkMaskedEncoder


class NeuralinkSSLTests(unittest.TestCase):
    def test_mask_and_token_shapes(self):
        model = NeuralinkMaskedEncoder(
            window_samples=1024, patch_size=128, embed_dim=64, depth=1,
            num_heads=4, rush_dim=768, rush_queries=24,
        )
        x = torch.randn(2, 1024)
        train = model(x, mask_seed=123)
        self.assertTrue(torch.isfinite(train["loss"]))
        self.assertEqual(train["mask"].shape, (2, 8))
        self.assertEqual(train["mask"].sum(dim=1).tolist(), [4, 4])
        self.assertTrue(torch.equal(train["mask"], model(x, mask_seed=123)["mask"]))
        model.eval()
        self.assertEqual(model.encode_rush_tokens(x).shape, (2, 24, 768))

    def test_header_read_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(25):
                with wave.open(str(root / f"sample_{i:02d}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(19531)
                    w.writeframes((np.arange(1024, dtype="<i2")).tobytes())
            records = inspect_wavs(root)
            self.assertEqual(len(records), 25)
            self.assertEqual({r["rate"] for r in records}, {19531})
            self.assertEqual(split_recording(records[0]["name"], 42), split_recording(records[0]["name"], 42))
            split = split_recording(records[0]["name"], 42)
            ds = NeuralinkWavDataset(root, split=split, window_samples=512, patch_size=128)
            self.assertEqual(ds[0]["waveform"].shape, (512,))
            self.assertTrue(torch.isfinite(ds[0]["waveform"]).all())


if __name__ == "__main__":
    unittest.main()
