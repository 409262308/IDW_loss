"""CPU-only checks of the recorder. Does NOT test a Taiwan model checkpoint.
Run: python test_feature_recorder.py
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from feature_recorder import UNetFeatureRecorder


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.SiLU())
        self.mid = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.SiLU())
        self.out = nn.Conv2d(4, 1, 1)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.out(self.mid(self.enc(x)))


def toy_trajectory(model, x, recorder=None):
    """Only a test driver; NOT EDM or a physical diffusion model."""
    for step in range(3):
        if recorder is None:
            d = model(x)
        else:
            with recorder.capture(step=step, sigma_cur=3-step,
                                  sigma_eval=3-step+0.1, sigma_next=2-step,
                                  residual_input=x):
                d = model(x)
                recorder.record_denoised(d)
        x = x * 0.7 + d * 0.3
        # Still run the second model call, but intentionally do not cache it.
        if step != 2:
            d2 = model(x)
            x = x * 0.9 + d2 * 0.1
    return x


class RecorderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = ToyNet().eval()
        self.x = torch.randn(1, 1, 8, 6)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_only_shapes_and_state(self):
        with torch.no_grad():
            expected = toy_trajectory(self.model, self.x.clone())
            rng_before = torch.get_rng_state().clone()
            with UNetFeatureRecorder(
                self.model, ["enc.1", "mid.1"], self.root / "member_000",
                num_steps=3, spatial_steps=[0, 2], capture_inputs=True,
                metadata={"sample_id": "toy", "member_id": 0},
            ) as recorder:
                actual = toy_trajectory(self.model, self.x.clone(), recorder)
                result = recorder.finish(prediction=actual, final_residual=actual)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(self.model.calls, 10)
        meta = json.loads((result / "metadata.json").read_text())
        self.assertTrue(meta["complete"])
        self.assertEqual(len(meta["records"]), 3)
        self.assertEqual(len(meta["layers"]), 4)
        summary = np.load(result / "layer_00_output_summary.npy", mmap_mode="r", allow_pickle=False)
        spatial = np.load(result / "layer_00_output_spatial.npy", mmap_mode="r", allow_pickle=False)
        self.assertEqual(summary.shape, (3, 4, 4))
        self.assertEqual(spatial.shape, (2, 4, 8, 6))
        self.assertEqual(summary.dtype, np.float32)
        self.assertEqual(spatial.dtype, np.float16)
        self.assertTrue(np.isfinite(summary).all())
        self.assertEqual(len(self.model.enc[1]._forward_hooks), 0)
        self.assertFalse((self.root / "member_000.partial").exists())
        # Check signed and magnitude statistics against an independent forward.
        with torch.no_grad():
            feat = self.model.enc(self.x)[0]
            np.testing.assert_allclose(summary[0, :, 0], feat.mean((-2, -1)).numpy())
            np.testing.assert_allclose(summary[0, :, 1], feat.abs().mean((-2, -1)).numpy())

    def test_missing_step_not_committed(self):
        with UNetFeatureRecorder(self.model, ["mid.1"], self.root / "partial", num_steps=3) as rec:
            with self.assertRaises(RuntimeError):
                rec.finish(prediction=self.x, final_residual=self.x)
        self.assertFalse((self.root / "partial").exists())
        self.assertTrue((self.root / "partial.partial").exists())

    def test_reject_batch_and_remove_hooks(self):
        with self.assertRaises(ValueError):
            with UNetFeatureRecorder(self.model, ["mid.1"], self.root / "batch", num_steps=1) as rec:
                with rec.capture(step=0, sigma_cur=1, sigma_eval=1, sigma_next=0,
                                 residual_input=self.x.repeat(2, 1, 1, 1)):
                    pass
        self.assertEqual(len(self.model.mid[1]._forward_hooks), 0)

    def test_reject_train_mode(self):
        self.model.train()
        with self.assertRaises(ValueError):
            UNetFeatureRecorder(self.model, ["mid.1"], self.root / "train", num_steps=1)

    def test_reject_missing_module(self):
        with self.assertRaises(KeyError):
            UNetFeatureRecorder(self.model, ["missing"], self.root / "bad", num_steps=1)

    def test_reject_duplicate_step(self):
        with UNetFeatureRecorder(self.model, ["mid.1"], self.root / "dup", num_steps=2) as rec:
            with rec.capture(step=0, sigma_cur=1, sigma_eval=1, sigma_next=0.5,
                             residual_input=self.x):
                rec.record_denoised(self.model(self.x))
            with self.assertRaises(ValueError):
                with rec.capture(step=0, sigma_cur=1, sigma_eval=1, sigma_next=0.5,
                                 residual_input=self.x):
                    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
