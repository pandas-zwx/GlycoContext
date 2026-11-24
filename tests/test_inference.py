import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glycocontext.inference import build_scan_features, normalize_sequence, score_tracks
from glycocontext.model import GlycoContext


class InferenceTests(unittest.TestCase):
    def test_sequence_normalization(self):
        self.assertEqual(normalize_sequence(" acgt\n"), "ACGU")
        with self.assertRaises(ValueError):
            normalize_sequence("ACGR")

    def test_two_track_scan_shapes(self):
        sequence = "ACGUACGU"
        centers = np.arange(len(sequence), dtype=np.int64)
        structure = {
            "structure": "((....))",
            "probabilities": [.7, .3],
            "RNA_version": "2.7.2",
        }
        views = np.zeros((len(centers), 3, 640), dtype=np.float32)
        features = build_scan_features(sequence, centers, structure, views)
        model = GlycoContext("full").eval()
        checkpoint = {
            "thresholds": {"0": .4, "1": .6},
        }
        rows = score_tracks(model, checkpoint, features, ["HeLa", "H9"], "cpu", 3)
        self.assertEqual(len(rows), 2 * len(sequence))
        self.assertEqual([row["position1"] for row in rows[:len(sequence)]], list(range(1, 9)))
        self.assertEqual({row["cell"] for row in rows}, {"HeLa", "H9"})
        for row in rows:
            self.assertAlmostEqual(
                row["logit"], row["cnn_logit"] + row["gnn_logit"] + row["rnafm_logit"], places=5
            )


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
