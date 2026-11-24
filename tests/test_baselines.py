import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.baselines import encode_windows, PositionDinucleotideProfile, assemble_rnafm, inner_folds, class_weights


class BaselineTests(unittest.TestCase):
    def test_sequence_encoding_dimensions_and_padding(self):
        records = [dict(sequence_sha256='x', rna_index0=0)]
        onehot, knf, codes = encode_windows(records, {'x': 'ACGU'*20})
        self.assertEqual(onehot.shape, (1, 204)); self.assertEqual(knf.shape, (1, 336))
        self.assertEqual(codes.shape, (1, 50)); self.assertTrue((codes[0, :25] == -1).all())
        self.assertEqual(float(onehot[0, :100].sum()), 0.)
        np.testing.assert_allclose([knf[0, :16].sum(), knf[0, 16:80].sum(), knf[0, 80:].sum()], 1)

    def test_profile_own_group_has_no_label_influence(self):
        codes = np.array([[0]*50, [1]*50, [2]*50, [3]*50])
        groups = np.array(['a', 'a', 'b', 'b'])
        p = PositionDinucleotideProfile().fit(codes, [1, 0, 0, 1], groups)
        q = PositionDinucleotideProfile().fit(codes, [0, 1, 0, 1], groups)
        np.testing.assert_array_equal(p.transform(codes[:2], groups[:2]), q.transform(codes[:2], groups[:2]))
        self.assertFalse(np.array_equal(p.transform(codes[:2]), q.transform(codes[:2])))

    def test_tree_dimensions_and_no_training_statistics(self):
        rng = np.random.default_rng(0)
        args = [rng.normal(size=(3, 3, 640)).astype('f'), rng.normal(size=(3, 640)).astype('f'),
            np.array([10, 20, 30]), np.array([0, 7, 29]), np.array([0, 1, 0])]
        for processing in ('raw', 'token_LN'):
            x = assemble_rnafm(*args, processing)
            self.assertEqual(x.shape, (3, 2564))
            np.testing.assert_array_equal(x[:1], assemble_rnafm(*[v[:1] for v in args], processing))
        np.testing.assert_array_equal(x[:, -2:], np.eye(2)[args[-1]])

    def test_group_folds_and_class_weights(self):
        groups = np.array([f'g{i//2}' for i in range(12)])
        folds = inner_folds(groups, 'dev0')
        self.assertTrue((folds[::2] == folds[1::2]).all())
        np.testing.assert_array_equal(folds, inner_folds(groups, 'dev0'))
        np.testing.assert_allclose(class_weights(np.array([1, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 1])), [2, 1, 1, 1, 1])


if __name__ == '__main__': unittest.main()
