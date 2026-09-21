"""Check nonlinear aggregation order, grouping, normalization and missing entities."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgp_align.profile_aggregation import encode_replicate_mean


class AggregationTest(unittest.TestCase):
    def test_order_and_grouping(self):
        x = np.array([[1., 2.], [3., 1.], [2., 4.]], dtype=np.float32)
        def encode(a):
            z = a ** 2
            return z / np.linalg.norm(z, axis=1, keepdims=True)
        result, counts = encode_replicate_mean(x, [2, 0, 1], [1, 0, 0], 2, encode, batch_size=1)
        expected = np.stack([encode(x[:2]).mean(axis=0), encode(x[2:])[0]])
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(result, expected, atol=1e-7)
        np.testing.assert_array_equal(counts, [2, 1])
        self.assertGreater(np.linalg.norm(result[0] - encode(x[:2].mean(axis=0, keepdims=True))[0]), .01)
        with self.assertRaises(ValueError):
            encode_replicate_mean(x, [0], [0], 2, encode)


if __name__ == '__main__':
    unittest.main()
