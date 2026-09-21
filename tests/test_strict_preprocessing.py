"""Held-out perturbation must not change fitted parameters or transformed training rows."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'scripts')]
from cgp_align.strict_preprocessing import fit_transform
import build_compound_mocop_corrected_profile_dataset as original_compound
import build_gene_mocop_corrected_profile_dataset as original_gene


class StrictFit(unittest.TestCase):
    def test_heldout_invariance_and_historical_math(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(12, 5)).astype(np.float32)
        reps = pd.DataFrame(dict(entity_index=np.arange(12), perturbation_modality=['orf', 'crispr'] * 6,
                                 Metadata_Plate=['A', 'B'] * 5 + ['UNSEEN', 'UNSEEN'],
                                 Metadata_Well=['1', '2', '3'] * 4))
        fit = np.arange(8)
        for kind, old in [('compound', original_compound), ('gene', original_gene)]:
            baseline, parameters, _ = fit_transform(x, reps, fit, kind)
            changed = x.copy(); changed[8:] = 1e6
            result, new_parameters, _ = fit_transform(changed, reps, fit, kind)
            self.assertEqual(set(parameters), set(new_parameters))
            for key in parameters: np.testing.assert_array_equal(parameters[key], new_parameters[key])
            np.testing.assert_array_equal(baseline[fit], result[fit])
            expected, _ = old.source_train_zscore(x, reps, fit)
            for cycle in range(2):
                for column in ('Metadata_Plate', 'Metadata_Well'):
                    expected, _ = old.subtract_train_label_means(expected, reps[column].to_numpy(), fit, column)
            np.testing.assert_allclose(baseline, expected, atol=1e-6, rtol=1e-6)
            self.assertFalse(any('UNSEEN' in v.tolist() for k, v in parameters.items() if k.endswith('_keys')))


if __name__ == '__main__': unittest.main()
