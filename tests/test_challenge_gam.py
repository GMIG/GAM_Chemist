import math
import unittest

from scripts.challenge_gam import molecule_descriptors, mixture_descriptors, metrics, encode_rows


class ChallengeTests(unittest.TestCase):
    def test_molecular_transfer_and_mixture_weighting(self):
        dec = molecule_descriptors('CCOC(=O)OCC')
        dmc = molecule_descriptors('COC(=O)OC')
        ec = molecule_descriptors('O=C1OCCO1')
        self.assertGreater(dec['mw'], dmc['mw'])
        self.assertAlmostEqual(dec['tpsa'], dmc['tpsa'])
        self.assertEqual(dec['rings'], 0)
        self.assertEqual(ec['rings'], 1)
        mix = mixture_descriptors(['CCOC(=O)OCC', 'O=C1OCCO1'], [0.3, 0.7])
        self.assertAlmostEqual(mix['mix_mw'], 0.3 * dec['mw'] + 0.7 * ec['mw'])
        self.assertAlmostEqual(mix['mix_rings'], 0.7)
        with self.assertRaises(ValueError):
            mixture_descriptors(['CCOC(=O)OCC'], [0.5])

    def test_metrics_use_log_targets_without_clipping_and_label_trimmed_score(self):
        result = metrics([-4, 0, 1], [-3, 1, 1])
        self.assertAlmostEqual(result['rmse'], math.sqrt(2 / 3))
        self.assertAlmostEqual(result['bias'], -2 / 3)
        self.assertAlmostEqual(result['trimmed_rmse'], math.sqrt(1 / 2))
        self.assertEqual(result['trimmed_n'], 2)
        self.assertEqual(result['excluded_from_trimmed'], 1)

    def test_unseen_salt_has_no_fitted_salt_adjustment(self):
        row = {'temperature_K': 298, 'conc_native': 1, 'conc_unit': 'mol/kg',
               'salt_name': 'unseen', 'log_k': 123, 'source_doi': 'not-a-predictor'}
        matrix, names = encode_rows([row], ['LiPF6', 'LiBF4'], False)
        self.assertEqual(matrix[0, -2:].tolist(), [0, 0])
        self.assertNotIn('log_k', names)
        self.assertNotIn('source_doi', names)


if __name__ == '__main__':
    unittest.main()
