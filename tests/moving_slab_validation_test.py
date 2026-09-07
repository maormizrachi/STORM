"""The checker must reject lost spectral bands, invalid values and normalization errors."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

checker_path = Path(__file__).parents[1] / 'examples/moving_slab_mc_32/check_moving_slab_mc_32.py'
spec = importlib.util.spec_from_file_location('moving_slab_check', checker_path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

class ValidationTest(unittest.TestCase):
    def setUp(self):
        bounds = np.geomspace(.01, 100, 33)
        self.lower, self.upper = bounds[:-1], bounds[1:]
        self.reference = np.ones(32)
    def check(self, code):
        return checker.ValidationErrors(code, self.reference, self.lower, self.upper)
    def test_reference(self):
        self.assertEqual(self.check(self.reference.copy()), [])
    def test_missing_band(self):
        code = self.reference.copy()
        code[:8] = 0
        self.assertLess(checker.EnergyWeightedFractionalError(code, self.reference), .3)
        self.assertTrue(any('band' in error for error in self.check(code)))
    def test_bad_normalization(self):
        self.assertTrue(self.check(self.reference * .5))
    def test_invalid_values(self):
        for value in (np.nan, np.inf, -1):
            code = self.reference.copy()
            code[0] = value
            self.assertTrue(self.check(code))

if __name__ == '__main__':
    unittest.main()
