"""Integer row-sum validation, early exit, and the optional Numba fallback."""

import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from discrete_pid import _validation
from discrete_pid import redundancy_binary_sources, redundancy_binary_target


class RowValidationTests(unittest.TestCase):
    def test_scan_matches_numpy_on_valid_and_invalid_inputs(self):
        rng = np.random.default_rng(819)
        for k, d, m in ((1, 1, 2), (7, 9, 2), (5, 2, 13)):
            counts = rng.multinomial(100, np.ones(m) / m, size=(k, d)).astype(np.uint32)
            support = rng.random(d) > .4
            support[-1] = True
            for data in (counts, counts[:, ::-1], counts[:, :, ::-1]):
                for corrupt in (False, True):
                    candidate = data.copy()
                    if corrupt:
                        candidate[-1, -1, -1] += 1
                    totals = candidate.sum(axis=2, dtype=np.uint64)[:, support]
                    expected = bool(np.all(totals > 0) and np.all(totals == totals[:, :1]))
                    self.assertEqual(_validation._same_row_sums(candidate, support), expected)
                    if _validation._compiled_same_row_sums is not None:
                        self.assertEqual(_validation._compiled_same_row_sums(candidate, support), expected)

    def test_wide_sums_inactive_rows_readonly_and_strided_arrays(self):
        m = np.iinfo(np.uint32).max
        counts = np.array([[[0, 0, 0], [m, m, m], [m, m, m]],
                           [[1, 0, 0], [m, m, 0], [m, 0, m]]], dtype=np.uint32)
        support = np.array([False, True, True])
        before = counts.copy()
        counts.flags.writeable = False
        support.flags.writeable = False
        for data in (counts, counts[:, :, ::-1], counts.astype(np.int64)):
            _validation.validate_row_sums(data, support)
            with patch.object(_validation, '_compiled_same_row_sums', None):
                _validation.validate_row_sums(data, support)
        np.testing.assert_array_equal(counts, before)
        with self.assertRaisesRegex(ValueError, 'same positive row sum'):
            _validation.validate_row_sums(np.zeros((1, 3, 2), dtype=np.uint32), support)

    def test_python_scan_exits_on_first_mismatch(self):
        class UnreadTail:
            shape = (2, 2, 2)

            def __getitem__(self, index):
                i, y, x = index
                if i != 0:
                    raise AssertionError('scan continued after the mismatch')
                return np.uint32(1 + y)

        self.assertFalse(_validation._same_row_sums(UnreadTail(), np.ones(2, dtype=bool)))

    def test_both_solvers_reject_invalid_rows_before_geometry_early_return(self):
        # The first source is uninformative, but the later invalid channel
        # must still be rejected instead of returning zero redundancy.
        counts = np.array([[[5, 5], [5, 5]], [[3, 7], [3, 8]]], dtype=np.uint32)
        for solver in (redundancy_binary_sources, redundancy_binary_target):
            with self.subTest(solver=solver.__name__):
                with self.assertRaisesRegex(ValueError, 'same positive row sum'):
                    solver([1, 1], counts)
                with patch.object(_validation, '_compiled_same_row_sums', None):
                    with self.assertRaisesRegex(ValueError, 'same positive row sum'):
                        solver([1, 1], counts)

    @unittest.skipIf(_validation._compiled_same_row_sums is None, 'Numba is optional')
    def test_compiles_at_import(self):
        self.assertTrue(_validation._compiled_same_row_sums.signatures)

    def test_no_numba_warns_once_and_uses_python(self):
        code = '''
import sys
import warnings
sys.modules['numba'] = None
with warnings.catch_warnings(record=True) as messages:
    warnings.simplefilter('always')
    from discrete_pid import redundancy_binary_sources, redundancy_binary_target, _validation
    assert _validation._compiled_same_row_sums is None
    for solver in (redundancy_binary_sources, redundancy_binary_target):
        for _ in range(2):
            assert solver([1, 1], [[[9, 1], [1, 9]]]).redundancy_bits > 0
    notices = [w for w in messages if 'Numba is unavailable' in str(w.message)]
    assert len(notices) == 1, messages
    assert 'discrete-pid[speed]' in str(notices[0].message)
'''
        run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == '__main__':
    unittest.main()
