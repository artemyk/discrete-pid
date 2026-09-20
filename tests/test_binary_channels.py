"""Conditional-input validation, integer bounds, and an exact posterior oracle."""

from fractions import Fraction as F
from functools import partial
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from discrete_pid import redundancy_binary_sources, redundancy_binary_target
from discrete_pid import _source_scan

# These existing oracle/witness tests explicitly request both optional outputs.
solve = partial(redundancy_binary_sources, return_channel=True, return_garblings=True)
M = 2**32 - 1


def joints(prior, channels):
    p = np.array(prior, dtype=float, copy=True)
    p /= p.sum()
    c = np.asarray(channels, dtype=float)
    total = c.sum(axis=2, keepdims=True)
    return list(p[None, :, None] * np.divide(c, total, out=np.zeros_like(c), where=total > 0))


def posterior_oracle(prior, channels):
    """Independent Fraction implementation: intersect posterior intervals."""
    p = [F(int(v), sum(map(int, prior))) for v in prior]
    posterior, intervals = [], []
    reference = None
    for table in channels:
        joint = [[p[y] * F(int(v), sum(map(int, row))) for v in row]
                 for y, row in enumerate(table)]
        mass = [sum(row[x] for row in joint) for x in (0, 1)]
        if 0 in mass:
            return np.array([[float(v)] for v in p])
        columns = [[row[x] / mass[x] for row in joint] for x in (0, 1)]
        delta = [r - q for r, q in zip(columns[1], p)]
        if not any(delta):
            return np.array([[float(v)] for v in p])
        if reference is None:
            reference = delta
            pivot = next(y for y, v in enumerate(delta) if v)
        if any(v * reference[pivot] != reference[y] * delta[pivot]
               for y, v in enumerate(delta)):
            return np.array([[float(v)] for v in p])
        posterior.append(columns)
        intervals.append([(column[pivot] - p[pivot]) / reference[pivot]
                          for column in columns])
    lower = max(min(pair) for pair in intervals)
    upper = min(max(pair) for pair in intervals)
    result = []
    for t, w in ((lower, upper / (upper - lower)), (upper, -lower / (upper - lower))):
        result.append([float(w * (q + t * v)) for q, v in zip(p, reference)])
    return np.array(result).T


class ChannelTests(unittest.TestCase):
    def assert_witness(self, prior, channels, result):
        self.assertTrue(np.isfinite(result.redundancy_nats))
        self.assertTrue(np.all(result.posteriors >= 0))
        self.assertTrue(np.all(result.posterior_weights > 0))
        np.testing.assert_allclose(result.posteriors.sum(axis=0), 1, atol=1e-12)
        np.testing.assert_allclose(result.target_auxiliary_joint.sum(axis=1),
                                   np.asarray(prior) / np.sum(prior), atol=1e-12)
        for joint, kernel in zip(joints(prior, channels), result.garblings):
            self.assertTrue(np.all(kernel >= 0))
            np.testing.assert_allclose(kernel.sum(axis=1), 1, rtol=0, atol=1e-14)
            np.testing.assert_allclose(joint @ kernel, result.target_auxiliary_joint,
                                       rtol=0, atol=1e-12)

    def assert_oracle(self, prior, channels):
        result = solve(prior, channels)
        expected = posterior_oracle(prior, channels)
        actual = result.target_auxiliary_joint
        if expected.shape[1] == 2:
            # Pick a coordinate that distinguishes the two posterior endpoints.
            pivot = np.argmax(np.abs(expected[:, 0] / expected[:, 0].sum()
                                     - expected[:, 1] / expected[:, 1].sum()))
            expected = expected[:, np.argsort(expected[pivot] / expected.sum(axis=0))]
            actual = actual[:, np.argsort(result.posteriors[pivot])]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
        self.assert_witness(prior, channels, result)
        return result

    def test_analytical_example_and_input_containers(self):
        c = np.array([[[4, 26], [16, 14], [10, 20]],
                      [[14, 16], [26, 4], [20, 10]]], dtype=np.uint32)
        p = np.array([1., 1., 1.])
        original = c.copy()
        for channels in (c, list(c), iter(c), c.tolist(), c.astype(object), c.astype(np.int64)):
            result = solve(p, channels)
            self.assertAlmostEqual(result.redundancy_bits, 0.043954629749672874, places=13)
            self.assert_witness(p, c, result)
        np.testing.assert_array_equal(c, original)
        np.testing.assert_array_equal(p, [1, 1, 1])
        c.flags.writeable = False
        self.assert_oracle([2, 3, 5], c)

    def test_nonuniform_prior_rescaling_and_relabeling(self):
        c = np.array([[[4, 26], [16, 14], [10, 20]],
                      [[14, 16], [26, 4], [20, 10]]], dtype=np.uint32)
        first = self.assert_oracle([2, 3, 5], c)
        changed = c[::-1, [2, 0, 1], ::-1].copy() * np.array([7, 11], dtype=np.uint32)[:, None, None]
        second = self.assert_oracle([5, 2, 3], changed)
        self.assertAlmostEqual(first.redundancy_nats, second.redundancy_nats, places=13)

    def test_uint64_products_beyond_signed_range(self):
        c = np.array([[[M, 0], [0, M], [M - 1, 1]]], dtype=np.uint32)
        c = np.concatenate((c, c[:, :, ::-1]), axis=0)
        self.assert_oracle([3, 2, 7], c)
        # The low-variation channel forces endpoint comparisons near M**2.
        c = np.array([[[1, M - 1], [0, M]], [[M, 0], [0, M]]], dtype=np.uint32)
        self.assert_oracle([1, 1], c)

    def test_determinant_difference_of_one_is_not_rounded_away(self):
        c = np.array([[[M, 0], [1, M - 1], [2, M - 2]],
                      [[M, 0], [0, M], [1, M - 1]]], dtype=np.uint32)
        self.assertEqual((M - 1)**2 - M * (M - 2), 1)
        self.assertEqual(float((M - 1)**2), float(M * (M - 2)))
        for tolerance in (None, 1):
            result = solve([1, 1, 1], c, tolerance=tolerance)
            self.assertEqual(result.redundancy_nats, 0)
            self.assert_witness([1, 1, 1], c, result)
        self.assertGreater(solve([1, 1, 1], c.astype(float), tolerance=1e-12).redundancy_nats, 0)

    def test_nearly_tied_endpoints_preserve_positive_garbling_entry(self):
        h = M // 2
        c = np.array([[[h, h - 1], [0, 2*h - 1]],
                      [[h + 1, h], [0, 2*h + 1]]], dtype=np.uint32)
        result = self.assert_oracle([2, 3], c)
        self.assertGreater(result.garblings[0][0, 1], 0)
        exact = (F(h, h + 1) - F(h - 1, h)) / (1 + F(h, h + 1))
        self.assertAlmostEqual(result.garblings[0][0, 1] / float(exact), 1, places=14)
        np.testing.assert_array_equal(result.garblings[1], np.eye(2))

    def test_row_sum_may_exceed_uint32(self):
        c = np.array([[[M, M - 2], [M - 2, M]]], dtype=np.uint32)
        self.assert_oracle([1, 2], c)

    def test_zero_prior_and_uninformative_sources(self):
        for dtype in (np.uint32, float):
            c = np.array([[[0, 0], [5, 0], [0, 5]],
                          [[M, M], [0, 9], [9, 0]]], dtype=dtype)
            kwargs = {} if dtype == np.uint32 else {'tolerance': 1e-12}
            result = solve([0, 2, 8], c, **kwargs)
            self.assert_witness([0, 2, 8], c, result)
            for independent in ([[0, 0], [5, 0], [5, 0]],
                                [[0, 0], [2, 3], [2, 3]]):
                result = solve([0, 2, 8], [c[0], np.array(independent, dtype=dtype)], **kwargs)
                self.assertEqual(result.redundancy_nats, 0)
        self.assertEqual(solve([1], [[[3, 7]]]).redundancy_nats, 0)

    def test_integer_input_validation_before_early_return(self):
        valid = np.array([[1, 1], [1, 1]])  # Uninformative, but still validate all inputs.
        for bad in (np.array([[1, 1], [1, 2]]), np.zeros((2, 2), dtype=int),
                    np.array([[-1, 3], [1, 1]]),
                    np.full((2, 2), M + 1, dtype=np.uint64),
                    np.full((2, 2), 10**400, dtype=object)):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    solve([1, 1], [valid, bad])
        for bad in ([], [np.ones((3, 2), dtype=int)], [np.ones((2, 3), dtype=int)],
                    [np.ones((2, 1), dtype=int)], [[[True, False], [False, True]]],
                    [[[1j, 0], [0, 1]]], [[['1', '0'], ['0', '1']]]):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    solve([1, 1], bad)
        with self.assertRaisesRegex(ValueError, 'same positive row sum'):
            solve([1, 1], [[[1, 1], [2, 2]]])

    def test_prior_and_tolerance_validation(self):
        for prior in ([], [0, 0], [-1, 2], [np.nan, 1], [np.inf, 1], [[1, 1]], [1j, 1]):
            with self.subTest(prior=prior):
                with self.assertRaises(ValueError):
                    solve(prior, [[[1, 0], [0, 1]]])
        for tolerance in (-1, np.nan, np.inf, True, '1e-12', 1j):
            with self.assertRaisesRegex(ValueError, 'tolerance must'):
                solve([1, 1], [[[1, 0], [0, 1]]], tolerance=tolerance)
        # Floating priors do not require a collinearity tolerance.
        self.assertGreater(solve([0.2, 0.8], [[[1, 0], [0, 1]]]).redundancy_nats, 0)

    def test_float_opt_in_and_weight_normalization(self):
        c = np.array([[[4, 26], [16, 14], [10, 20]]], dtype=np.uint32)
        reference = solve([1, 1, 1], c)
        for data in (c.astype(float), c.astype(float).tolist(), [c[0], c[0].astype(float)]):
            with self.assertRaisesRegex(ValueError, 'explicit numerical tolerance'):
                solve([1, 1, 1], data)
            result = solve([1, 1, 1], data, tolerance=1e-12)
            self.assertAlmostEqual(result.redundancy_nats, reference.redundancy_nats, places=13)
        for c in (np.array([[[8e307, 12e307], [12e307, 8e307]]]),
                  np.array([[[.4, .6], [.6, .4]]], dtype=object),
                  np.array([[[.4, .6], [.6, .4]]], dtype=np.float32)):
            result = solve([8e307, 8e307], c, tolerance=1e-12)
            self.assertTrue(np.isfinite(result.redundancy_nats))
        for value in (np.nan, np.inf, -np.inf, -1.):
            with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                solve([1, 1], [[[value, 1], [1, 1]]], tolerance=1e-12)

    def test_random_channels_against_exact_posterior_oracle(self):
        rng = np.random.default_rng(482)
        for d in (2, 3, 8, 32):
            for _ in range(10):
                p = rng.integers(1, 100, size=d)
                v = rng.integers(0, 100, size=d)
                channels = []
                for _ in range(4):
                    a, b, h = map(int, rng.integers(1, 100, size=3))
                    c = np.column_stack((b + h * (100 - v), a + h * v))
                    if rng.integers(2):
                        c = c[:, ::-1]
                    channels.append(c)
                self.assert_oracle(p, channels)
                if d > 2:
                    channels[-1] = channels[-1].copy()
                    channels[-1][0] += [1, -1]
                    self.assert_oracle(p, channels)

    def test_two_algorithms_agree_for_random_binary_target(self):
        rng = np.random.default_rng(726)
        for _ in range(30):
            p = rng.integers(1, 100, size=2)
            total = rng.integers(1, M + 1, size=(4, 1), dtype=np.uint64)
            a = rng.integers(0, total + 1, size=(4, 2), dtype=np.uint64)
            c = np.stack((a, total - a), axis=-1).astype(np.uint32)
            result = self.assert_oracle(p, c)
            other = redundancy_binary_target(joints(p, c))
            self.assertAlmostEqual(result.redundancy_nats, other.redundancy_nats, places=11)

    def test_python_fallback_matches_compiled_scan(self):
        c = np.array([[[M, 0], [0, M], [M - 1, 1]],
                      [[0, M], [M, 0], [1, M - 1]]], dtype=np.uint32)
        expected = solve([1, 2, 3], c)
        with patch.object(_source_scan, '_compiled_scan', None):
            actual = self.assert_oracle([1, 2, 3], c)
        np.testing.assert_array_equal(actual.target_auxiliary_joint, expected.target_auxiliary_joint)
        np.testing.assert_array_equal(actual.garblings, expected.garblings)

    @unittest.skipIf(_source_scan._compiled_scan is None, 'Numba is optional')
    def test_source_scan_compiles_at_import(self):
        run = subprocess.run([sys.executable, '-c',
                              'from discrete_pid import _source_scan; '
                              'assert _source_scan._compiled_scan.signatures'],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == '__main__':
    unittest.main()
