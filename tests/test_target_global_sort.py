"""Batched posterior ordering checked against exact, directly summed calls."""

from fractions import Fraction
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from discrete_pid import _target_scan, binary_target
from discrete_pid import redundancy_binary_target as solve


def exact_call_hull(prior, channels):
    """Small integer oracle: sum each call directly using rational arithmetic.

    No posterior sorting, suffix accumulators, or production hull code is used
    to evaluate the calls. Python integers also make large-count cases exact.
    """
    p0, p1 = (Fraction(int(p), sum(map(int, prior))) for p in prior)
    knots = {Fraction(0): p1, Fraction(1): Fraction(0)}
    for channel in channels:
        total = sum(map(int, channel[0]))
        law = []
        for a, b in zip(*channel):
            mass = p0 * int(a) + p1 * int(b)
            if mass:
                law.append((p1 * int(b) / mass, mass / total))
        for t, _ in law:
            height = sum(w * max(q - t, 0) for q, w in law)
            knots[t] = min(knots.get(t, height), height)
    hull = []
    for t, height in sorted(knots.items()):
        while len(hull) > 1:
            a, b = hull[-2:]
            if (b[0] - a[0]) * (height - b[1]) > (b[1] - a[1]) * (t - b[0]):
                break
            hull.pop()
        hull.append((t, height))
    return np.asarray(hull, dtype=float).T


class GlobalSortTests(unittest.TestCase):
    def assert_exact_calls(self, prior, channels, result):
        x, y = exact_call_hull(prior, channels)
        support, weights = result.posteriors[1], result.posterior_weights
        # Checking both sets of corners compares the whole piecewise-linear
        # call function, without requiring roundoff-sized atoms to coincide.
        grid = np.unique(np.concatenate((x, support)))
        calls = np.maximum(support[:, None] - grid, 0).T @ weights
        np.testing.assert_allclose(calls, np.interp(grid, x, y), rtol=0, atol=3e-13)

    def test_uint32_calls_with_ties_endpoints_and_different_source_scales(self):
        maximum = np.iinfo(np.uint32).max
        tied = np.array([[[2, 6, 0, 0], [1, 3, 4, 0]],
                         [[8, 0, 4, 4], [8, 0, 6, 2]],
                         [[0, 0, 6, 6], [3, 9, 0, 0]]], dtype=np.uint32)
        large = np.array([[[maximum, maximum, 0], [0, maximum, maximum]],
                          [[maximum, 0, maximum], [maximum, maximum, 0]]],
                         dtype=np.uint32)
        rng = np.random.default_rng(547)
        random = rng.multinomial(257, np.full(7, 1 / 7), size=(9, 2)).astype(np.uint32)
        random *= np.arange(1, 10, dtype=np.uint32)[:, None, None]
        for data in (tied, large, random):
            before = data.copy()
            for entries in (4, 32, 10000):
                with self.subTest(shape=data.shape, entries=entries), \
                     patch.object(binary_target, '_BATCH_ENTRIES', entries):
                    result = solve([2, 3], data, return_channel=True)
                    self.assert_exact_calls([2, 3], data, result)
            np.testing.assert_array_equal(data, before)

    def test_readonly_strides_ragged_iterators_and_optional_outputs(self):
        packed = np.array([[[2, 6, 0, 0], [1, 3, 4, 0]],
                           [[4, 0, 2, 2], [4, 0, 3, 1]]], dtype=np.uint32)
        packed = packed[::-1, :, ::-1]
        packed.setflags(write=False)
        ragged = [packed[0], np.array([[9, 1, 0], [1, 9, 0]], dtype=np.uint32)]
        for channels in (packed, ragged):
            reference = solve([2, 3], channels, return_channel=True)
            self.assert_exact_calls([2, 3], channels, reference)
            for as_iterator in (False, True):
                for channel_flag, garbling_flag in ((False, False), (True, False),
                                                    (False, True), (True, True)):
                    data = iter(channels) if as_iterator else channels
                    with self.subTest(iterator=as_iterator,
                                      flags=(channel_flag, garbling_flag)), \
                         patch.object(binary_target, '_BATCH_ENTRIES', 8):
                        result = solve([2, 3], data, return_channel=channel_flag,
                                       return_garblings=garbling_flag)
                    self.assertAlmostEqual(result.redundancy_nats,
                                           reference.redundancy_nats, places=12)
                    self.assertEqual(result.channel is not None, channel_flag)
                    self.assertEqual(result.garblings is not None, garbling_flag)
                    if channel_flag:
                        self.assert_exact_calls([2, 3], channels, result)
                    if garbling_flag:
                        self.assertEqual(len(result.garblings), len(channels))
                        for counts, kernel in zip(channels, result.garblings):
                            joint = np.array([.4, .6])[:, None] * counts / counts[0].sum()
                            np.testing.assert_allclose(joint @ kernel,
                                                       reference.target_auxiliary_joint,
                                                       rtol=0, atol=2e-12)
                            np.testing.assert_allclose(np.asarray(kernel.sum(axis=1)).ravel(),
                                                       1, rtol=0, atol=2e-12)

    def test_float_fallback_handles_signed_zero_and_independent_row_scales(self):
        counts = np.array([[[5, 5, 0], [0, 1, 9]],
                           [[4, 5, 1], [1, 4, 5]]], dtype=np.uint32)
        floating = counts.astype(float)
        floating[0, 1, 0] = -0.0
        floating *= np.array([1e306, 1e-300])[None, :, None]
        before = floating.copy()
        with patch.object(binary_target, '_BATCH_ENTRIES', 8):
            result = solve([3, 2], floating, return_channel=True)
        self.assert_exact_calls([3, 2], counts, result)
        np.testing.assert_array_equal(floating, before)

    def test_almost_degenerate_prior_still_validates_every_batch(self):
        invalid = np.array([[[3, 7], [6, 4]], [[1, 0], [2, 0]]], dtype=np.uint32)
        for prior in ([1e-20, 1], [1, 1e-20]):
            for data in (invalid, iter(invalid)):
                with self.subTest(prior=prior), \
                     patch.object(binary_target, '_BATCH_ENTRIES', 4), \
                     self.assertRaisesRegex(ValueError, 'same positive row sum'):
                    solve(prior, data)

    def test_raw_batches_are_bounded_without_normalized_joint_arrays(self):
        channels = np.tile(np.array([[[9, 1], [1, 9]]], dtype=np.uint32), (101, 1, 1))
        prepare = binary_target._prepare_channels
        for data in (channels, iter(channels)):
            seen = []

            def checked(raw, prior):
                seen.append(raw.size)
                self.assertLessEqual(raw.size, 32)
                return prepare(raw, prior)

            with patch.object(binary_target, '_BATCH_ENTRIES', 32), \
                 patch.object(binary_target, '_prepare_channels', side_effect=checked), \
                 patch.object(binary_target, '_normalize_channels',
                              side_effect=AssertionError('unnecessary joint array')):
                result = solve([1, 1], data, return_channel=True)
            self.assert_exact_calls([1, 1], channels[:1], result)
            self.assertEqual(sum(seen), channels.size)

    def test_python_kernels_match_compiled_solver(self):
        channels = np.array([[[2, 6, 0, 0], [1, 3, 4, 0]],
                             [[4, 0, 2, 2], [4, 0, 3, 1]]], dtype=np.uint32)
        for data in (channels, channels.astype(float)):
            with patch.object(binary_target, '_BATCH_ENTRIES', 8):
                compiled = solve([2, 3], data, return_channel=True, return_garblings=True)
                with patch.object(_target_scan, '_compiled_posteriors', None), \
                     patch.object(_target_scan, '_compiled_scan', None), \
                     patch.object(_target_scan, '_compiled_orders', None):
                    python = solve([2, 3], data, return_channel=True, return_garblings=True)
            self.assert_exact_calls([2, 3], channels, python)
            self.assertAlmostEqual(compiled.redundancy_nats, python.redundancy_nats, places=13)
            self.assertLess(python.max_garbling_residual, 2e-12)

    @unittest.skipIf(_target_scan._compiled_scan is None, 'Numba is optional')
    def test_target_kernels_are_compiled_at_import(self):
        code = ('from discrete_pid import _target_scan\n'
                'assert _target_scan._compiled_posteriors.signatures\n'
                'assert _target_scan._compiled_scan.signatures\n'
                'assert _target_scan._compiled_orders.signatures\n')
        run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    @unittest.skipUnless(np.finfo(np.longdouble).eps < np.finfo(float).eps,
                         'longdouble is not wider than float64 on this platform')
    def test_knot_scan_preserves_extended_precision(self):
        theta = np.array([0, .5, .5, 1], dtype=np.longdouble)
        theta[2] += 4 * np.finfo(np.longdouble).eps
        self.assertEqual(float(theta[1]), float(theta[2]))
        values = np.stack((1 - theta, theta))[None, :, :] / 4
        with patch.object(_target_scan, '_compiled_scan',
                          side_effect=AssertionError('downcast')):
            x, _ = _target_scan.call_knots(values, theta, np.arange(4, dtype=np.int64),
                                            np.ones(1), np.ones(2))
        self.assertEqual(x.dtype, np.dtype(np.longdouble))
        np.testing.assert_array_equal(x, theta)


if __name__ == '__main__':
    unittest.main()
