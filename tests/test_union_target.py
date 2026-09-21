"""Binary-target joins, checked against calls and full coupling optimization."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np

from discrete_pid.union_binary_target import union_binary_target as solve
from discrete_pid import _union_target_scan, binary_target


def information(joint):
    positive = joint > 0
    denominator = joint.sum(axis=1)[:, None] * joint.sum(axis=0)
    return np.sum(joint[positive] * np.log(joint[positive] / denominator[positive]))


def normalized_joint(prior, channel):
    channel = np.asarray(channel, dtype=float)
    totals = channel.sum(axis=1, keepdims=True)
    return np.asarray(prior)[:, None] * np.divide(
        channel, totals, out=np.zeros_like(channel), where=totals > 0)


class UnionTargetTests(unittest.TestCase):
    def assert_calls(self, prior, channels, result):
        support, weights = result.posteriors[1], result.posterior_weights
        laws = []
        for channel in channels:
            joint = normalized_joint(prior, channel)
            mass = joint.sum(axis=0)
            active = mass > 0
            laws.append((joint[1, active] / mass[active], mass[active]))
        grid = np.unique(np.concatenate([np.linspace(0, 1, 301), support]
                                        + [r for r, _ in laws]))
        expected = np.max([np.maximum(r[:, None] - grid, 0).T @ w
                           for r, w in laws], axis=0)
        actual = np.maximum(support[:, None] - grid, 0).T @ weights
        np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-13)
        self.assertLessEqual(len(weights), sum(c.shape[1] for c in channels) - len(channels) + 1)
        np.testing.assert_allclose(result.channel.sum(axis=1), 1, atol=2e-12)
        np.testing.assert_allclose(result.target_auxiliary_joint.sum(axis=1), prior,
                                   rtol=0, atol=2e-12)

    def test_bsc_bec_analytic_join_and_decoders(self):
        channels = [np.array([[9, 1], [1, 9]], dtype=np.uint32),
                    np.array([[5, 0, 5], [0, 5, 5]], dtype=np.uint32)]
        result = solve([1, 1], channels, return_channel=True, return_garblings=True)
        expected = np.log(2) + .5 * (.2 * np.log(.2) + .8 * np.log(.8))
        self.assertAlmostEqual(result.union_nats, expected, places=13)
        np.testing.assert_allclose(result.posteriors[1], [0, .2, .8, 1], atol=1e-14)
        np.testing.assert_allclose(result.posterior_weights, .25, atol=1e-14)
        for channel, decoder in zip(channels, result.garblings):
            self.assertEqual(decoder.shape, (4, channel.shape[1]))
            np.testing.assert_allclose(np.asarray(decoder.sum(axis=1)).ravel(), 1, atol=1e-13)
            np.testing.assert_allclose(result.channel @ decoder, channel / 10, atol=1e-13)
        self.assertLess(result.max_garbling_residual, 1e-13)
        self.assertEqual(result.gap_nats, 0)
        self.assertEqual(result.lower_bound_nats, result.union_nats)
        self.assertEqual(result.upper_bound_nats, result.union_nats)

    def test_random_call_envelopes_and_source_decoding(self):
        rng = np.random.default_rng(761)
        for k, m in ((1, 1), (1, 8), (2, 3), (7, 5), (11, 2)):
            channels = rng.dirichlet(np.ones(m), size=(k, 2))
            result = solve([2, 3], channels, return_channel=True, return_garblings=True)
            self.assert_calls(np.array([.4, .6]), channels, result)
            for channel, decoder in zip(channels, result.garblings):
                np.testing.assert_allclose(result.channel @ decoder, channel, atol=2e-11)
                self.assertLessEqual(decoder.nnz, 3 * (m + len(result.posterior_weights)))
            self.assertGreaterEqual(result.union_nats + 1e-13,
                                    max(information(.4 * c * [[1], [1.5]]) for c in channels))

    def test_full_coupling_optimization(self):
        try:
            import cvxpy as cp
        except ImportError:
            self.skipTest('CVXPY is an optional independent full-coupling oracle')
        rng = np.random.default_rng(7003)
        for sizes in ((2, 2), (2, 3), (2, 2, 2), (3, 3)):
            p = np.array([.37, .63])
            channels = [rng.dirichlet(np.ones(m), size=2) for m in sizes]
            states = np.array(list(itertools.product(*(range(m) for m in sizes))))
            q = cp.Variable((2, len(states)), nonneg=True)
            mass = cp.sum(q, axis=0, keepdims=True)
            constraints = [cp.sum(q, axis=1) == p]
            for i, channel in enumerate(channels):
                for x in range(sizes[i] - 1):
                    constraints.append(cp.sum(q[:, states[:, i] == x], axis=1) == p * channel[:, x])
            problem = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(q, p[:, None] @ mass))), constraints)
            problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10,
                          tol_gap_rel=1e-10, max_iter=300)
            self.assertIn(problem.status, ('optimal', 'optimal_inaccurate'))
            self.assertAlmostEqual(solve(p, channels).union_nats, problem.value, places=7)

    def test_ties_endpoints_zero_columns_and_large_counts(self):
        maximum = np.iinfo(np.uint32).max
        families = [np.array([[[2, 6, 0, 0], [1, 3, 4, 0]],
                              [[8, 0, 4, 4], [8, 0, 6, 2]],
                              [[0, 0, 6, 6], [3, 9, 0, 0]]], dtype=np.uint32),
                    np.array([[[maximum, maximum, 0], [0, maximum, maximum]],
                              [[maximum, 0, maximum], [maximum, maximum, 0]]], dtype=np.uint32)]
        for channels in families:
            original = channels.copy()
            result = solve([2, 3], channels, return_channel=True, return_garblings=True)
            self.assert_calls(np.array([.4, .6]), channels, result)
            for channel, decoder in zip(channels, result.garblings):
                np.testing.assert_allclose(result.target_auxiliary_joint @ decoder,
                                           normalized_joint([.4, .6], channel), atol=2e-12)
            np.testing.assert_array_equal(channels, original)

    def test_ragged_iterators_readonly_strides_and_output_flags(self):
        packed = np.array([[[9, 1, 0], [1, 9, 0]],
                           [[5, 0, 5], [0, 5, 5]]], dtype=np.uint32)[:, :, ::-1]
        packed.setflags(write=False)
        ragged = [packed[0], np.array([[9, 1], [1, 9]], dtype=np.uint32)]
        for channels in (packed, ragged):
            expected = solve([2, 3], channels, return_channel=True)
            for return_channel, return_garblings in itertools.product((False, True), repeat=2):
                for data in (channels, iter(channels)):
                    result = solve([2, 3], data, return_channel=return_channel,
                                   return_garblings=return_garblings)
                    self.assertAlmostEqual(result.union_nats, expected.union_nats, places=13)
                    for name in ('channel', 'posteriors', 'posterior_weights', 'target_auxiliary_joint'):
                        self.assertEqual(getattr(result, name) is not None, return_channel)
                    self.assertEqual(result.garblings is not None, return_garblings)
                    self.assertEqual(result.max_garbling_residual is not None, return_garblings)

    def test_independent_row_scales_and_zero_prior(self):
        channel = np.array([[.6, .4, 0], [.2, .8, 0]])
        scaled = channel * np.array([1e308, 1e-300])[:, None]
        original = scaled.copy()
        result = solve([2e300, 3e300], [scaled])
        self.assertAlmostEqual(result.union_nats, information(np.array([.4, .6])[:, None] * channel),
                               places=13)
        np.testing.assert_array_equal(scaled, original)
        counts = np.array([[3, 7, 0], [0, 0, 0]], dtype=np.uint32)
        result = solve([1, 0], [counts], return_channel=True, return_garblings=True)
        self.assertEqual(result.union_nats, 0)
        np.testing.assert_array_equal(result.channel, np.ones((2, 1)))
        np.testing.assert_allclose(result.garblings[0].toarray(), [[.3, .7, 0]])
        with self.assertRaises(ValueError):
            solve([1, 0], [counts, np.array([[1., 0], [np.nan, 0]])])

    def test_invalid_inputs(self):
        good = np.array([[[3, 7], [6, 4]]], dtype=np.uint32)
        for prior in ([], [1], [1, 2, 3], [0, 0], [-1, 2], [np.nan, 1], [1j, 1]):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                solve(prior, good)
        for channels in ([], np.empty((0, 2, 2)), [np.zeros((2, 2))],
                         [np.ones((3, 2))], [np.ones((2, 0))], good[0],
                         [[[3, 7], [3, 8]]], [[[2**32, 0], [0, 2**32]]],
                         [[[-1, 2], [0, 1]]], [[[1., np.inf], [0, 1]]],
                         [[[1., np.nan], [0, 1]]], [[[1j, 0], [0, 1]]],
                         [[['1', '0'], ['0', '1']]], [[[True, False], [False, True]]]):
            with self.subTest(channels=repr(channels)), self.assertRaises(ValueError):
                solve([1, 1], channels)
        for atol in (0, -1, float('nan'), float('inf'), True, 'bad'):
            with self.subTest(atol=atol), self.assertRaises(ValueError):
                solve([1, 1], good, atol=atol)

    def test_bounded_batches_and_python_fallback(self):
        rng = np.random.default_rng(547)
        channels = rng.multinomial(257, np.full(7, 1 / 7), size=(9, 2)).astype(np.uint32)
        expected = solve([2, 3], channels, return_channel=True)
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
                              side_effect=AssertionError('unnecessary joint array')), \
                 patch.object(_union_target_scan, '_compiled_lines', None), \
                 patch.object(_union_target_scan, '_compiled_envelope', None):
                result = solve([2, 3], data, return_channel=True)
            self.assertAlmostEqual(result.union_nats, expected.union_nats, places=13)
            self.assertEqual(sum(seen), channels.size)
            self.assert_calls(np.array([.4, .6]), channels, result)

    @unittest.skipUnless(np.finfo(np.longdouble).eps < np.finfo(float).eps,
                         'longdouble is not wider than float64 on this platform')
    def test_scan_preserves_extended_precision(self):
        theta = np.array([.5, .5], dtype=np.longdouble)
        theta[1] += 4 * np.finfo(np.longdouble).eps
        values = np.stack((1 - theta, theta))[None, :, :] / 2
        with patch.object(_union_target_scan, '_compiled_lines',
                          side_effect=AssertionError('downcast')), \
             patch.object(_union_target_scan, '_compiled_envelope',
                          side_effect=AssertionError('downcast')):
            a, b = _union_target_scan.supporting_lines(
                values, theta, np.arange(2, dtype=np.int64),
                np.ones(1, dtype=np.uint64), np.ones(2, dtype=np.longdouble))
            self.assertEqual(a.dtype, np.dtype(np.longdouble))
            self.assertEqual(b[0], theta[1] / 2)
            slopes = np.array([-1, -.5, 0], dtype=np.longdouble)
            intercepts = np.array([theta.mean(), b[0], 0], dtype=np.longdouble)
            indices, starts = _union_target_scan.upper_envelope(slopes, intercepts)
            self.assertEqual(starts.dtype, np.dtype(np.longdouble))
            self.assertEqual(len(indices), 3)


if __name__ == '__main__':
    unittest.main()
