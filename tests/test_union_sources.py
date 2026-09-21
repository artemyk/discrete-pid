"""Independent full-coupling checks and returned upper-channel witnesses."""

import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from discrete_pid import union_binary_sources
from discrete_pid._union_pricing import (cube_patterns, global_pricing,
                                         pattern_scores)

try:
    import cvxpy as cp
except ImportError:
    cp = None


def channels(a):
    a = np.asarray(a, dtype=float)
    return np.stack((1-a, a), axis=2)


def information(p, q):
    w = q.sum(axis=0)
    selected = q > 0
    return float(np.sum(q[selected]*np.log((q/(p[:, None]*w+1e-300))[selected])))


def full_coupling(p, probabilities):
    """Independent CVXPY model containing every source tuple."""
    p = np.asarray(p, dtype=float)
    p /= p.sum()
    a = np.asarray(probabilities, dtype=float)
    patterns = np.asarray(list(itertools.product((0., 1.), repeat=len(a))))
    joint = cp.Variable((len(p), len(patterns)), nonneg=True)
    marginal = cp.sum(joint, axis=0, keepdims=True)
    objective = cp.sum(cp.rel_entr(joint, p[:, None] @ marginal))
    problem = cp.Problem(cp.Minimize(objective),
                         [cp.sum(joint, axis=1) == p, joint @ patterns == p[:, None]*a.T])
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10,
                  tol_gap_rel=1e-10, max_iter=500)
    if problem.status not in ('optimal', 'optimal_inaccurate'):
        raise AssertionError(problem.status)
    q = np.maximum(joint.value, 0)
    q *= (p/q.sum(axis=1))[:, None]
    return information(p, q)


class UnionSourcesTests(unittest.TestCase):
    def assert_witness(self, p, data, result):
        p = np.asarray(p, dtype=float)
        p /= p.sum()
        values = np.asarray(data, dtype=float)
        sums = values.sum(axis=2, keepdims=True)
        normalized = np.divide(values, sums, out=np.zeros_like(values), where=sums > 0)
        self.assertTrue(np.isfinite(result.union_nats))
        self.assertGreaterEqual(result.union_nats, 0)
        np.testing.assert_allclose(result.channel.sum(axis=1), 1, atol=1e-10)
        np.testing.assert_allclose(result.posteriors.sum(axis=0), 1, atol=1e-10)
        np.testing.assert_allclose(result.target_auxiliary_joint.sum(axis=1), p, atol=1e-10)
        self.assertEqual(len(result.garblings), len(data))
        for source, decoder in zip(normalized, result.garblings):
            if hasattr(decoder, 'toarray'):
                decoder = decoder.toarray()
            self.assertEqual(decoder.shape, (result.channel.shape[1], 2))
            self.assertTrue(np.all(decoder >= 0))
            np.testing.assert_allclose(decoder.sum(axis=1), 1, atol=1e-12)
            np.testing.assert_allclose(result.target_auxiliary_joint @ decoder,
                                       p[:, None]*source, atol=1e-8)
        self.assertLessEqual(result.gap_nats, 1.001e-7)
        self.assertLessEqual(result.max_garbling_residual, 1e-8)

    @unittest.skipIf(cp is None, 'CVXPY is an independent test-only oracle')
    def test_random_full_coupling_values(self):
        rng = np.random.default_rng(71034)
        for d, k in ((2, 4), (3, 2), (3, 3), (3, 5), (4, 4), (8, 3)):
            for _ in range(2):
                p = rng.dirichlet(np.ones(d))
                a = rng.uniform(.03, .97, (k, d))
                result = union_binary_sources(p, channels(a), return_channel=True,
                                               return_garblings=True)
                reference = full_coupling(p, a)
                self.assertLessEqual(result.lower_bound_nats-1e-7, reference)
                self.assertLessEqual(reference, result.upper_bound_nats+1e-7)
                self.assert_witness(p, channels(a), result)

    @unittest.skipIf(cp is None, 'CVXPY is an independent test-only oracle')
    def test_boundary_face_and_scalar_endpoints(self):
        cases = [np.array([[0, .4, .9], [1, .6, .2], [.2, .7, 0], [.3, 1, .2]]),
                 np.array([[0, 1, 0], [0, 0, 1], [.2, .8, .4]]),
                 np.array([[.1, .8, 1], [.9, .2, 0]]),
                 np.array([[0, 1, .5], [.5, 0, 1]])]
        p = np.array([.2, .3, .5])
        for a in cases:
            result = union_binary_sources(p, channels(a), reduce_sources=False,
                                           return_channel=True, return_garblings=True)
            reference = full_coupling(p, a)
            self.assertLess(abs(result.union_nats-reference), 2e-7)
            self.assert_witness(p, channels(a), result)

    def test_source_reduction_recovers_every_original_decoder(self):
        rng = np.random.default_rng(178)
        a = rng.uniform(.05, .95, (80, 3))
        a = np.vstack((a, 1-a[:10], a[:10], .25*a[:10]+.75*a[10:20],
                       np.tile([.3, .3, .3], (2, 1))))
        p = [2., 3., 5.]
        data = channels(a)
        result = union_binary_sources(p, data, return_channel=True, return_garblings=True)
        self.assertLess(result.diagnostics['retained_sources'], len(a))
        self.assertLessEqual(result.diagnostics['source_reduction_residual'], 1e-12)
        self.assert_witness(p, data, result)

    @unittest.skipIf(cp is None, 'CVXPY is an independent test-only oracle')
    def test_compressed_exchange_matches_full_coupling(self):
        rng = np.random.default_rng(27)
        p = rng.dirichlet(np.ones(3))
        a = rng.uniform(.03, .97, (3, 6)).T
        result = union_binary_sources(p, channels(a), reduce_sources=False,
                                       return_channel=True, return_garblings=True)
        self.assertGreater(result.diagnostics['compressions'], 0)
        reference = full_coupling(p, a)
        self.assertLessEqual(result.lower_bound_nats-1e-7, reference)
        self.assertLessEqual(reference, result.upper_bound_nats+1e-7)
        self.assert_witness(p, channels(a), result)

    def test_compressed_master_failure_restores_generated_patterns(self):
        from discrete_pid import _union_sources_opt as module
        rng = np.random.default_rng(27)
        p = rng.dirichlet(np.ones(3))
        a = rng.uniform(.03, .97, (3, 6)).T
        compress = module.compress_support
        master = module.restricted_master
        state = dict(compressed=False, damaged=False)

        def track_compression(*args, **kwargs):
            result = compress(*args, **kwargs)
            state['compressed'] |= result[2]
            return result

        def fail_once_after_compression(*args, **kwargs):
            if state['compressed'] and not state['damaged']:
                state['damaged'] = True
                raise ArithmeticError('test: compressed support is numerically infeasible')
            return master(*args, **kwargs)

        with patch.object(module, 'compress_support', side_effect=track_compression), \
                patch.object(module, 'restricted_master', side_effect=fail_once_after_compression):
            result = union_binary_sources(p, channels(a), reduce_sources=False,
                                           return_channel=True, return_garblings=True)
        self.assertTrue(state['damaged'])
        self.assertGreater(result.diagnostics['compression_fallbacks'], 0)
        self.assert_witness(p, channels(a), result)

    def test_zero_prior_rows_target_dispatch_and_output_flags(self):
        data = np.array([[[0, 0], [1, 4], [4, 1]],
                         [[0, 0], [2, 3], [5, 0]]], dtype=np.uint32)
        before = data.copy()
        p = [0, 2, 3]
        both = union_binary_sources(p, data, return_channel=True, return_garblings=True)
        self.assert_witness(p, data, both)
        np.testing.assert_allclose(both.channel[0], both.posterior_weights)
        np.testing.assert_array_equal(data, before)
        for channel_flag, decoder_flag in itertools.product((False, True), repeat=2):
            result = union_binary_sources(p, (row for row in data),
                                           return_channel=channel_flag, return_garblings=decoder_flag)
            self.assertEqual(result.channel is not None, channel_flag)
            self.assertEqual(result.posteriors is not None, channel_flag)
            self.assertEqual(result.posterior_weights is not None, channel_flag)
            self.assertEqual(result.garblings is not None, decoder_flag)
            self.assertAlmostEqual(result.union_nats, both.union_nats, places=12)

    def test_constant_sources_single_source_and_single_active_target(self):
        for p, a in [([1, 2, 3], [[.3, .3, .3], [.8, .8, .8]]),
                     ([0, 4, 0], [[.3, .7, .2], [.5, .3, .8]]),
                     ([1, 2, 3], [[.2, .8, .5]])]:
            result = union_binary_sources(p, channels(a), return_channel=True, return_garblings=True)
            self.assert_witness(p, channels(a), result)
            if len(a) > 1:
                self.assertAlmostEqual(result.union_nats, 0, places=13)

    def test_scalar_range_failure_uses_four_pattern_fallback(self):
        p = [1., 2., 3.]
        data = channels([[.2, .8, .4], [.7, .3, .6]])
        with patch('discrete_pid._union_two_sources.union_two_binary',
                   side_effect=FloatingPointError('log-odds range')):
            result = union_binary_sources(p, data, return_channel=True, return_garblings=True)
        self.assertEqual(result.method, 'two-source relative-entropy fallback')
        self.assertEqual(result.diagnostics['solver_threads'], 1)
        self.assert_witness(p, data, result)

    def test_source_reduction_failure_keeps_constraints(self):
        from scipy.spatial import QhullError
        a = np.random.default_rng(891).uniform(.1, .9, (4, 3))
        with patch('discrete_pid._union_sources_opt.Delaunay', side_effect=QhullError('test')):
            result = union_binary_sources([1, 1, 1], channels(a),
                                           return_channel=True, return_garblings=True)
        self.assert_witness([1, 1, 1], channels(a), result)

    def test_many_sources_do_not_allocate_a_dense_identity(self):
        from discrete_pid._union_sources_opt import reduce_sources
        original_eye = np.eye
        def small_eye(n, *args, **kwargs):
            if n > 100:
                raise AssertionError('quadratic identity allocation')
            return original_eye(n, *args, **kwargs)
        a = np.random.default_rng(9391).uniform(.05, .95, (3, 2000))
        with patch('numpy.eye', side_effect=small_eye):
            retained, matrix, offset, residual = reduce_sources(a)
        self.assertLess(retained.shape[1], 100)
        self.assertLessEqual(residual, 1e-12)
        np.testing.assert_allclose(retained @ matrix + offset, a, atol=1e-12)

    def test_invalid_inputs_and_work_limit(self):
        valid = channels([[.2, .6, .8], [.8, .4, .2], [.3, .5, .7]])
        for keyword, value in [('tolerance', 0), ('tolerance', 1e-12),
                               ('tolerance', np.nan), ('batch_size', 0),
                               ('max_iterations', True), ('max_pricing_nodes', 0)]:
            with self.assertRaises(ValueError):
                union_binary_sources([1, 1, 1], valid, **{keyword: value})
        for bad in [np.zeros((0, 3, 2)), np.zeros((2, 3, 3)), valid.astype(complex),
                    np.full((3, 3, 2), -1.), np.full((3, 3, 2), np.nan)]:
            with self.assertRaises(ValueError):
                union_binary_sources([1, 1, 1], bad)
        with self.assertRaisesRegex(ValueError, 'same positive row sum'):
            union_binary_sources([1, 1, 1], np.array([[[1, 2], [2, 3], [1, 2]]]))
        with self.assertRaisesRegex(ValueError, 'same positive row sum'):
            union_binary_sources([1, 1, 1], [np.array([[1, 2], [2, 3], [1, 2]]), valid[0]])
        B = np.random.default_rng(927).normal(size=(4, 13))
        with self.assertRaisesRegex(RuntimeError, 'no convergence gap'):
            global_pricing(B, np.zeros(4), None, max_nodes=1)


class PricingTests(unittest.TestCase):
    def test_global_oracles_against_exhaustive_binary_patterns(self):
        rng = np.random.default_rng(8347)
        for d, k, boundary in ((3, 13, False), (3, 13, True), (4, 13, False), (4, 13, True)):
            B = rng.normal(size=(d, k))
            c = rng.normal(size=d)
            a = rng.uniform(.1, .9, (d, k))
            if boundary:
                a[0, :2] = [0, 1]
                a[1, :2] = [1, 0]
            expected = pattern_scores(B, c, cube_patterns(k), a if boundary else None).max()
            _, scores, upper = global_pricing(B, c, a, tolerance=1e-10)
            self.assertLessEqual(expected, upper+1e-12)
            self.assertLessEqual(upper-expected, 2e-9)
            self.assertLessEqual(abs(scores.max()-expected), 2e-9)


class SupportCompressionTests(unittest.TestCase):
    def fixture(self, *, boundary=False):
        p = np.array([.2, .3, .5])
        patterns = cube_patterns(5)
        q = np.random.default_rng(394).uniform(.01, 1., (len(p), len(patterns)))
        if boundary:
            q[0, patterns[:, 0] == 1] = 0
            q[1, patterns[:, 0] == 0] = 0
            q[2, patterns[:, 1] == 0] = 0
        q[:, 0] = 0  # A zero-mass column must not require a posterior.
        q *= (p/q.sum(axis=1))[:, None]
        a = (q @ patterns)/p[:, None]
        return p, a, q, patterns

    def test_compression_preserves_moments_and_does_not_increase_information(self):
        from discrete_pid._union_sources_opt import compress_support
        for boundary in (False, True):
            with self.subTest(boundary=boundary):
                p, a, q, patterns = self.fixture(boundary=boundary)
                before_q, before_patterns = q.copy(), patterns.copy()
                compressed, kept, accepted = compress_support(p, a, q, patterns, 1e-7)
                self.assertTrue(accepted)
                self.assertLess(len(kept), len(patterns))
                self.assertLessEqual(len(kept), len(p)*(a.shape[1]+1))
                self.assertTrue(np.all(np.isfinite(compressed)))
                self.assertTrue(np.all(compressed >= 0))
                self.assertTrue(np.all(compressed.sum(axis=0) > 0))
                np.testing.assert_allclose(compressed.sum(axis=1), p, rtol=0, atol=1e-8)
                np.testing.assert_allclose(compressed @ kept, p[:, None]*a, rtol=0, atol=1e-8)
                self.assertLessEqual(information(p, compressed), information(p, q)+1e-10)
                np.testing.assert_array_equal(q, before_q)
                np.testing.assert_array_equal(patterns, before_patterns)

    def test_failed_nonfinite_or_infeasible_lp_keeps_original_support(self):
        from discrete_pid import _union_sources_opt as module
        p, a, q, patterns = self.fixture()
        positive = np.count_nonzero(q.sum(axis=0) > 0)
        answers = [SimpleNamespace(success=False)]
        answers.extend(SimpleNamespace(success=True, x=np.full(positive, value))
                       for value in (np.nan, np.inf, 0.))
        for answer in answers:
            with self.subTest(answer=answer), patch.object(module, 'linprog', return_value=answer):
                returned_q, returned_patterns, accepted = module.compress_support(
                    p, a, q, patterns, 1e-7)
                self.assertFalse(accepted)
                np.testing.assert_array_equal(returned_q, q)
                np.testing.assert_array_equal(returned_patterns, patterns)


class MasterPrecisionTests(unittest.TestCase):
    def setUp(self):
        from discrete_pid import _union_sources_opt
        self.module = _union_sources_opt
        self.prior = np.array([.2, .3, .5])
        self.a = np.array([[.1, .7, .4], [.8, .2, .6], [.5, .9, .3]])
        self.patterns = cube_patterns(3)

    def inaccurate_solver(self, settings_seen, *, always=False, damage_dual=False):
        original = self.module.clarabel.DefaultSolver
        def factory(*args):
            settings = args[-1]
            settings_seen.append(settings)
            solver = original(*args)
            damage = always or len(settings_seen) == 1
            def solve():
                result = solver.solve()
                if not damage:
                    return result
                if damage_dual:
                    damaged = np.asarray(result.z).copy()
                    damaged[:len(self.prior)] += 1e-4
                    return SimpleNamespace(status='AlmostSolved', x=result.x, z=damaged)
                # An AlmostSolved result can have an accurate epigraph/dual
                # objective but unacceptable probability-table residuals.
                damaged = np.asarray(result.x).copy()
                damaged[0] += 1e-4
                return SimpleNamespace(status='AlmostSolved', x=damaged, z=result.z)
            return SimpleNamespace(solve=solve)
        return factory

    def test_inaccurate_master_retries_with_tighter_different_scaling(self):
        settings = []
        with patch.object(self.module.clarabel, 'DefaultSolver',
                          side_effect=self.inaccurate_solver(settings)):
            value, alpha, B, dual, q, residual, retries = self.module.restricted_master(
                self.prior, self.a, self.patterns, 1e-7)
        self.assertEqual(retries, 1)
        self.assertEqual(len(settings), 2)
        self.assertLess(settings[1].tol_feas, settings[0].tol_feas)
        self.assertFalse(settings[1].equilibrate_enable)
        self.assertLessEqual(settings[0].reduced_tol_feas, 1e-9)
        self.assertLess(residual, 1e-8)
        np.testing.assert_allclose(q @ self.patterns, self.prior[:, None]*self.a, atol=1e-8)
        price = pattern_scores(B, np.log(self.prior)+alpha, self.patterns, self.a).max()
        self.assertLessEqual(value-(dual-max(0., price)), 5e-8)

    def test_reconstructed_gap_rejects_inaccurate_dual(self):
        settings = []
        with patch.object(self.module.clarabel, 'DefaultSolver',
                          side_effect=self.inaccurate_solver(settings, damage_dual=True)):
            result = self.module.restricted_master(
                self.prior, self.a, self.patterns, 1e-7)
        self.assertEqual(result[-1], 1)
        self.assertEqual(len(settings), 2)

    def test_persistent_inaccuracy_raises_after_bounded_retries(self):
        settings = []
        with patch.object(self.module.clarabel, 'DefaultSolver',
                          side_effect=self.inaccurate_solver(settings, always=True)):
            with self.assertRaisesRegex(ArithmeticError, 'requested precision'):
                self.module.restricted_master(self.prior, self.a, self.patterns, 1e-7)
        self.assertEqual(len(settings), 3)
        self.assertTrue(all(setting.reduced_tol_feas <= 1e-9 for setting in settings))


if __name__ == '__main__':
    unittest.main()
