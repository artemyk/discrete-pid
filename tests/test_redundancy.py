"""Analytical examples, witness checks, and an independent finite LP oracle."""

import subprocess
import sys
import unittest
from functools import partial
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from discrete_pid import redundancy_binary_sources, redundancy_binary_target
from discrete_pid import _hull, _martingale, binary_target


def target_from_joints(joints, **kwargs):
    """Keep analytical joint-law fixtures independent of the channel API."""
    prior = joints[0].sum(axis=1)
    channels = [np.divide(j, prior[:, None], out=np.zeros_like(j),
                          where=prior[:, None] > 0) for j in joints]
    return redundancy_binary_target(prior, channels, **kwargs)


def sources_from_joints(joints, *, tolerance):
    """Convert existing analytical probability fixtures to conditional input."""
    prior = joints[0].sum(axis=1)
    channels = [np.divide(j, prior[:, None], out=np.zeros_like(j),
                          where=prior[:, None] > 0) for j in joints]
    return redundancy_binary_sources(prior, channels, tolerance=tolerance,
                                     return_channel=True, return_garblings=True)


def binary_entropy(p):
    if p in (0, 1):
        return 0.0
    return -p * np.log(p) - (1 - p) * np.log1p(-p)


def mutual_information(joint):
    y, x = joint.sum(axis=1), joint.sum(axis=0)
    product = y[:, None] * x
    active = joint > 0
    return float(np.sum(joint[active] * np.log(joint[active] / product[active])))


def line_source(prior, direction, lower, upper):
    weights = np.array([upper, -lower]) / (upper - lower)
    return (prior[:, None] + direction[:, None] * [lower, upper]) * weights


def finite_call_lp(joints):
    """Independent binary-target oracle: maximize KL on all source knots.

    Its constraints are the full set of call inequalities, plus mass and
    mean. It uses no hull construction or slope-jump reconstruction.
    """
    prior = joints[0].sum(axis=1)[1]
    laws = [(j.sum(axis=0), j[1] / j.sum(axis=0)) for j in joints]
    support = np.unique(np.concatenate(([0.0, 1.0], *(r for _, r in laws))))
    rows, rhs = [], []
    for mass, r in laws:
        for t in support:
            rows.append(np.maximum(support - t, 0))
            rhs.append(mass @ np.maximum(r - t, 0))
    kl = np.array([-binary_entropy(t) - t * np.log(prior)
                   - (1 - t) * np.log1p(-prior) for t in support])
    solved = linprog(-kl, A_ub=rows, b_ub=rhs,
                     A_eq=[np.ones_like(support), support], b_eq=[1, prior],
                     bounds=(0, None), method="highs")
    if not solved.success:
        raise AssertionError(solved.message)
    return -solved.fun


class RedundancyTests(unittest.TestCase):
    def assert_witness(self, joints, result):
        self.assertGreaterEqual(result.redundancy_nats, 0)
        self.assertTrue(np.all(result.posteriors >= 0))
        self.assertTrue(np.all(result.posterior_weights > 0))
        np.testing.assert_allclose(result.posteriors.sum(axis=0), 1, atol=1e-12)
        np.testing.assert_allclose(result.target_auxiliary_joint.sum(axis=1),
                                   joints[0].sum(axis=1), rtol=0, atol=1e-11)
        self.assertAlmostEqual(result.redundancy_nats,
                               mutual_information(result.target_auxiliary_joint), places=11)
        if result.garblings is not None:
            for joint, kernel in zip(joints, result.garblings):
                entries = kernel.data if sparse.issparse(kernel) else kernel
                self.assertTrue(np.all(entries >= 0))
                np.testing.assert_allclose(np.asarray(kernel.sum(axis=1)).ravel(), 1,
                                           rtol=0, atol=1e-10)
                np.testing.assert_allclose(joint @ kernel, result.target_auxiliary_joint,
                                           rtol=0, atol=1e-9)

    def test_bsc_and_erasure_have_three_output_meet(self):
        joints = [np.array([[0.45, 0.05], [0.05, 0.45]]),
                  np.array([[0.25, 0, 0.25], [0, 0.25, 0.25]])]
        result = target_from_joints(joints, return_garblings=True, return_channel=True)
        self.assertAlmostEqual(result.redundancy_nats,
                               5 / 8 * (np.log(2) - binary_entropy(0.1)), places=12)
        np.testing.assert_allclose(result.posteriors[1], [0.1, 0.5, 0.9], atol=1e-12)
        np.testing.assert_allclose(result.posterior_weights, [5 / 16, 3 / 8, 5 / 16])
        self.assert_witness(joints, result)

    def test_binary_target_against_independent_lp(self):
        rng = np.random.default_rng(902)
        for _ in range(12):
            prior = rng.dirichlet([2, 2])
            joints = [prior[:, None] * rng.dirichlet(np.ones(m), size=2)
                      for m in [3, 4, 5]]
            result = target_from_joints(joints, return_garblings=True, return_channel=True)
            self.assertAlmostEqual(result.redundancy_nats, finite_call_lp(joints), places=9)
            self.assert_witness(joints, result)

    def test_collinear_incomparable_binary_sources(self):
        prior = np.ones(3) / 3
        direction = np.array([0.1, -0.1, 0])
        joints = [line_source(prior, direction, -2, 1),
                  line_source(prior, direction, -1, 2)]
        expected = line_source(prior, direction, -1, 1)
        result = sources_from_joints(joints, tolerance=1e-12)
        self.assertAlmostEqual(result.redundancy_nats, mutual_information(expected), places=12)
        # Auxiliary labels have no prescribed ordering.
        actual = result.target_auxiliary_joint
        np.testing.assert_allclose(actual[:, np.argsort(actual[0])],
                                   expected[:, np.argsort(expected[0])], atol=1e-12)
        self.assert_witness(joints, result)

    def test_different_posterior_lines_give_zero(self):
        prior = np.ones(3) / 3
        joints = [line_source(prior, np.array([0.1, -0.1, 0]), -1, 1),
                  line_source(prior, np.array([0, 0.1, -0.1]), -1, 1)]
        result = sources_from_joints(joints, tolerance=1e-12)
        self.assertEqual(result.redundancy_nats, 0)
        self.assertEqual(len(result.posterior_weights), 1)
        self.assert_witness(joints, result)

    def test_algorithms_agree_when_target_and_sources_are_binary(self):
        rng = np.random.default_rng(631)
        for _ in range(20):
            prior = rng.dirichlet([2, 2])
            joints = [prior[:, None] * rng.dirichlet([1, 1], size=2) for _ in range(4)]
            hull = target_from_joints(joints, return_channel=True)
            segment = sources_from_joints(joints, tolerance=1e-12)
            self.assertAlmostEqual(hull.redundancy_nats, segment.redundancy_nats, places=11)
            self.assert_witness(joints, segment)

    def test_relabeling_and_reordering_preserve_information(self):
        prior = np.array([0.2, 0.3, 0.5])
        direction = np.array([0.04, 0.06, -0.10])
        joints = [line_source(prior, direction, -2, 1),
                  line_source(prior, direction, -1, 2)]
        expected = sources_from_joints(joints, tolerance=1e-12).redundancy_nats
        permuted = [j[[2, 0, 1]][:, ::-1] for j in joints[::-1]]
        result = sources_from_joints(permuted, tolerance=1e-12)
        self.assertAlmostEqual(result.redundancy_nats, expected, places=12)
        self.assert_witness(permuted, result)

    def test_single_source_and_duplicate_sources(self):
        joint = np.array([[0.1, 0.2], [0.2, 0.05], [0.15, 0.3]])
        for joints in ([joint], [joint, joint.copy()]):
            result = sources_from_joints(joints, tolerance=1e-12)
            self.assertAlmostEqual(result.redundancy_nats, mutual_information(joint), places=12)
            self.assert_witness(joints, result)
        binary = joint[:2] / joint[:2].sum()
        result = target_from_joints([binary], return_channel=True)
        self.assertAlmostEqual(result.redundancy_nats, mutual_information(binary), places=12)

    def test_zero_states_and_constant_variables(self):
        joint = np.array([[0.2, 0, 0], [0, 0.8, 0]])
        result = target_from_joints([joint, joint.copy()], return_channel=True)
        self.assertAlmostEqual(result.redundancy_nats, binary_entropy(0.2), places=12)
        self.assert_witness([joint, joint], result)
        joints = [np.array([[0.2, 0.8], [0, 0]]), np.array([[1.0], [0]])]
        result = target_from_joints(joints, return_channel=True)
        self.assertEqual(result.redundancy_nats, 0)
        self.assert_witness(joints, result)

    def test_uninformative_source_gives_zero(self):
        joint = np.array([[0.4, 0.1], [0.1, 0.4]])
        independent = np.full((2, 2), 0.25)
        for solver in (partial(target_from_joints, return_channel=True),
                       partial(sources_from_joints, tolerance=1e-12)):
            result = solver([joint, independent])
            self.assertAlmostEqual(result.redundancy_nats, 0, places=12)
            self.assert_witness([joint, independent], result)

    def test_nearly_deterministic_source(self):
        joint = np.array([[1 - 1e-16, 1e-16], [1e-16, 1 - 1e-16]]) / 2
        for solver in (partial(target_from_joints, return_channel=True),
                       partial(sources_from_joints, tolerance=1e-12)):
            result = solver([joint])
            self.assertAlmostEqual(result.redundancy_bits, 1, places=12)
            self.assert_witness([joint], result)

    def test_near_collinearity_tolerance_is_explicit(self):
        prior = np.ones(3) / 3
        v = np.array([0.1, -0.1, 0])
        perturbed = v + np.array([0, 1e-8, -1e-8])
        joints = [line_source(prior, v, -1, 1), line_source(prior, perturbed, -1, 1)]
        strict = sources_from_joints(joints, tolerance=1e-12)
        self.assertEqual(strict.redundancy_nats, 0)
        approximate = sources_from_joints(joints, tolerance=1e-5)
        self.assertGreater(approximate.redundancy_nats, 0)
        self.assertGreater(approximate.max_garbling_residual, 0)

    @unittest.skipIf(_hull._compiled_scan is None, "Numba is optional")
    def test_hull_is_compiled_at_import(self):
        # Check in a fresh process, before any solver has run.
        code = (
            "from discrete_pid import _hull\n"
            "assert _hull._compiled_scan.signatures\n"
        )
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


    def test_batched_and_ragged_target_paths_agree(self):
        rng = np.random.default_rng(420)
        for size in (2, 7, 32):
            joints = [np.array([0.3, 0.7])[:, None] *
                      rng.dirichlet(np.ones(size), size=2) for _ in range(5)]
            original = [joint.copy() for joint in joints]
            # Unequal zero padding forces the general validation/knot path.
            ragged = [np.pad(joint, ((0, 0), (0, i))) for i, joint in enumerate(joints)]
            batched = target_from_joints(joints, return_channel=True)
            general = target_from_joints(ragged, return_channel=True)
            self.assertAlmostEqual(batched.redundancy_nats, general.redundancy_nats, places=12)
            self.assert_witness(joints, batched)
            for joint, before in zip(joints, original):
                np.testing.assert_array_equal(joint, before)

    def test_hull_fallback_preserves_extended_precision(self):
        x = np.array([0, 0.25, 0.5, 0.75, 1], dtype=np.longdouble)
        y = np.array([0.5, 0.3, 0.12, 0.04, 0], dtype=np.longdouble)
        expected = _hull._scan(x, y)
        with patch.object(_hull, "_compiled_scan", None):
            np.testing.assert_array_equal(_hull.lower_hull(x, y), expected)
        if x.dtype != np.dtype(np.float64):
            with patch.object(_hull, "_compiled_scan", side_effect=AssertionError("downcast")):
                np.testing.assert_array_equal(_hull.lower_hull(x, y), expected)

    @unittest.skipIf(_hull._compiled_scan is None, "Numba is optional")
    def test_compiled_hull_matches_python(self):
        rng = np.random.default_rng(37)
        for count in (2, 13, 200):
            x = np.sort(rng.random(count))
            y = rng.random(count)
            np.testing.assert_array_equal(_hull._compiled_scan(x, y), _hull._scan(x, y))
        # Include exactly collinear knots and a very small positive turn.
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        for y in (1 - x, (1 - x) + np.array([0, 0, -1e-15, 0, 0])):
            np.testing.assert_array_equal(_hull._compiled_scan(x, y), _hull._scan(x, y))


class SparseGarblingTests(unittest.TestCase):
    def assert_kernel(self, joint, target_joint, kernel):
        self.assertIsInstance(kernel, sparse.csr_matrix)
        self.assertEqual(kernel.shape, (joint.shape[1], target_joint.shape[1]))
        self.assertTrue(np.all(np.isfinite(kernel.data)))
        self.assertTrue(np.all(kernel.data >= 0))
        np.testing.assert_allclose(np.asarray(kernel.sum(axis=1)).ravel(), 1,
                                   rtol=0, atol=2e-12)
        np.testing.assert_allclose(joint @ kernel, target_joint, rtol=0, atol=2e-12)
        # Each merged quantile cell and each deficit/excess match contributes
        # only a constant number of entries, including zero-mass source rows.
        self.assertLessEqual(kernel.nnz, 4 * sum(kernel.shape))

    def reconstruct(self, joint, known_kernel):
        target_joint = joint @ known_kernel
        joint_before, target_before = joint.copy(), target_joint.copy()
        actual = binary_target._garbling(joint, target_joint, target_joint.sum(axis=0), 1e-12)
        self.assert_kernel(joint, target_joint, actual)
        np.testing.assert_array_equal(joint, joint_before)
        np.testing.assert_array_equal(target_joint, target_before)
        return actual

    def test_known_feasible_random_channels(self):
        # Generate feasibility independently: every row-stochastic kernel is
        # a valid garbling, without relying on the meet or coupling algorithm.
        rng = np.random.default_rng(129)
        for m, q in ((2, 3), (7, 2), (12, 17), (80, 110)):
            for _ in range(4):
                joint = rng.dirichlet(np.ones(2 * m)).reshape(2, m)
                known = rng.dirichlet(np.ones(q), size=m)
                with self.subTest(m=m, q=q):
                    self.reconstruct(joint, known)

    def test_duplicate_posteriors_permutations_and_zero_columns(self):
        theta = np.array([.8, .2, .8, .2, .5, 0.])
        mass = np.array([.1, .2, .15, .25, .3, 0.])
        joint = np.vstack((1 - theta, theta)) * mass
        # Identity preserves duplicated output posteriors but zero output
        # states are omitted, matching the solver's output contract.
        known = np.eye(6)[:, [4, 2, 0, 3, 1]]
        known[-1, 0] = 1
        actual = self.reconstruct(joint, known)
        self.assertEqual(actual.getrow(5).nnz, 1)
        self.assertEqual(actual[5].sum(), 1)

    def test_tiny_source_masses_and_narrow_posterior_support(self):
        for theta, mass in (
                (np.array([0., .25, .5, 1.]), np.array([1e-14, .2, .3, .5 - 1e-14])),
                (np.array([0., .25, .5, 1.]), np.array([1e-300, .2, .3, .5])),
                (.5 + 1e-8 * np.array([-3., -1., 1., 3.]), np.full(4, .25)),
                (.5 + 1e-14 * np.array([-.5, -1/6, 1/6, .5]), np.full(4, .25))):
            joint = np.vstack((1 - theta, theta)) * mass
            known = np.array([[.8, .2, 0.], [.3, .6, .1],
                              [.1, .6, .3], [0., .2, .8]])
            self.reconstruct(joint, known)

    def test_public_solver_uses_sparse_garblings_without_linear_programming(self):
        prior = np.array([.4, .6])
        channels = [np.array([[.9, .1, 0], [.1, .9, 0]]),
                    np.array([[.5, 0, .5], [0, .5, .5]])]
        with patch('scipy.optimize.linprog', side_effect=AssertionError('LP called')):
            result = redundancy_binary_target(prior, channels, return_channel=True,
                                               return_garblings=True)
        for channel, kernel in zip(channels, result.garblings):
            self.assert_kernel(prior[:, None] * channel, result.target_auxiliary_joint, kernel)
        self.assertLess(result.max_garbling_residual, 2e-12)

    def test_public_solver_reuses_sorted_posterior_order(self):
        prior = np.array([.37, .63])
        channels = [np.array([[.08, .72, 0., .2], [.75, .15, 0., .1]]),
                    np.array([[.65, .25, .1], [.1, .2, .7]])]
        reconstruct = binary_target._garbling
        seen = []

        def checked(joint, *args, **kwargs):
            order = kwargs.get('source_order')
            self.assertIsNotNone(order)
            mass = joint.sum(axis=0)
            active = order[mass[order] > 0]
            np.testing.assert_array_equal(np.sort(active), np.flatnonzero(mass > 0))
            theta = joint[1, active] / mass[active]
            self.assertTrue(np.all(np.diff(theta) >= 0))
            seen.append(order.copy())
            # Passing the order alone is insufficient: reconstruction must
            # use it instead of performing another per-source sort.
            with patch.object(binary_target.np, 'argsort',
                              side_effect=AssertionError('posterior order recomputed')):
                return reconstruct(joint, *args, **kwargs)

        with patch.object(binary_target, '_garbling', side_effect=checked):
            result = redundancy_binary_target(prior, channels, return_channel=True,
                                               return_garblings=True)
        self.assertEqual(len(seen), len(channels))
        self.assertGreater(result.channel.shape[1], 1)
        for channel, kernel in zip(channels, result.garblings):
            self.assert_kernel(prior[:, None] * channel, result.target_auxiliary_joint, kernel)

    def test_constant_meet_and_target_have_sparse_stochastic_kernels(self):
        channels = [np.array([[.9, .1, 0], [.1, .9, 0]]),
                    np.array([[.3, .7], [.3, .7]])]
        for prior in (np.array([.4, .6]), np.array([1., 0.])):
            result = redundancy_binary_target(prior, channels, return_channel=True,
                                               return_garblings=True)
            self.assertEqual(result.channel.shape[1], 1)
            for channel, kernel in zip(channels, result.garblings):
                self.assert_kernel(prior[:, None] * channel, result.target_auxiliary_joint, kernel)
                self.assertEqual(kernel.nnz, channel.shape[1])

    @unittest.skipIf(_martingale._compiled_couple is None, 'Numba is optional')
    def test_martingale_is_compiled_at_import(self):
        run = subprocess.run([sys.executable, '-c',
                              'from discrete_pid import _martingale\n'
                              'assert _martingale._compiled_couple.signatures\n'],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_martingale_fallback_preserves_extended_precision(self):
        theta = np.array([0., .2, .8, 1.], dtype=np.longdouble)
        mass = np.full(4, .25, dtype=np.longdouble)
        support = np.array([.3, .7], dtype=np.longdouble)
        weights = np.full(2, .5, dtype=np.longdouble)
        expected = _martingale._couple(theta, mass, support, weights)
        with patch.object(_martingale, '_compiled_couple', None):
            actual = _martingale.inverse_transform(theta, mass, support, weights)
        for computed, reference in zip(actual, expected):
            np.testing.assert_array_equal(computed, reference)
        if theta.dtype != np.dtype(np.float64):
            with patch.object(_martingale, '_compiled_couple', side_effect=AssertionError('downcast')):
                _martingale.inverse_transform(theta, mass, support, weights)

    @unittest.skipIf(_martingale._compiled_couple is None, 'Numba is optional')
    def test_compiled_martingale_matches_python(self):
        rng = np.random.default_rng(614)
        for m, q in ((2, 1), (5, 7), (40, 19)):
            theta = np.sort(rng.random(m))
            mass = rng.dirichlet(np.ones(m))
            known = rng.dirichlet(np.ones(q), size=m)
            weights = mass @ known
            support = ((mass * theta) @ known) / weights
            order = np.argsort(support)
            support, weights = support[order], weights[order]
            expected = _martingale._couple(theta, mass, support, weights)
            actual = _martingale._compiled_couple(theta, mass, support, weights)
            for computed, reference in zip(actual, expected):
                np.testing.assert_array_equal(computed, reference)


if __name__ == "__main__":
    unittest.main()
