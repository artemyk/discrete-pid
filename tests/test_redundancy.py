"""Analytical examples, witness checks, and an independent finite LP oracle."""

import subprocess
import sys
import unittest
from functools import partial
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

from discrete_pid import redundancy_binary_sources, redundancy_binary_target
from discrete_pid import _hull


def sources_from_joints(joints, *, tolerance):
    """Convert existing analytical probability fixtures to conditional input."""
    prior = joints[0].sum(axis=1)
    channels = [np.divide(j, prior[:, None], out=np.zeros_like(j),
                          where=prior[:, None] > 0) for j in joints]
    return redundancy_binary_sources(prior, channels, tolerance=tolerance)


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
                self.assertTrue(np.all(kernel >= 0))
                np.testing.assert_allclose(kernel.sum(axis=1), 1, rtol=0, atol=1e-10)
                np.testing.assert_allclose(joint @ kernel, result.target_auxiliary_joint,
                                           rtol=0, atol=1e-9)

    def test_bsc_and_erasure_have_three_output_meet(self):
        joints = [np.array([[0.45, 0.05], [0.05, 0.45]]),
                  np.array([[0.25, 0, 0.25], [0, 0.25, 0.25]])]
        result = redundancy_binary_target(joints, return_garblings=True)
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
            result = redundancy_binary_target(joints, return_garblings=True)
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
            hull = redundancy_binary_target(joints)
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
        result = redundancy_binary_target([binary])
        self.assertAlmostEqual(result.redundancy_nats, mutual_information(binary), places=12)

    def test_zero_states_and_constant_variables(self):
        joint = np.array([[0.2, 0, 0], [0, 0.8, 0]])
        result = redundancy_binary_target([joint, joint.copy()])
        self.assertAlmostEqual(result.redundancy_nats, binary_entropy(0.2), places=12)
        self.assert_witness([joint, joint], result)
        joints = [np.array([[0.2, 0.8], [0, 0]]), np.array([[1.0], [0]])]
        result = redundancy_binary_target(joints)
        self.assertEqual(result.redundancy_nats, 0)
        self.assert_witness(joints, result)

    def test_uninformative_source_gives_zero(self):
        joint = np.array([[0.4, 0.1], [0.1, 0.4]])
        independent = np.full((2, 2), 0.25)
        for solver in (redundancy_binary_target,
                       partial(sources_from_joints, tolerance=1e-12)):
            result = solver([joint, independent])
            self.assertAlmostEqual(result.redundancy_nats, 0, places=12)
            self.assert_witness([joint, independent], result)

    def test_nearly_deterministic_source(self):
        joint = np.array([[1 - 1e-16, 1e-16], [1e-16, 1 - 1e-16]]) / 2
        for solver in (redundancy_binary_target,
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

    def test_validation_and_no_input_mutation(self):
        joint = np.array([[0.3, 0.2], [0.1, 0.4]])
        before = joint.copy()
        redundancy_binary_target([joint, joint.copy()])
        np.testing.assert_array_equal(joint, before)
        for bad in ([], [np.zeros((2, 2))], [np.array([[np.nan], [0]])],
                    [np.array([[-0.1, 0.6], [0.1, 0.4]])],
                    [np.array([[0.6], [0.4]]), np.array([[0.5], [0.5]])],
                    [np.ones((2, 2))], [np.ones((3, 2)) / 6]):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    redundancy_binary_target(bad)
        with self.assertRaises(ValueError):
            redundancy_binary_target([joint], atol=-1)

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
            batched = redundancy_binary_target(joints)
            general = redundancy_binary_target(ragged)
            self.assertAlmostEqual(batched.redundancy_nats, general.redundancy_nats, places=12)
            self.assert_witness(joints, batched)
            for joint, before in zip(joints, original):
                np.testing.assert_array_equal(joint, before)

    def test_batch_marginal_reconciliation_and_zero_support(self):
        a = np.array([[0.2, 0.3], [0.1, 0.4]])
        b = a + np.array([[1e-14, 0], [-1e-14, 0]])
        batched = redundancy_binary_target([a, b])
        general = redundancy_binary_target([a, np.pad(b, ((0, 0), (0, 1)))])
        self.assertAlmostEqual(batched.redundancy_nats, general.redundancy_nats, places=12)
        self.assertEqual(batched.input_adjustment, general.input_adjustment)
        with self.assertRaisesRegex(ValueError, "zero-probability target"):
            redundancy_binary_target([np.array([[1.0, 0], [0, 0]]),
                                      np.array([[1 - 1e-14, 0], [1e-14, 0]])])


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


if __name__ == "__main__":
    unittest.main()
