"""Analytical examples, witness checks, and an independent finite LP oracle."""

import unittest

import numpy as np
from scipy.optimize import linprog

from discrete_pid import redundancy_binary_sources, redundancy_binary_target


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
        result = redundancy_binary_sources(joints)
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
        result = redundancy_binary_sources(joints)
        self.assertEqual(result.redundancy_nats, 0)
        self.assertEqual(len(result.posterior_weights), 1)
        self.assert_witness(joints, result)

    def test_algorithms_agree_when_target_and_sources_are_binary(self):
        rng = np.random.default_rng(631)
        for _ in range(20):
            prior = rng.dirichlet([2, 2])
            joints = [prior[:, None] * rng.dirichlet([1, 1], size=2) for _ in range(4)]
            hull = redundancy_binary_target(joints)
            segment = redundancy_binary_sources(joints)
            self.assertAlmostEqual(hull.redundancy_nats, segment.redundancy_nats, places=11)
            self.assert_witness(joints, segment)

    def test_relabeling_and_reordering_preserve_information(self):
        prior = np.array([0.2, 0.3, 0.5])
        direction = np.array([0.04, 0.06, -0.10])
        joints = [line_source(prior, direction, -2, 1),
                  line_source(prior, direction, -1, 2)]
        expected = redundancy_binary_sources(joints).redundancy_nats
        permuted = [j[[2, 0, 1]][:, ::-1] for j in joints[::-1]]
        result = redundancy_binary_sources(permuted)
        self.assertAlmostEqual(result.redundancy_nats, expected, places=12)
        self.assert_witness(permuted, result)

    def test_single_source_and_duplicate_sources(self):
        joint = np.array([[0.1, 0.2], [0.2, 0.05], [0.15, 0.3]])
        for joints in ([joint], [joint, joint.copy()]):
            result = redundancy_binary_sources(joints)
            self.assertAlmostEqual(result.redundancy_nats, mutual_information(joint), places=12)
            self.assert_witness(joints, result)
        binary = joint[:2] / joint[:2].sum()
        result = redundancy_binary_target([binary])
        self.assertAlmostEqual(result.redundancy_nats, mutual_information(binary), places=12)

    def test_zero_states_and_constant_variables(self):
        # A padded binary source with a zero-probability target state.
        joint = np.array([[0.2, 0, 0], [0, 0.8, 0], [0, 0, 0]])
        result = redundancy_binary_sources([joint, joint.copy()])
        self.assertAlmostEqual(result.redundancy_nats, binary_entropy(0.2), places=12)
        self.assert_witness([joint, joint], result)
        for solver in (redundancy_binary_target, redundancy_binary_sources):
            with self.subTest(solver=solver.__name__):
                joints = [np.array([[0.2, 0.8], [0, 0]]), np.array([[1.0], [0]])]
                result = solver(joints)
                self.assertEqual(result.redundancy_nats, 0)
                self.assert_witness(joints, result)

    def test_uninformative_source_gives_zero(self):
        joint = np.array([[0.4, 0.1], [0.1, 0.4]])
        independent = np.full((2, 2), 0.25)
        for solver in (redundancy_binary_target, redundancy_binary_sources):
            result = solver([joint, independent])
            self.assertAlmostEqual(result.redundancy_nats, 0, places=12)
            self.assert_witness([joint, independent], result)

    def test_nearly_deterministic_source(self):
        joint = np.array([[1 - 1e-16, 1e-16], [1e-16, 1 - 1e-16]]) / 2
        for solver in (redundancy_binary_target, redundancy_binary_sources):
            result = solver([joint])
            self.assertAlmostEqual(result.redundancy_bits, 1, places=12)
            self.assert_witness([joint], result)

    def test_near_collinearity_tolerance_is_explicit(self):
        prior = np.ones(3) / 3
        v = np.array([0.1, -0.1, 0])
        perturbed = v + np.array([0, 1e-8, -1e-8])
        joints = [line_source(prior, v, -1, 1), line_source(prior, perturbed, -1, 1)]
        exact_directions = redundancy_binary_sources(joints)
        self.assertEqual(exact_directions.redundancy_nats, 0)
        approximate = redundancy_binary_sources(joints, geometry_tol=1e-5)
        self.assertGreater(approximate.redundancy_nats, 0)
        self.assertGreater(approximate.max_garbling_residual, 0)

    def test_validation_and_no_input_mutation(self):
        joint = np.array([[0.3, 0.2], [0.1, 0.4]])
        before = joint.copy()
        for solver in (redundancy_binary_target, redundancy_binary_sources):
            solver([joint, joint.copy()])
            np.testing.assert_array_equal(joint, before)
            for bad in ([], [np.ones((2, 2))], [np.array([[np.nan], [0]])],
                        [np.array([[-0.1, 0.6], [0.1, 0.4]])],
                        [np.array([[0.6], [0.4]]), np.array([[0.5], [0.5]])]):
                with self.subTest(solver=solver.__name__, bad=repr(bad)):
                    with self.assertRaises(ValueError):
                        solver(bad)
            with self.assertRaises(ValueError):
                solver([joint], atol=-1)
        with self.assertRaises(ValueError):
            redundancy_binary_target([np.ones((3, 2)) / 6])
        with self.assertRaises(ValueError):
            redundancy_binary_sources([np.ones((2, 3)) / 6])
        with self.assertRaises(ValueError):
            redundancy_binary_sources([joint], geometry_tol=-1)


if __name__ == "__main__":
    unittest.main()
