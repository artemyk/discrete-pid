"""Conditional input contract and batch reduction for the binary-target solver."""

import unittest
from unittest.mock import patch

import numpy as np

from discrete_pid import redundancy_binary_target as solve
from discrete_pid import binary_target
from test_redundancy import finite_call_lp, mutual_information


class TargetChannelTests(unittest.TestCase):
    def test_integer_float_ragged_and_iterable_inputs(self):
        prior = np.array([2, 3])
        channels = [np.array([[9, 1], [1, 9]], dtype=np.uint32),
                    np.array([[5, 0, 5], [0, 5, 5]], dtype=np.uint32)]
        joints = [prior[:, None] / prior.sum() * c / 10 for c in channels]
        expected = finite_call_lp(joints)
        for data in (channels, [c.tolist() for c in channels],
                     [c / 10 for c in channels], iter(channels)):
            result = solve(prior, data, return_channel=True, return_garblings=True)
            self.assertAlmostEqual(result.redundancy_nats, expected, places=10)
            for joint, kernel in zip(joints, result.garblings):
                np.testing.assert_allclose(joint @ kernel, result.target_auxiliary_joint, atol=1e-10)
            self.assertEqual(result.input_adjustment, 0)

    def test_large_uint32_entries_and_row_totals(self):
        m = 2**32 - 1
        counts = np.array([[[m, m, 0], [0, m, m]],
                           [[m, m, 0], [0, m, m]]], dtype=np.uint32)
        before = counts.copy()
        integer = solve([1, 1], counts)
        floating = solve([.5, .5], counts.astype(float) / (2 * m))
        self.assertAlmostEqual(integer.redundancy_bits, .5, places=13)
        self.assertAlmostEqual(integer.redundancy_nats, floating.redundancy_nats, places=14)
        np.testing.assert_array_equal(counts, before)

    def test_float_row_scaling_and_no_mutation(self):
        channel = np.array([[.6, .4], [.2, .8]])
        scaled = channel * np.array([1e308, 1e-300])[:, None]
        before = scaled.copy()
        expected = solve([2, 3], [channel]).redundancy_nats
        self.assertAlmostEqual(solve([2e300, 3e300], [scaled]).redundancy_nats, expected, places=13)
        np.testing.assert_array_equal(scaled, before)
        joint = np.array([.4, .6])[:, None] * channel
        self.assertAlmostEqual(expected, mutual_information(joint), places=13)

    def test_zero_prior_rows_and_zero_source_columns(self):
        channel = np.array([[3, 7, 0], [0, 0, 0]], dtype=np.uint32)
        result = solve([1, 0], [channel], return_channel=True, return_garblings=True)
        self.assertEqual(result.redundancy_nats, 0)
        np.testing.assert_array_equal(result.channel, np.ones((2, 1)))
        np.testing.assert_array_equal(result.garblings[0], np.ones((3, 1)))
        # A degenerate prior still requires all other entries to be valid.
        with self.assertRaises(ValueError):
            solve([1, 0], [channel, np.array([[1., 0], [np.nan, 0]])])

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

    def test_batch_hulls_agree_with_single_hull_and_lp(self):
        rng = np.random.default_rng(827)
        for size in (2, 5, 17):
            channels = rng.dirichlet(np.ones(size), size=(7, 2))
            prior = np.array([.3, .7])
            one = solve(prior, channels, return_channel=True)
            with patch.object(binary_target, '_BATCH_ENTRIES', 2 * size):
                many = solve(prior, channels, return_channel=True, return_garblings=True)
                listed = solve(prior, list(channels), return_channel=True)
            self.assertAlmostEqual(one.redundancy_nats, many.redundancy_nats, places=12)
            self.assertAlmostEqual(one.redundancy_nats, listed.redundancy_nats, places=12)
            self.assertAlmostEqual(one.redundancy_nats,
                                   finite_call_lp(list(prior[None, :, None] * channels)), places=9)
            for c, kernel in zip(channels, many.garblings):
                np.testing.assert_allclose(c @ kernel, many.channel, atol=1e-9)

    def test_continuity_across_tied_hull_knots(self):
        base = np.array([[.9, .1], [.1, .9]])
        reference = solve([1, 1], [base, base]).redundancy_nats
        errors = []
        for epsilon in (1e-3, 1e-5, 1e-7):
            perturbed = base + epsilon * np.array([[-1, 1], [1, -1]])
            result = solve([1, 1], [base, perturbed])
            expected = mutual_information(.5 * perturbed)
            self.assertAlmostEqual(result.redundancy_nats, expected, places=12)
            errors.append(abs(result.redundancy_nats - reference))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2], 1e-6)

    def test_normalization_batches_are_bounded(self):
        channels = np.tile(np.array([[[9, 1], [1, 9]]], dtype=np.uint32), (101, 1, 1))
        normalize = binary_target._normalize_channels
        seen = []

        def checked(raw, prior):
            seen.append(raw.size)
            self.assertLessEqual(raw.size, 32)
            return normalize(raw, prior)

        for data in (channels, iter(channels)):
            with patch.object(binary_target, '_BATCH_ENTRIES', 32), \
                 patch.object(binary_target, '_normalize_channels', side_effect=checked):
                result = solve([1, 1], data)
            self.assertAlmostEqual(result.redundancy_nats,
                                   mutual_information(channels[0] / 20), places=12)
        self.assertEqual(sum(seen), 2 * channels.size)


if __name__ == '__main__':
    unittest.main()
