"""Independent opt-in outputs and avoidance of optional reconstruction work."""

from functools import partial
import unittest
from unittest.mock import patch

import numpy as np

from discrete_pid import redundancy_binary_sources, redundancy_binary_target
from discrete_pid import _source_scan


class OutputFlagTests(unittest.TestCase):
    def test_flags_are_independent_on_all_paths(self):
        informative = np.array([[[9, 1], [1, 9]], [[8, 2], [2, 8]]], dtype=np.uint32)
        constant = np.full((2, 2, 2), 5, dtype=np.uint32)
        for prior, counts in (([.3, .7], informative), ([.3, .7], constant),
                              ([1., 0.], informative)):
            channels = counts / counts.sum(axis=2, keepdims=True)
            joint = channels * np.array(prior)[None, :, None]
            solvers = [partial(redundancy_binary_target, joint),
                       partial(redundancy_binary_sources, prior, counts),
                       partial(redundancy_binary_sources, prior, channels, tolerance=1e-12)]
            for solver in solvers:
                reference = solver(return_channel=True, return_garblings=True)
                default = solver()
                for channel_flag, garbling_flag in ((False, False), (False, True),
                                                     (True, False), (True, True)):
                    with self.subTest(prior=prior, solver=solver, flags=(channel_flag, garbling_flag)):
                        result = solver(return_channel=channel_flag, return_garblings=garbling_flag)
                        self.assertAlmostEqual(result.redundancy_nats, reference.redundancy_nats, places=14)
                        self.assertEqual(result.channel is not None, channel_flag)
                        self.assertEqual(result.posteriors is not None, channel_flag)
                        self.assertEqual(result.posterior_weights is not None, channel_flag)
                        self.assertEqual(result.target_auxiliary_joint is not None, channel_flag)
                        self.assertEqual(result.garblings is not None, garbling_flag)
                        self.assertEqual(result.max_garbling_residual is not None, garbling_flag)
                        if channel_flag:
                            np.testing.assert_allclose(result.channel.sum(axis=1), 1, atol=1e-14)
                            np.testing.assert_allclose(np.array(prior)[:, None] * result.channel,
                                                       reference.target_auxiliary_joint, atol=1e-14)
                            if prior[1] == 0:
                                np.testing.assert_array_equal(result.channel[1], result.posterior_weights)
                        if garbling_flag:
                            for j, kernel in zip(joint, result.garblings):
                                np.testing.assert_allclose(j @ kernel, reference.target_auxiliary_joint,
                                                           rtol=0, atol=1e-10)
                self.assertEqual(default.redundancy_nats, reference.redundancy_nats)
                for field in ('channel', 'posteriors', 'posterior_weights',
                              'target_auxiliary_joint', 'garblings', 'max_garbling_residual'):
                    self.assertIsNone(getattr(default, field))

    def test_default_target_never_calls_garbling_lp(self):
        joint = np.array([[.45, .05], [.05, .45]])
        with patch('discrete_pid.binary_target._garbling', side_effect=AssertionError('LP called')):
            redundancy_binary_target([joint])
            redundancy_binary_target([joint], return_channel=True)

    def test_source_scan_omits_kernel_allocation_including_early_returns(self):
        for counts in (np.array([[[9, 1], [1, 9]]], dtype=np.uint32),
                       np.ones((3, 2, 2), dtype=np.uint32)):
            for fn in (_source_scan._scan, _source_scan.geometry):
                meet, omitted = fn(counts, np.ones(2, dtype=bool), False)
                expected, kernels = fn(counts, np.ones(2, dtype=bool), True)
                self.assertEqual(omitted.shape[0], 0)
                self.assertEqual(kernels.shape[0], len(counts))
                np.testing.assert_array_equal(meet, expected)


if __name__ == '__main__':
    unittest.main()
