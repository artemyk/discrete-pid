"""Run with: python examples/quickstart.py (after installing the package)."""

import numpy as np

from discrete_pid import redundancy_binary_sources, redundancy_binary_target


def main():
    # Binary target: a binary symmetric channel and a three-state erasure
    # channel, both observing the same fair bit.
    bsc = np.array([[0.45, 0.05], [0.05, 0.45]])
    bec = np.array([[0.25, 0.0, 0.25], [0.0, 0.25, 0.25]])
    result = redundancy_binary_target([bsc, bec])
    print(f"Binary target: {result.redundancy_bits:.9f} bits")
    print("  P(Y=1 | Q):", result.posteriors[1])
    print("  P(Q):", result.posterior_weights)

    # Three-state target, two binary sources. Their posterior intervals are
    # [-2,1] and [-1,2] along the line prior + t*direction.
    # Supply the prior and conditional weights with a constant row sum.
    prior = np.ones(3) / 3
    channels = np.array([[[4, 26], [16, 14], [10, 20]],
                         [[14, 16], [26, 4], [20, 10]]], dtype=np.uint32)
    result = redundancy_binary_sources(prior, channels)
    print(f"Binary sources: {result.redundancy_bits:.9f} bits")
    print("  P(Q):", result.posterior_weights)
    for channel, kernel in zip(channels / 30, result.garblings):
        np.testing.assert_allclose((prior[:, None] * channel) @ kernel, result.target_auxiliary_joint,
                                   rtol=0, atol=1e-12)
    print("  Common experiment verified from both sources.")
    approximate = redundancy_binary_sources(prior, channels / 30, tolerance=1e-12)
    print(f"Floating channels: {approximate.redundancy_bits:.9f} bits")


if __name__ == "__main__":
    main()
