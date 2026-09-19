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
    # Keep the joint tables as integer counts for exact collinearity checks.
    joints = [np.array([[4, 26], [16, 14], [10, 20]]),
              np.array([[14, 16], [26, 4], [20, 10]])]
    result = redundancy_binary_sources(joints)
    print(f"Binary sources: {result.redundancy_bits:.9f} bits")
    print("  P(Q):", result.posterior_weights)
    for joint, kernel in zip(joints, result.garblings):
        np.testing.assert_allclose((joint / joint.sum()) @ kernel, result.target_auxiliary_joint,
                                   rtol=0, atol=1e-12)
    print("  Common experiment verified from both sources.")


if __name__ == "__main__":
    main()
