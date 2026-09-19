# discrete-pid

Fast Python implementations of two special cases of **Blackwell redundancy**
for discrete random variables. Given a target `Y` and sources `X_1, ..., X_k`,
it maximizes `I(Y; Q)` over experiments `Q` obtainable by stochastic processing
of **every** source, with a common joint distribution `P(Y, Q)`.

| Function | Applicable inputs | Arithmetic cost |
| --- | --- | --- |
| `redundancy_binary_target` | Binary target; any number of sources with arbitrary finite alphabets | `O(N log N)` time and `O(N)` working memory |
| `redundancy_binary_sources` | Arbitrary finite target; every source has at most two active states | `O(k d)` time for binary input tables |

Here `N` is the total number of source states, `k` is the number of sources,
and `d` is the number of target states. Bounds exclude logarithm evaluation
and integer bit costs; padded zero columns add their input size.
The output is redundancy and an optimal common experiment, not all PID atoms.

## Install

Requires Python 3.10+ and NumPy. Install from this repository:

```bash
git clone https://github.com/artemyk/discrete-pid.git
cd discrete-pid
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Optional extras add SciPy for garbling reconstruction and tests, or Numba
for acceleration:

```bash
python -m pip install ".[test]"   # SciPy
python -m pip install ".[speed]"  # Numba
```

## Quick example

Each input is a joint probability table `P(Y, X_i)`: **rows are target states,
columns are source states**. All tables must give the same target marginal
after normalization. The binary-target solver takes normalized probabilities;
the binary-source solver takes **unnormalized integer counts by default**
to preserve exact collinearity. Source–target pairwise tables suffice; a full
joint distribution of all sources is not required.

```python
import numpy as np
from discrete_pid import redundancy_binary_target, redundancy_binary_sources

# A fair binary target observed through a binary symmetric channel
# and a three-output binary erasure channel.
bsc = np.array([[0.45, 0.05], [0.05, 0.45]])
bec = np.array([[0.25, 0.0, 0.25], [0.0, 0.25, 0.25]])
result = redundancy_binary_target([bsc, bec])
print(f"Redundancy: {result.redundancy_bits:.9f} bits")  # 0.331877754
print(result.target_auxiliary_joint)  # P(Y, Q)

# The second algorithm allows any target alphabet, but all sources must
# be binary. Supply integer counts, without dividing by their totals.
# These sources observe a uniform three-state target.
counts = [np.array([[4, 26], [16, 14], [10, 20]]),
          np.array([[14, 16], [26, 4], [20, 10]])]
result = redundancy_binary_sources(counts)
print(f"Redundancy: {result.redundancy_bits:.9f} bits")  # 0.043954630
for joint, kernel in zip(counts, result.garblings):
    assert np.allclose((joint / joint.sum()) @ kernel, result.target_auxiliary_joint)

# With floating-point probabilities, explicitly allow approximate collinearity.
float_joints = [joint / joint.sum() for joint in counts]
approximate = redundancy_binary_sources(float_joints, tolerance=1e-12)
print(f"Redundancy: {approximate.redundancy_bits:.9f} bits")  # 0.043954630
```

A runnable example of each algorithm is included:

```bash
python examples/quickstart.py
```

## Algorithms and outputs

**Binary target.** Represent a source by its posterior probabilities
`r = P(Y=1 | X_i)` and call function `C_i(t) = E[max(r - t, 0)]`.
The meet is the greatest convex function below all source call functions:
sort their knots, construct a lower convex hull, and recover posteriors and
weights from its slope jumps.
An optimal `Q` can have more than two outputs even when one source is binary.

This implements the classical binary Blackwell meet construction of
[Bertschinger and Rauh (2014), Section 4](https://arxiv.org/abs/1401.3146).
Their channel *input* is the variable called the *target* here.

**All binary sources.** Each source has a posterior segment containing the
prior. An uninformative source or segments on distinct lines give zero
redundancy. Otherwise, intersect their scalar intervals; the endpoints and
mean-preserving weights define the meet. Garbling matrices follow directly.

Both functions return a `RedundancyResult` with:

- `redundancy_nats` and `redundancy_bits`.
- `target_prior`: `P(Y)`.
- `posteriors`: a `(d, q)` array whose columns are `P(Y | Q)`.
- `posterior_weights`: the vector `P(Q)`.
- `target_auxiliary_joint`: the `(d, q)` table `P(Y, Q)`.
- `garblings`: matrices `K_i = P(Q | X_i)` of shape `(m_i, q)`, satisfying
  `P(Y,X_i) @ K_i = P(Y,Q)` to numerical precision for normalized inputs.
- `max_garbling_residual`: largest absolute entrywise error in this equality,
  or `None` when garblings were not requested.
- `input_adjustment`: largest entrywise change during within-tolerance
  normalization or marginal reconciliation, measured after converting
  binary-source weights to probabilities; zero in integer mode.

The binary-source function always returns garblings. For the binary-target
function, use `return_garblings=True` with the SciPy extra. This optional
reconstruction solves linear programs **outside** the `O(N log N)` bound.

## Performance

NumPy batches equal-shaped tables and sorts hull knots. Optional Numba
compiles the binary-target hull scan where `np.longdouble` has `float64`
precision (including Apple Silicon). Compilation or cache loading happens
at import time, so the first solver call has no compilation overhead.
Wider `longdouble` uses the Python scan to preserve precision, since
[Numba does not support extended-precision floats](https://numba.readthedocs.io/en/stable/reference/numpysupported.html#scalar-types).
The package works without Numba. Integer-source geometry always uses
arbitrary-precision integer/rational arithmetic.

## Numerical behavior

With more than two target states, an arbitrarily small perturbation can
move binary-source posterior segments onto different lines, changing positive
redundancy to zero. Thus the default `tolerance=None` requires integer inputs.
Use original counts or exact integer weights; **do not round or cast
probabilities to integers** to bypass this check.

- Integer tables may have different positive totals, but their normalized
  target marginals must agree exactly. Integer cross-products check
  collinearity; rational arithmetic gives endpoints, weights, and garblings.
  Arbitrary-size integers prevent overflow. NumPy integer arrays and nested
  lists are accepted. A numerical `tolerance` does not weaken integer checks.
- Any floating-point input requires a finite, nonnegative `tolerance`, even
  if its values are integers. As in the example above, this compares posterior
  directions scaled to maximum absolute coordinate one, allowing sign reversal.
  Nearly aligned lines may be treated as collinear when exact redundancy is
  zero. Neither `tolerance` nor a small `max_garbling_residual` bounds the
  redundancy error; `tolerance=0` still uses floating-point arithmetic.
- Floating-point binary-source tables may be unnormalized. After normalization,
  target-marginal discrepancies within `atol` (default `1e-12`) are reconciled
  by row scaling. `atol` neither controls collinearity nor permits float inputs.
- Binary-target tables must sum to one within `atol`; normalization and marginal
  discrepancies within `atol` are reconciled. The hull uses platform-dependent
  extended precision and discards slope jumps at roundoff scale, so extremely
  small atoms may be lost.
- Outputs are floating point even with exact geometry. In integer mode, a
  positive probability too small to represent raises `ArithmeticError`.

Inputs must be finite, nonnegative, nonempty, and have positive total mass.
Zero source columns are allowed; zero target rows must agree across sources.
Binary-target tables have exactly two rows; binary-source tables have at most
two positive-mass columns. Input arrays are not modified.

## Tests

```bash
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

Tests include analytical examples, an independent LP oracle, agreement of
both algorithms, garbling feasibility, and numerical edge cases. GitHub
Actions runs tests and examples on Python 3.10, 3.12, and 3.13, with a separate
Numba check on Python 3.12.

## References and authorship

- Artemy Kolchinsky, [*A Novel Approach to the Partial Information
  Decomposition*](https://doi.org/10.3390/e24030403), Entropy **24**, 403 (2022).
  Defines Blackwell redundancy and its common-garbling optimization.
- Nils Bertschinger and Johannes Rauh, [*The Blackwell relation defines no
  lattice*](https://arxiv.org/abs/1401.3146), 2014. Section 4 describes the
  binary-target lattice and its constructive meet.

The code, tests, and documentation in this repository were written by
**OpenAI Codex**, at Artemy Kolchinsky's request. The binary-target solver
was adapted from the existing implementation in the BlackwellPID-Discrete
research project. Mathematical sources are credited above.
