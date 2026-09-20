# discrete-pid

Fast Python implementations of two special cases of **Blackwell redundancy**
for discrete random variables. Given a target `Y` and sources `X_1, ..., X_k`,
it maximizes `I(Y; Q)` over experiments `Q` obtainable by stochastic processing
of **every** source, with a common joint distribution `P(Y, Q)`.

| Function | Applicable inputs | Arithmetic cost |
| --- | --- | --- |
| `redundancy_binary_target` | Binary target; any number of sources with arbitrary finite alphabets | `O(N log N)` time and `O(N)` working memory |
| `redundancy_binary_sources` | Arbitrary finite target; every source is binary | `O(k d)` time for binary input tables |

Here `N` is the total number of source states, `k` is the number of sources,
and `d` is the number of target states. Bounds exclude logarithm evaluation
and integer bit costs.
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

The binary-target solver takes normalized joint tables `P(Y, X_i)`.
The binary-source solver takes **a target prior and conditional channels
`P(X_i | Y)`**. In both cases, rows are target states and columns are source
states. Pairwise information suffices; a joint distribution of all sources
is not required.

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

# Binary sources: a prior P(Y) and integer conditional weights.
# Each channel has a constant row sum (30 here), used as its denominator.
prior = np.array([1/3, 1/3, 1/3])
channels = np.array([[[4, 26], [16, 14], [10, 20]],
                     [[14, 16], [26, 4], [20, 10]]], dtype=np.uint32)
result = redundancy_binary_sources(prior, channels)
print(f"Redundancy: {result.redundancy_bits:.9f} bits")  # 0.043954630
for channel, kernel in zip(channels / 30, result.garblings):
    assert np.allclose((prior[:, None] * channel) @ kernel,
                       result.target_auxiliary_joint)

# Floating channels require an explicit collinearity tolerance.
approximate = redundancy_binary_sources(prior, channels / 30, tolerance=1e-12)
print(f"Redundancy: {approximate.redundancy_bits:.9f} bits")  # 0.043954630
```

A runnable example of each algorithm is included:

```bash
python examples/quickstart.py
```

## Binary-source input format

```python
redundancy_binary_sources(prior, channels, *, tolerance=None)
```

- `prior`: a length-`d` vector of nonnegative target weights, normalized internally.
  Floating-point priors are allowed without a tolerance: only their support
  enters the exact collinearity check.
- `channels`: a list of `(d, 2)` arrays, or one `(k, d, 2)` array. By default,
  entries must be integers in `[0, 2**32 - 1]`; `np.uint32` is recommended.
  Other integer dtypes and nested lists are accepted if their values fit.
- Within each channel, **every positive-prior row must have the same positive
  sum `L_i`**. The entries represent `P(X_i=x | Y=y) = channels[i,y,x] / L_i`.
  Different sources may have different denominators. The row sum itself may
  exceed `uint32`; it is computed in `uint64`.
- Zero-prior rows are ignored and may be all zero. A constant source still has
  two columns, one of which may be zero. Inputs are never modified.

This replaces the earlier joint-count API. General joint counts, or integer
rows with different totals, cannot be passed as conditional weights. Supply
an exact channel representation with a common denominator; do not round or
cast probabilities to integers. Larger integer weights are rejected before
conversion to `uint32`.

With floating-point channels, explicitly set a finite, nonnegative `tolerance`.
Rows may then contain unnormalized nonnegative weights and are normalized
individually. Mixing floating-point and integer channels uses this approximate
mode. A tolerance does not relax checks when all channels are integer.

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
  normalization or marginal reconciliation for binary-target inputs; zero
  for binary-source inputs, whose weights define their normalization.

The binary-source function always returns garblings. For the binary-target
function, use `return_garblings=True` with the SciPy extra. This optional
reconstruction solves linear programs **outside** the `O(N log N)` bound.

## Performance and numerical behavior

For binary sources, anchored differences of conditional weights allow exact
collinearity checks with signed `int64` differences and `uint64` products.
Each product is at most `(2**32 - 1)**2`. Endpoint comparisons and the
integer numerators used to construct garblings have the same bound.
There are no GCDs or arbitrary-precision integers. Probabilities and mutual
information are evaluated in floating point.

Optional Numba compiles this scan and the binary-target hull scan at import,
so solver calls have no compilation overhead. The package also works without
Numba. On platforms with wider `np.longdouble`, the hull uses the Python scan
to preserve precision; Numba accelerates the hull when `longdouble` has
`float64` precision, including Apple Silicon.

With more than two target states, arbitrarily small perturbations can rotate
posterior segments onto different lines and change positive redundancy to zero.
That is why floating-point channels require explicit opt-in. The tolerance
compares anchored channel differences scaled to maximum absolute coordinate
one, allowing sign reversal. Neither it nor a small `max_garbling_residual`
bounds the error in redundancy; `tolerance=0` still uses floating-point arithmetic.

Binary-target tables must sum to one within `atol` (default `1e-12`) and have
the same target marginal. Small discrepancies are reconciled. Their hull uses
platform-dependent extended precision and discards slope jumps at roundoff
scale, so extremely small atoms may be lost. All inputs must be finite and
nonnegative; zero source columns are allowed for this solver.

## Tests

```bash
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

Tests include analytical examples, an independent LP oracle, agreement of
both algorithms, an exact rational posterior oracle, garbling feasibility,
`uint32` boundary cases, and compiled/Python fallback agreement. GitHub
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
