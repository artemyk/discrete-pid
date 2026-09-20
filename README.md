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
Optional binary-target garblings take an additional `O(N + k q)` time after
sorting and `O(N + k q)` sparse storage, where `q` is the number of output states.
Since `q <= N`, their worst-case reconstruction cost is `O(k N)`.
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

Optional extras add SciPy for sparse binary-target garblings and tests, or
Numba for acceleration:

```bash
python -m pip install ".[garblings]"  # SciPy sparse matrices; needed for the example below
python -m pip install ".[speed]"      # Numba
```

## Quick example

Both solvers take **a target prior and conditional channels `P(X_i | Y)`**.
Rows are target states and columns are source states. Pairwise information suffices; a joint distribution of all sources
is not required.

```python
import numpy as np
from discrete_pid import redundancy_binary_target, redundancy_binary_sources

# A fair binary target observed through a binary symmetric channel
# and a three-output binary erasure channel.
bsc = np.array([[9, 1], [1, 9]], dtype=np.uint32)
bec = np.array([[5, 0, 5], [0, 5, 5]], dtype=np.uint32)
result = redundancy_binary_target([1, 1], [bsc, bec],
                                  return_channel=True, return_garblings=True)
print(f"Redundancy: {result.redundancy_bits:.9f} bits")  # 0.331877754
print(result.channel)  # P(Q | Y)
print(result.garblings[0].toarray())  # P(Q | X_1), stored as a SciPy CSR matrix
for channel, kernel in zip([bsc / 10, bec / 10], result.garblings):
    assert np.allclose((0.5 * channel) @ kernel, result.target_auxiliary_joint)
# Binary targets also accept floats without a tolerance.
floating = redundancy_binary_target([.5, .5], [bsc / 10, bec / 10])

# Binary sources: a prior P(Y) and integer conditional weights.
# Each channel has a constant row sum (30 here), used as its denominator.
prior = np.array([1/3, 1/3, 1/3])
channels = np.array([[[4, 26], [16, 14], [10, 20]],
                     [[14, 16], [26, 4], [20, 10]]], dtype=np.uint32)
result = redundancy_binary_sources(prior, channels,
                                   return_channel=True, return_garblings=True)
print(f"Redundancy: {result.redundancy_bits:.9f} bits")  # 0.043954630
for channel, kernel in zip(channels / 30, result.garblings):
    assert np.allclose((prior[:, None] * channel) @ kernel,
                       result.target_auxiliary_joint)

# Binary-source floating channels require an explicit collinearity tolerance.
approximate = redundancy_binary_sources(prior, channels / 30, tolerance=1e-12)
print(f"Redundancy: {approximate.redundancy_bits:.9f} bits")  # 0.043954630
```

A runnable example of each algorithm is included:

```bash
python examples/quickstart.py
```

## Input format

```python
redundancy_binary_target(prior, channels, *,
                         return_channel=False, return_garblings=False, atol=1e-12)
redundancy_binary_sources(prior, channels, *, tolerance=None,
                          return_channel=False, return_garblings=False)
```

- `prior`: nonnegative target weights, normalized internally; floats are allowed.
  It has two entries for the binary-target solver, or `d` entries for binary sources.
- `channels`: a list or iterable of `(d, m_i)` arrays. Rows are target states.
  The target solver requires `d=2`; the source solver requires every `m_i=2`.
  A packed `(k, d, m)` array works when source sizes agree.
- **Integer channels:** entries in `[0, 2**32 - 1]`; `np.uint32` is recommended.
  Every positive-prior row within a source must have the same positive sum `L_i`.
  Entries represent `P(X_i=x | Y=y) = channels[i,y,x] / L_i`.
  Denominators may differ across sources and may exceed `uint32`.
- **Floating channels:** finite nonnegative row weights, normalized separately.
  Binary targets accept these directly. Binary sources require an explicit finite,
  nonnegative `tolerance` for approximate collinearity (see below).
- Zero-prior rows are ignored and may be all zero. Zero source columns are allowed.
  Inputs are never modified.

This replaces the earlier joint-table APIs. To convert normalized joint tables,
use `prior = joints[0].sum(axis=1)` and divide each positive-prior row by its
prior probability. Do not pass general joint counts as integer conditional weights,
round probabilities to integers, or cast oversized weights to `uint32`.

For binary sources, mixing floating and integer channels uses the approximate
mode; a tolerance does not relax checks when all channels are integer.
The binary-target solver validates integer channels individually, even in mixed lists,
then uses floating-point arithmetic for all inputs.

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
Optional garblings use the inverse-transform martingale coupling of
[Jourdain and Margheriti](https://doi.org/10.1214/20-EJP543). In posterior
coordinates, a garbling is the reverse conditional of a martingale coupling
from `P(Y=1 | Q)` to `P(Y=1 | X_i)`. Merging quantile intervals and matching
their mean deficits and excesses constructs sparse couplings without linear
programming.

**All binary sources.** Each source has a posterior segment containing the
prior. An uninformative source or segments on distinct lines give zero
redundancy. Otherwise, intersect their scalar intervals; the endpoints and
mean-preserving weights define the meet. Garbling matrices follow directly.

Both functions return a `RedundancyResult`. **Both output flags default to
`False` and are independent:**

```python
result = redundancy_binary_sources(prior, channels)  # information only
with_channel = redundancy_binary_sources(prior, channels, return_channel=True)
with_garblings = redundancy_binary_sources(prior, channels, return_garblings=True)
# The binary-target solver accepts the same two flags.
```

Always available:

- `redundancy_nats`, `redundancy_bits`, and `target_prior` (`P(Y)`).
- `input_adjustment`: zero; the common prior is supplied directly and weights
  are normalized by definition.

With `return_channel=True`:

- `channel`: the row-stochastic `(d, q)` matrix **`P(Q | Y)`**.
- `posteriors`: the `(d, q)` matrix `P(Y | Q)`.
- `posterior_weights`: `P(Q)`.
- `target_auxiliary_joint`: the `(d, q)` table `P(Y, Q)`.

With `return_garblings=True`:

- `garblings`: a tuple of `(m_i, q)` matrices **`P(Q | X_i)`**, satisfying
  `P(Y,X_i) @ K_i = P(Y,Q)` to numerical precision. Binary-target matrices are
  SciPy CSR sparse matrices; binary-source matrices are dense NumPy arrays.
  Use `K_i.toarray()` if a dense binary-target matrix is needed.
- `max_garbling_residual`: largest absolute entrywise error in this equality.

Unrequested fields are `None`. Requesting garblings does not implicitly enable
channel output, or vice versa. Garbling construction and residual evaluation
are skipped by default. Binary-target garblings require the SciPy extra for
sparse storage; no linear programs are solved. Including these outputs gives
`O(N log N + k q)` total time. Materializing dense matrices costs `O(N q)`.
On zero-prior target rows, the returned channel is set to `P(Q)` as an arbitrary
stochastic extension; the joint-distribution identities are unaffected.

## Performance and numerical behavior

For binary sources, anchored differences of conditional weights allow exact
collinearity checks with signed `int64` differences and `uint64` products.
Each product is at most `(2**32 - 1)**2`. Endpoint comparisons and the
integer numerators used to construct garblings have the same bound.
There are no GCDs or arbitrary-precision integers. Probabilities and mutual
information are evaluated in floating point.

Both solvers validate integer row sums with an optional Numba loop, compiled
at import, that stops at the first mismatch. It avoids full source-by-target
row-sum and comparison arrays; without Numba, the same bare loop runs in
Python and the package warns once at import that installing `discrete-pid[speed]`
is faster.

Optional Numba compiles this scan, the binary-target hull scan, and the
martingale coupling scan at import,
so solver calls have no compilation overhead. The package also works without
Numba. On platforms with wider `np.longdouble`, the hull and martingale coupling
use Python scans to preserve precision; Numba accelerates them when `longdouble` has
`float64` precision, including Apple Silicon.

With more than two target states, arbitrarily small perturbations can rotate
posterior segments onto different lines and change positive redundancy to zero.
That is why the binary-source solver requires explicit opt-in for floating channels. The tolerance
compares anchored channel differences scaled to maximum absolute coordinate
one, allowing sign reversal. Neither it nor a small `max_garbling_residual`
bounds the error in redundancy; `tolerance=0` still uses floating-point arithmetic.

For a **binary target, redundancy is continuous** (Kolchinsky, 2022,
Section 5.5 and Appendix D), so the discontinuous collinearity test is absent.
Exact arithmetic is unnecessary for numerical evaluation: integer weights are
normalized to floating point before computing the hull. The hull uses
platform-dependent extended precision and discards slope jumps at roundoff
scale, so extremely small atoms may be lost. `atol` controls numerical
consistency checks; it is not a collinearity tolerance or a redundancy error bound.

The target solver processes bounded batches and retains only their lower hulls
for the final hull of their union. This avoids full-input temporary matrices;
worst-case working memory remains `O(N)`. Normalized joint tables are retained
only when garblings are requested.

## Tests

```bash
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

Tests include analytical examples, an independent LP oracle, agreement of
both algorithms, an exact rational posterior oracle, garbling feasibility,
`uint32` boundary cases, sparse garblings from known feasible channels, and
compiled/Python fallback agreement. GitHub
Actions runs tests and examples on Python 3.10, 3.12, and 3.13, with a separate
Numba check on Python 3.12.

## References and authorship

- Artemy Kolchinsky, [*A Novel Approach to the Partial Information
  Decomposition*](https://doi.org/10.3390/e24030403), Entropy **24**, 403 (2022).
  Defines Blackwell redundancy and its common-garbling optimization.
- Nils Bertschinger and Johannes Rauh, [*The Blackwell relation defines no
  lattice*](https://arxiv.org/abs/1401.3146), 2014. Section 4 describes the
  binary-target lattice and its constructive meet.
- Benjamin Jourdain and William Margheriti, [*A new family of one dimensional
  martingale couplings*](https://doi.org/10.1214/20-EJP543), 2020. The
  inverse-transform coupling supplies sparse binary-target garblings.

The code, tests, and documentation in this repository were written by
**OpenAI Codex**, at Artemy Kolchinsky's request. The binary-target solver
was adapted from the existing implementation in the BlackwellPID-Discrete
research project. Mathematical sources are credited above.
