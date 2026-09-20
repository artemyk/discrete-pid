# discrete-pid

Fast Python implementations of **Blackwell redundancy** for discrete variables:
the maximum `I(Y; Q)` over experiments `Q` obtainable by stochastic processing
of every source `X_i`. The package computes redundancy and, optionally, a common
experiment and its garblings; it does not compute all PID atoms.

| Function | Applicable inputs | Time / working memory |
| --- | --- | --- |
| `redundancy_binary_target` | Binary target; arbitrary finite sources | `O(N log N)` / `O(N)` |
| `redundancy_binary_sources` | Arbitrary finite target; binary sources | `O(k d)` / `O(k d)` |

Here `k` is the number of sources, `d` the number of target states, and `N` the
total number of source states. These bounds exclude optional output construction,
logarithm evaluation, and integer bit costs.

## Install

Requires Python 3.10+ and NumPy:

```bash
git clone https://github.com/artemyk/discrete-pid.git
cd discrete-pid
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Optional extras provide Numba acceleration and SciPy sparse binary-target garblings:

```bash
python -m pip install ".[speed,garblings]"  # Includes everything used below
```

## Quick example

Both solvers take a target prior and conditional channels `P(X_i | Y)`.
Rows are target states; columns are source states. No full joint law of all
sources is required.

```python
import numpy as np
from discrete_pid import redundancy_binary_target, redundancy_binary_sources

# A fair target, a binary symmetric channel, and a three-output erasure channel.
bsc = np.array([[9, 1], [1, 9]], dtype=np.uint32)
bec = np.array([[5, 0, 5], [0, 5, 5]], dtype=np.uint32)
result = redundancy_binary_target([1, 1], [bsc, bec],
                                  return_channel=True, return_garblings=True)
print(result.redundancy_bits)         # 0.331877754
print(result.channel)                 # P(Q | Y)
print(result.garblings[0].toarray())   # P(Q | X_1); stored as a SciPy CSR matrix
# Binary targets also accept floats without a tolerance.
floating = redundancy_binary_target([.5, .5], [bsc / 10, bec / 10])

# Three target states, two binary sources; every channel row sums to 30.
prior = [1, 1, 1]
channels = np.array([[[4, 26], [16, 14], [10, 20]],
                     [[14, 16], [26, 4], [20, 10]]], dtype=np.uint32)
print(redundancy_binary_sources(prior, channels).redundancy_bits)  # 0.043954630
# Binary-source floats require an explicit collinearity tolerance.
approximate = redundancy_binary_sources(prior, channels / 30, tolerance=1e-12)
print(approximate.redundancy_bits)                                # 0.043954630
```

Run the complete example, including checks of the garbling identities:

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

- `prior`: nonnegative target weights with positive total, normalized internally;
  floats are allowed.
  It has two entries for binary targets, or `d` entries for binary sources.
- `channels`: an iterable of `(d, m_i)` arrays, or a packed `(k, d, m)` array
  when source sizes agree. Binary targets require `d=2`; binary sources require
  every `m_i=2`. Inputs are never modified.
- **Integer channels:** entries in `[0, 2**32 - 1]`; `np.uint32` is recommended.
  Every positive-prior row within a source must have the same positive sum `L_i`.
  The probabilities are `channels[i,y,x] / L_i`. Denominators may differ across
  sources and may exceed `uint32`. Other fitting integer dtypes/lists also work.
- **Floating channels:** finite nonnegative weights, normalized row by row.
  Binary targets accept these directly; binary sources require a finite,
  nonnegative `tolerance`. This does not relax checks on purely integer inputs.
- Zero-prior rows are ignored and may be all zero. Zero source columns are allowed.
  Other rows must have positive mass; all entries must be finite and nonnegative.

For binary sources, mixing floating and integer channels uses approximate
collinearity. Binary targets validate integer channels individually, even in
mixed lists, then use floating-point arithmetic for all inputs.

These calls replace the earlier joint-table APIs. To convert normalized joint
tables, use `prior = joints[0].sum(axis=1)` and divide each positive-prior row by
its prior probability. General joint counts are not integer conditional weights;
do not round probabilities or cast oversized weights to `uint32`.

## Outputs

Both solvers return a `RedundancyResult`. **The two output flags are independent
and default to `False`.** Unrequested outputs are `None`.

| Available when | Fields |
| --- | --- |
| Always | `redundancy_nats`, `redundancy_bits`, `target_prior` |
| `return_channel=True` | `channel` = `P(Q | Y)`, `posteriors` = `P(Y | Q)`, `posterior_weights` = `P(Q)`, `target_auxiliary_joint` = `P(Y,Q)` |
| `return_garblings=True` | `garblings` = tuple of `P(Q | X_i)` matrices, `max_garbling_residual` |

Channels have shape `(d, q)` and garblings `(m_i, q)`, where `q` is the number
of output states. They satisfy `P(Y,X_i) @ K_i = P(Y,Q)` to numerical precision;
`max_garbling_residual` is the largest absolute entrywise error in this identity.
Binary-target garblings are SciPy CSR matrices (`K_i.toarray()` makes them dense);
binary-source garblings are NumPy arrays.

Optional binary-target garblings use `O(N + k q)` additional work and sparse
storage after sorting, giving `O(N log N + k q)` total time. Since `q <= N`,
their worst-case reconstruction cost is `O(k N)`; dense conversion costs `O(N q)`.
SciPy is needed only for binary-target garblings. Reconstruction and residual
checks are skipped unless requested.

On zero-prior target rows, `channel` uses `P(Q)` as an arbitrary stochastic
extension. The legacy `input_adjustment` field remains zero: the prior is shared
and weights are normalized by definition.

## Algorithms and numerical behavior

**Binary target.** Sort the source posterior probabilities `P(Y=1 | X_i)`,
construct the lower convex hull of their call-function knots, and recover the
meet's posterior law from slope jumps. This implements the classical binary
Blackwell meet of Bertschinger and Rauh (2014), Section 4. Optional sparse
garblings use the inverse-transform martingale coupling of Jourdain and
Margheriti (2020), without linear programming. A meet can have more than two
outputs even when one source is binary.

Binary-target redundancy is continuous (Kolchinsky, 2022, Section 5.5 and
Appendix D), so exact arithmetic and a collinearity tolerance are unnecessary.
The hull uses `np.longdouble` and discards slope jumps at roundoff scale;
extremely small atoms may be lost. `atol` controls numerical consistency
checks, not a bound on redundancy error. Bounded batches retain only their
lower hulls for the final hull, avoiding large temporary matrices. Normalized
joint tables are retained only when garblings are requested.

**Binary sources.** Intersect the posterior segments through the prior. An
uninformative source or segments on distinct lines give zero redundancy;
otherwise the intersection's endpoints define the meet. Integer weights permit
exact collinearity and endpoint comparisons: signed `int64` differences and
`uint64` products bounded by `(2**32 - 1)**2`, without GCDs or arbitrary-precision
integers. Probabilities and information are evaluated in floating point.

With more than two target states, arbitrarily small perturbations can rotate
segments onto different lines and change positive redundancy to zero. This is
why floating channels require an explicit `tolerance`. It compares anchored
channel differences scaled to maximum absolute coordinate one, allowing sign
reversal. Neither this tolerance nor a small garbling residual bounds the error
in redundancy; `tolerance=0` still uses floating-point arithmetic.

**Acceleration.** Optional Numba compiles the scans at import, excluding
compilation from solver calls. Without it, the same bare Python loops run and
one warning recommends `discrete-pid[speed]`. Integer row-sum validation stops
at the first mismatch without full source-by-target temporary arrays. On
platforms where `longdouble` is wider than `float64`, the hull and martingale
coupling retain Python scans to preserve precision; Numba accelerates those
scans when the precisions agree, including Apple Silicon.

## Tests

```bash
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

Tests cover analytical examples, independent LP and rational posterior oracles,
input bounds, sparse/dense garbling identities, optional outputs, and compiled/
Python agreement. GitHub Actions checks Python 3.10, 3.12, and 3.13, plus Numba.

## References and authorship

- Artemy Kolchinsky, [*A Novel Approach to the Partial Information
  Decomposition*](https://doi.org/10.3390/e24030403), Entropy **24**, 403 (2022).
  Defines Blackwell redundancy and its common-garbling optimization.
- Nils Bertschinger and Johannes Rauh, [*The Blackwell relation defines no
  lattice*](https://arxiv.org/abs/1401.3146), 2014. Their channel *input* is the
  variable called the *target* here.
- Benjamin Jourdain and William Margheriti, [*A new family of one dimensional
  martingale couplings*](https://doi.org/10.1214/20-EJP543), 2020.

The code, tests, and documentation were written by **OpenAI Codex**, at Artemy
Kolchinsky's request. The binary-target solver was adapted from the existing
implementation in the BlackwellPID-Discrete research project. Mathematical
sources are credited above.
