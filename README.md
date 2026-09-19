# discrete-pid

Fast Python implementations of two special cases of **Blackwell redundancy**
for discrete random variables. Given a target `Y` and sources `X_1, ..., X_k`,
Blackwell redundancy maximizes `I(Y; Q)` over experiments `Q` obtainable by
stochastic processing of **every** source. Each source can use a different
processing kernel, but they must produce the same joint distribution `P(Y, Q)`.

| Function | Applicable inputs | Arithmetic cost |
| --- | --- | --- |
| `redundancy_binary_target` | Binary target; any number of sources with arbitrary finite alphabets | `O(N log N)` time and `O(N)` working memory |
| `redundancy_binary_sources` | Arbitrary finite target; every source has at most two active states | `O(k d)` time for binary input tables |

Here `N` is the total number of source states, `k` is the number of sources,
and `d` is the number of target states. These are arithmetic-operation bounds;
logarithm evaluation and arbitrary-precision bit costs are separate.
If tables contain padded zero columns, reading them adds their input size.
These functions compute redundancy and an optimal common experiment, rather
than all atoms of a partial information decomposition.

## Install

Requires Python 3.10 or newer. NumPy is the only required runtime dependency.
From a checkout:

```bash
git clone https://github.com/artemyk/discrete-pid.git
cd discrete-pid
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For optional binary-target garbling reconstruction or the test suite, install
SciPy as well:

```bash
python -m pip install ".[test]"
```

Optional Numba acceleration is available with:

```bash
python -m pip install ".[speed]"
```

The package is installed from this repository; no PyPI release is assumed.

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
```

A runnable version of both examples is included:

```bash
python examples/quickstart.py
```

## Algorithms and outputs

**Binary target.** Represent a source by its posterior probabilities
`r = P(Y=1 | X_i)` and call function `C_i(t) = E[max(r - t, 0)]`.
The meet is the greatest convex function below all source call functions.
The implementation sorts their knots, constructs one lower convex hull, and
recovers the optimal posterior probabilities and weights from its slope jumps.
An optimal `Q` can have more than two outputs even when one source is binary.

This is a posterior-coordinate implementation of the classical binary
Blackwell meet construction described by
[Bertschinger and Rauh (2014), Section 4](https://arxiv.org/abs/1401.3146).
The underlying meet construction is not claimed as a new result here.
In that reference the channel *input* is the variable called the *target* here.

**All binary sources.** Each source has a posterior segment containing the
prior. If two informative segments lie on different lines, their intersection
is only the prior and redundancy is zero. Otherwise, intersect their scalar
intervals. The endpoints and their unique mean-preserving weights define the
meet. Source-to-meet garbling matrices are constructed directly. An
uninformative source also forces zero redundancy.

Both functions return a `RedundancyResult` with:

- `redundancy_nats` and `redundancy_bits`.
- `target_prior`: `P(Y)`.
- `posteriors`: a `(d, q)` array whose columns are `P(Y | Q)`.
- `posterior_weights`: the vector `P(Q)`.
- `target_auxiliary_joint`: the `(d, q)` table `P(Y, Q)`.
- `garblings`: when available, matrices `K_i = P(Q | X_i)` with source states
  on rows and auxiliary states on columns, satisfying `P(Y,X_i) @ K_i = P(Y,Q)`
  to numerical precision, with normalized probabilities in this equality.
  Auxiliary labels are arbitrary.
- `max_garbling_residual`: the largest absolute entrywise error in these
  equalities, or `None` when garblings were not requested.
- `input_adjustment`: the largest absolute entrywise change made during
  within-tolerance normalization or marginal reconciliation. For binary
  sources, this is measured after converting weights to probabilities;
  it is zero in integer mode.

The binary-source function always returns garblings. For the binary-target
function, use `return_garblings=True` and install `.[garblings]` or `.[test]`.
This optional reconstruction solves linear programs and is **outside** the
`O(N log N)` bound. Both fast algorithms run without SciPy by default.

## Performance

Equal-shaped source tables are validated and processed in NumPy batches.
The binary-target solver also sorts knots with NumPy and, when Numba is
installed, compiles the hull scan on platforms where `np.longdouble` has the
same precision as `float64` (including Apple Silicon). The first call incurs
compilation or cache-loading overhead; later calls reuse the compiled code.
No `fastmath` or parallel reductions are used. On platforms with wider
`longdouble`, the hull retains that precision and uses the Python scan;
[Numba does not support extended-precision NumPy floats](https://numba.readthedocs.io/en/stable/reference/numpysupported.html#scalar-types).
The package also works without Numba.

Binary-source geometry remains in arbitrary-precision integer/rational
arithmetic. Output conversion and kernel construction avoid unnecessary
fraction reductions, and garbling residuals are checked in NumPy batches
when shapes agree. These optimizations preserve the default exact
collinearity checks and the explicit opt-in for floating-point inputs.

## Numerical behavior

For binary sources, collinearity is sensitive: when the target has more than
two states, an arbitrarily small perturbation can move posterior segments
onto different lines and change positive redundancy to zero. Therefore
`redundancy_binary_sources(joints, tolerance=None)` requires integer inputs.
Use original integer counts or exact integer weights; **do not round or cast
floating-point probabilities to integers** to bypass this check.

- Integer tables may have different positive totals, but their normalized
  target marginals must agree exactly. Collinearity is checked by integer
  cross-products, and segment endpoints, weights, and garblings are computed
  with rational arithmetic. Python arbitrary-size integers prevent overflow;
  NumPy integer arrays and nested lists of integers are accepted. Supplying a
  numerical `tolerance` does not weaken exact checks on integer-only inputs.
- Floating-point tables, including integer-valued float arrays or a mixture
  of integer and float tables, raise an informative error by default. To opt
  into approximate geometry, explicitly pass a finite, nonnegative tolerance:

  ```python
  approximate = redundancy_binary_sources(
      [joint.astype(float) for joint in counts], tolerance=1e-12
  )
  ```

  `tolerance` compares posterior directions scaled to maximum absolute
  coordinate one, allowing a sign reversal. It replaces `geometry_tol` from
  the initial version. Nearly aligned but distinct lines may then be treated
  as collinear, yielding a positive value where the exact value is zero.
  Neither this tolerance nor a small `max_garbling_residual` bounds the error
  in redundancy. `tolerance=0` still uses floating-point arithmetic.
- In floating-point mode, binary-source tables may contain normalized
  probabilities or unnormalized weights. After normalization, target-marginal
  discrepancies within `atol` (default `1e-12`) are reconciled by row scaling.
  `atol` does not control collinearity or permit float inputs by itself.
- The binary-target solver continues to use floating-point arithmetic and
  requires tables summing to one within `atol`; normalization and marginal
  discrepancies within `atol` are reconciled. Its hull calculation uses
  NumPy's platform-dependent extended precision; slope jumps at machine
  roundoff scale are discarded, so extremely small atoms may be lost.
- Both functions return floating-point probabilities and mutual information,
  even when the geometry is computed exactly. In integer mode, a positive
  probability too small to represent in the output raises `ArithmeticError`.
  Inputs must be finite, nonnegative, nonempty, and have positive total mass.
  Zero source columns are allowed; zero target rows must agree across sources.
  Binary-target tables have exactly two rows, while binary-source tables have
  any positive row count and at most two positive-mass columns. Input arrays
  are not modified.

## Tests

```bash
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

Tests cover analytical examples, an independent LP check for random binary
targets, agreement of the algorithms on their shared domain, garbling
feasibility, permutations, degenerate variables, zero-probability states,
exact integer collinearity below floating-point resolution, large integer
counts, explicit floating-point tolerances, and invalid inputs. GitHub Actions
runs the tests and the example script on Python 3.10, 3.12, and 3.13.

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
research project. Mathematical ideas and prior constructions are credited
above; code authorship does not imply mathematical originality.
