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

The package is installed from this repository; no PyPI release is assumed.

## Quick example

Each input is a joint probability table `P(Y, X_i)`: **rows are target states,
columns are source states**. Every table sums to one, and all row sums must
give the same target marginal. Source–target pairwise tables suffice; a full
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
# be binary. These sources observe a three-state target.
prior = np.ones(3) / 3
direction = np.array([0.1, -0.1, 0.0])
joints = []
for lower, upper in [(-2.0, 1.0), (-1.0, 2.0)]:
    weights = np.array([upper, -lower]) / (upper - lower)
    posteriors = prior[:, None] + direction[:, None] * [lower, upper]
    joints.append(posteriors * weights)

result = redundancy_binary_sources(joints)
print(f"Redundancy: {result.redundancy_bits:.9f} bits")
for joint, kernel in zip(joints, result.garblings):
    assert np.allclose(joint @ kernel, result.target_auxiliary_joint)
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
  on rows and auxiliary states on columns, satisfying `joint_i @ K_i = P(Y,Q)`
  to numerical precision. Auxiliary labels are arbitrary.
- `max_garbling_residual`: the largest absolute entrywise error in these
  equalities, or `None` when garblings were not requested.
- `input_adjustment`: the largest absolute entrywise change made during
  within-tolerance input normalization.

The binary-source function always returns garblings. For the binary-target
function, use `return_garblings=True` and install `.[garblings]` or `.[test]`.
This optional reconstruction solves linear programs and is **outside** the
`O(N log N)` bound. Both fast algorithms run without SciPy by default.

## Numerical behavior

The mathematical algorithms are exact constructions. These implementations
use floating-point arithmetic, not symbolic or interval arithmetic. They do
not enumerate polytope vertices or discretize a posterior grid.

- Inputs must be finite, nonnegative, nonempty, and normalized. Discrepancies
  in normalization or common target marginals within `atol` (default `1e-12`)
  are reconciled to the first table by row scaling. The result describes these
  processed tables. Input arrays are not modified.
- Zero-probability source columns are allowed. Zero-probability target rows
  must agree across sources. In the binary-target function, tables still
  have exactly two rows; the binary-source function allows any positive row
  count and at most two positive-mass source columns.
- The hull calculation uses NumPy's extended-precision type where available,
  whose precision is platform-dependent. Slope jumps at machine-roundoff
  scale are discarded; extremely small atoms may therefore be lost.
- For binary sources, `geometry_tol` (default `1e-12`) compares normalized
  posterior directions. Blackwell redundancy can be discontinuous at exact
  collinearity when the target has more than two states. Nearly aligned but
  distinct lines may be classified as aligned within this tolerance, yielding
  a positive value where the exact value is zero. A small garbling residual
  does **not** certify closeness to the exact redundancy in that case.
  `geometry_tol=0` requires equality of the floating-point directions but is
  not a substitute for exact arithmetic.

## Tests

```bash
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

Tests cover analytical examples, an independent LP check for random binary
targets, agreement of the algorithms on their shared domain, garbling
feasibility, permutations, degenerate variables, zero-probability states,
near-collinearity, and invalid inputs. GitHub Actions runs the tests and the
example script on Python 3.10, 3.12, and 3.13.

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
