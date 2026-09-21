# discrete-pid

Python implementations of **Blackwell redundancy and union information** for
discrete variables. Redundancy maximizes `I(Y; Q)` over channels obtainable by
stochastic processing of every source; union minimizes it over channels from
which every source can be decoded. The package optionally returns these
channels and their garblings. It does not compute all PID atoms.

Union also gives a synergy measure: subtract union information from the mutual
information of the observed source tuple and target. Only that total mutual
information needs the actual joint law; the solvers use source–target marginals.

| Function | Applicable inputs | Time / working memory |
| --- | --- | --- |
| `redundancy_binary_target` | Binary target; arbitrary finite sources | `O(N log N)` / `O(N)` |
| `redundancy_binary_sources` | Arbitrary finite target; binary sources | `O(k d)` / `O(k d)` |
| `union_binary_target` | Binary target; arbitrary finite sources | `O(N log N)` / `O(N)` |
| `union_binary_sources` | Arbitrary finite target; binary sources | Adaptive convex optimization; depends on target size and numerical tolerance |

Here `k` is the number of sources, `d` the number of target states, and `N` the
total number of source states. The geometric operation bounds exclude optional output construction,
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

Optional extras provide Numba acceleration, SciPy sparse garblings, and
SciPy/Clarabel for the binary-source union optimization:

```bash
python -m pip install ".[speed,garblings,union]"  # Includes everything used below
```

## Quick example

All four solvers take a target prior and conditional channels `P(X_i | Y)`.
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

## Union information

```python
from discrete_pid import union_binary_target, union_binary_sources

joined = union_binary_target([1, 1], [bsc, bec],
                             return_channel=True, return_garblings=True)
print(joined.union_bits)              # 0.639035953
print(joined.garblings[0].toarray())   # P(X_1 | Q), shape (q, 2)

# Binary-source union accepts floating weights without a collinearity tolerance.
union = union_binary_sources(prior, channels / 30, tolerance=1e-7)
print(union.union_bits, union.gap_nats)
```

A binary target always admits a **Blackwell join**, obtained from the upper
envelope of the source call functions. Its information is the union. The join
has at most `N-k+1` outputs; new posterior values can occur where curves cross.
This geometric solver requires only NumPy, plus optional Numba acceleration and
SciPy if decoding channels are requested.

Binary sources with a larger target need not have a join. Their union still
exists and is computed by a convex minimum-information (MinMI) optimization.
The solver starts with a feasible common-uniform coupling, solves a restricted
relative-entropy problem with Clarabel, and adds binary decoder patterns using
a global pricing oracle. It does not discretize target posteriors. Binary
targets and at most two retained sources use specialized routines.

When the active pattern set grows, a linear program reweights its fixed
posteriors to retain a smaller feasible support before the next conic solve.
Compression checks the original moments and information value; subsequent
masters reoptimize the posteriors. Numerical failure or repeated supports
restore all generated patterns and disable compression. Global pricing and
the requested stopping accuracy are unchanged.

For up to three active target states, global pricing uses the binary-pattern
geometry (or enumeration for small retained source sets). Larger targets use
a branch-and-bound search with an explicit work limit. Reaching that limit
raises an error rather than returning an unverified optimization gap.

The numerical stopping tolerance is an absolute primal/global-dual gap in
**nats**, with a minimum supported tolerance of `1e-10`. Returned bounds and feasibility diagnostics are floating-point
estimates, not interval certificates. Requested accuracy that cannot be reached
raises an error. The adaptive solver has no polynomial iteration guarantee;
larger target alphabets can make global pricing expensive. The fixed-target
ellipsoid complexity argument in the manuscript describes a different algorithm.

## Input format

```python
redundancy_binary_target(prior, channels, *,
                         return_channel=False, return_garblings=False, atol=1e-12)
redundancy_binary_sources(prior, channels, *, tolerance=None,
                          return_channel=False, return_garblings=False)
union_binary_target(prior, channels, *,
                    return_channel=False, return_garblings=False, atol=1e-12)
union_binary_sources(prior, channels, *, tolerance=1e-7,
                     return_channel=False, return_garblings=False)
```

- `prior`: nonnegative target weights with positive total, normalized internally;
  floats are allowed.
  It has two entries for binary targets, or `d` entries for binary sources.
- `channels`: an iterable of `(d, m_i)` arrays, or a packed `(k, d, m)` array
  when source sizes agree. Binary targets require `d=2`; binary sources require
  every `m_i=2`. Inputs are never modified.
- **Integer channels (all solvers):** unnormalized weights in `[0, 2**32 - 1]`;
  `np.uint32` is recommended.
  Every positive-prior row within a source must have the same positive sum `L_i`.
  The probabilities are `channels[i,y,x] / L_i`. Denominators may differ across
  sources and may exceed `uint32`. Other fitting integer dtypes/lists also work.
- **Floating channels:** finite nonnegative weights, normalized row by row.
  The binary-source **redundancy** solver requires an explicit finite, nonnegative
  collinearity `tolerance`; the other solvers do not. Union’s `tolerance` instead
  specifies its numerical optimization gap. Integer validation remains exact.
- Zero-prior rows are ignored and may be all zero. Zero source columns are allowed.
  Other rows must have positive mass; all entries must be finite and nonnegative.

For binary-source redundancy, mixing floating and integer channels uses approximate
collinearity. Binary targets validate integer channels individually, even in
mixed lists, then use floating-point arithmetic for all inputs.

These calls replace the earlier joint-table APIs. To convert normalized joint
tables, use `prior = joints[0].sum(axis=1)` and divide each positive-prior row by
its prior probability. General joint counts are not integer conditional weights;
do not round probabilities or cast oversized weights to `uint32`.

## Outputs

The redundancy solvers return a `RedundancyResult`. **The two output flags are independent
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

Union solvers return a `UnionResult` with `union_nats`, `union_bits`,
`target_prior`, `lower_bound_nats`, `upper_bound_nats`, `gap_nats`,
`feasibility_residual`, `iterations`, `method`, and `diagnostics`.
The optional channel/posterior fields use the same layout as redundancy and the
same independent output flags. Union `garblings[i]` has shape `(q, m_i)` and
means `P(X_i | Q)`: the decoding direction is reversed. Its residual measures
`P(Y,Q) @ garblings[i] - P(Y,X_i)` on the prior support. Binary-target decoders
are SciPy CSR matrices; binary-source decoders are NumPy arrays. A binary-target
join has `gap_nats=0`, reflecting an envelope construction rather than a
numerical optimization stopping test.

## Algorithms and numerical behavior

**Binary target.** Sort all posterior probabilities `P(Y=1 | X_i)` in a batch
together, then scan once with separate tail sums for each source. This produces
the call-function knots in order, ready for their lower convex hull. Retain each
batch hull, take the hull of their union, and recover the meet's posterior law
from slope jumps. uint32 inputs use the counts directly, with uint64 row-sum
validation and floating-point posterior/call calculations. This implements the classical binary
Blackwell meet of Bertschinger and Rauh (2014), Section 4. Optional sparse
garblings use the inverse-transform martingale coupling of Jourdain and
Margheriti (2020), without linear programming. A meet can have more than two
outputs even when one source is binary.

Binary-target redundancy is continuous (Kolchinsky, 2022, Section 5.5 and
Appendix D), so exact arithmetic and a collinearity tolerance are unnecessary.
The hull uses `np.longdouble` and discards slope jumps at roundoff scale;
extremely small atoms may be lost. `atol` controls numerical consistency
checks, not a bound on redundancy error. Sorting uses uint16 coarse keys followed
by refinement with the original coordinates. Bounded batches avoid large
temporary matrices; uint32 inputs need normalized joint tables only when
garblings are requested.

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
platforms where `longdouble` is wider than `float64`, the binary-target posterior,
call, hull and martingale scans run in Python to preserve precision; Numba accelerates those
scans when the precisions agree, including Apple Silicon.

## Tests

```bash
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

Tests cover analytical examples, independent LP and rational posterior oracles,
input bounds, sparse/dense garbling identities, optional outputs, and compiled/
Python agreement. GitHub Actions checks Python 3.10, 3.12, and 3.13, plus Numba.

## Union benchmarks

The benchmark harness records every timed call, its numerical gap and residual,
input hashes, package revision, dependency versions, and thread settings:

```bash
python benchmarks/benchmark_union.py --algorithm target --repeats 10 \
    --case-budget-seconds 180 --output benchmarks/results/union-target
python benchmarks/benchmark_union.py --algorithm sources --sizes 2 3 4 8 \
    --repeats 10 --case-budget-seconds 180 --output benchmarks/results/union-sources
```

Each case uses one worker and one native thread. Warmup, imports, compilation,
and input generation are excluded from timed calls. Output files include
`summary.csv`, `timings.csv`, `events.jsonl`, and metadata. Only cases completing
all repeats have a median. Timeouts and numerical failures are recorded, and
larger cases on the same curve are skipped by default. `--resume` continues
completed-case checkpoints for an unchanged implementation and configuration.

## References and authorship

- Artemy Kolchinsky, [*A Novel Approach to the Partial Information
  Decomposition*](https://doi.org/10.3390/e24030403), Entropy **24**, 403 (2022).
  Defines Blackwell redundancy, union information, and their optimization formulations.
- Nils Bertschinger and Johannes Rauh, [*The Blackwell relation defines no
  lattice*](https://arxiv.org/abs/1401.3146), 2014. Their channel *input* is the
  variable called the *target* here.
- Amir Globerson and Naftali Tishby, [*The Minimum Information Principle for
  Discriminative Learning*](https://people.csail.mit.edu/gamir/pubs/minmi_uai04.pdf),
  UAI 2004, pp. 193–200. The union solver specializes their MinMI primal, dual,
  and constraint-generation method using binary decoder patterns.
- Benjamin Jourdain and William Margheriti, [*A new family of one dimensional
  martingale couplings*](https://doi.org/10.1214/20-EJP543), 2020.

The code, tests, and documentation were written by **OpenAI Codex**, at Artemy
Kolchinsky's request. The binary-target solver was adapted from the existing
implementation in the BlackwellPID-Discrete research project. Mathematical
sources are credited above.
