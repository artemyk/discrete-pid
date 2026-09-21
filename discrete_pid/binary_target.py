"""Binary-target Blackwell meet via a lower convex hull of call-function knots.

This implements the classical binary-experiment meet in posterior coordinates;
see Bertschinger and Rauh (2014), Section 4, https://arxiv.org/abs/1401.3146.
The hull takes O(N log N) operations and O(N) memory. Optional sparse source
garblings use the inverse-transform martingale coupling of Jourdain and
Margheriti (2020), https://arxiv.org/abs/1808.01390. Given sorted posterior
laws, all couplings take O(N+k*q) additional operations and storage.

Each bounded batch uses one global posterior order and separate source tail
sums, retaining only its lower hull. uint32 counts bypass joint normalization.
"""

from __future__ import annotations

from numbers import Real
from typing import Iterable, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._common import Array, RedundancyResult, make_result, normalize_prior
from ._hull import lower_hull
from ._validation import integer_weights, normalize_rows, validate_row_sums
from ._martingale import inverse_transform, sparse_rows
from ._target_scan import call_knots, count_posteriors, source_orders as recover_source_orders

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


# Bound the arrays used for normalization, sorting, and call knots. Only
# each batch's lower hull is retained for the final hull of their union.
_BATCH_ENTRIES = 65536


def _channel_batches(channels, prior):
    if isinstance(channels, np.ndarray):
        if channels.ndim != 3 or channels.shape[1] != 2 or channels.shape[2] == 0:
            raise ValueError("channels must have shape (number of sources, 2, source states)")
        step = max(1, _BATCH_ENTRIES // (2 * channels.shape[2]))
        for start in range(0, len(channels), step):
            yield _prepare_channels(channels[start:start + step], prior)
        return
    pending, entries, key = [], 0, None
    for channel in channels:
        raw = np.asarray(channel)
        if raw.ndim != 2 or raw.shape[0] != 2 or raw.shape[1] == 0:
            raise ValueError("each channel must have shape (2, source states)")
        current = (raw.shape, raw.dtype)
        if pending and (current != key or entries + raw.size > _BATCH_ENTRIES):
            yield _prepare_channels(np.asarray(pending), prior)
            pending, entries = [], 0
        pending.append(raw)
        entries += raw.size
        key = current
    if pending:
        yield _prepare_channels(np.asarray(pending), prior)


def _prepare_channels(raw, prior):
    # Full-support uint32 counts are validated during posterior calculation.
    # A tiny positive p0 can coexist with p1 rounded to one: keep validation
    # on the normalization path whenever the hull takes its degenerate branch.
    if raw.dtype == np.uint32 and np.all(prior > 0) and 0. < prior[1] < 1.:
        return raw
    return _normalize_channels(raw, prior)


def _normalize_channels(raw, prior):
    active = prior > 0
    if integer_weights(raw):
        validate_row_sums(raw, active)
    weights = normalize_rows(raw, active, dtype=np.longdouble)
    weights *= prior[None, :, None]
    return weights


def _posterior_order(locations):
    # Coarse integer keys accelerate the sort; refinement retains the full
    # floating-point order, including ties, signed zero, and wider longdouble.
    coarse = (locations * 65535).astype(np.uint16)
    order = np.argsort(coarse, kind="stable")
    return order[np.argsort(locations[order], kind="stable")]


def _envelope(locations, heights, *, sorted=False):
    if not sorted:
        order = _posterior_order(locations)
        locations, heights = locations[order], heights[order]
    starts = np.flatnonzero(np.concatenate(([True], locations[1:] != locations[:-1])))
    # Only the lowest knot at each location can belong to the lower hull.
    locations, heights = locations[starts], np.minimum.reduceat(heights, starts)
    hull = lower_hull(locations, heights)
    return locations[hull], heights[hull]


def _posterior_meet(batches, prior, atol, tables, source_orders=None):
    extended = np.longdouble
    prior = np.asarray(prior, dtype=extended)
    p = prior[1]
    pieces = []
    seen = False
    for batch in batches:
        seen = True
        raw = batch.dtype == np.uint32
        if tables is not None:
            joint = _normalize_channels(batch, prior) if raw else batch
            tables.extend(np.asarray(joint, dtype=float))
        if p == 0.0 or p == 1.0:
            if source_orders is not None:
                source_orders.extend([None] * len(batch))
            continue  # Still validate every input, even for a constant target.
        if raw:
            theta, totals = count_posteriors(batch, prior)
            multipliers = prior
        else:
            weights = batch.sum(axis=1)
            theta = np.divide(batch[:, 1], weights, out=np.zeros_like(weights),
                              where=weights > 0).ravel()
            totals = np.ones(len(batch), dtype=np.uint64)
            multipliers = np.ones(2, dtype=extended)
        order = _posterior_order(theta)
        if source_orders is not None:
            source_orders.extend(recover_source_orders(order, len(batch), batch.shape[2]))
        locations, heights = call_knots(batch, theta, order, totals, multipliers)
        heights[0] = p
        # A point above a batch's lower hull cannot affect the global lower
        # hull. This reduction is optional mathematically, and saves memory.
        pieces.append(_envelope(locations, heights, sorted=True))
    if not seen:
        raise ValueError("at least one source channel is required")
    if p == 0.0 or p == 1.0:
        return np.array([p]), np.ones(1)
    if len(pieces) == 1:
        locations, heights = pieces[0]
    else:
        locations, heights = _envelope(np.concatenate([a for a, _ in pieces]),
                                       np.concatenate([b for _, b in pieces]))
    slopes = np.diff(heights) / np.diff(locations)
    # Short edges can amplify roundoff. The exact slopes are in [-1, 0].
    slopes = np.clip(slopes, -1, 0)
    masses = np.diff(np.concatenate(([extended(-1)], slopes, [extended(0)])))
    if np.min(masses) < -atol:
        raise ArithmeticError("numerical error produced an invalid posterior law")
    masses = np.maximum(masses, 0)
    positive = masses > 8 * np.finfo(extended).eps
    support = np.asarray(locations[positive], dtype=float)
    weights = np.asarray(masses[positive], dtype=float)
    weights /= weights.sum()
    if abs(float(support @ weights) - p) > 10 * atol:
        raise ArithmeticError("numerical error changed the optimal posterior mean")
    return support, weights


def _garbling(joint: Array, target_joint: Array, weights: Array, atol: float,
              *, support: Array | None = None,
              source_order: Array | None = None) -> csr_matrix:
    """Construct P(Q|X) by a sparse inverse-transform martingale coupling.

    Sort posterior laws, merge quantile cells, and exchange mass between
    matched moment deficits/excesses. The source-conditional kernel follows
    by dividing coupling rows by their mass. Zero-probability source states
    are assigned deterministically to the first output. Sparse construction
    and validation take O(m+q) work after sorting, without densifying K.
    """
    try:
        from scipy.sparse import csr_matrix
    except ImportError as exc:
        raise ImportError(
            "Sparse binary-target garblings need SciPy; install discrete-pid[garblings]"
        ) from exc
    m, q = joint.shape[1], len(weights)
    if q == 1:
        return csr_matrix((np.ones(m), np.zeros(m, dtype=np.int64),
                           np.arange(m + 1)), shape=(m, 1))
    extended = np.longdouble
    joint_ext = np.asarray(joint, dtype=extended)
    marginal = joint_ext.sum(axis=0)
    active = (np.flatnonzero(marginal > 0) if source_order is None
              else source_order[marginal[source_order] > 0])
    theta = joint_ext[1, active] / marginal[active]
    if source_order is None:
        order = np.argsort(theta, kind="stable")
        theta, active = theta[order], active[order]
    positive = np.flatnonzero(weights > 0)
    if support is None:
        support = (np.asarray(target_joint[1, positive], dtype=extended)
                   / np.asarray(weights[positive], dtype=extended))
        output_order = np.argsort(support, kind="stable")
        support, positive = support[output_order], positive[output_order]
    else:
        # The hull already supplies sorted output posteriors. Reuse them to
        # avoid an O(q log q) sort for every source (and a round-trip division).
        support = np.asarray(support[positive], dtype=extended)
    rows, cols, data = inverse_transform(
        np.ascontiguousarray(theta), np.ascontiguousarray(marginal[active]),
        np.ascontiguousarray(support),
        np.ascontiguousarray(weights[positive], dtype=extended))
    rows, cols = active[rows], positive[cols]
    data = np.asarray(data / marginal[rows], dtype=float)
    missing = np.flatnonzero(marginal == 0)
    rows = np.concatenate((rows, missing))
    cols = np.concatenate((cols, np.zeros(len(missing), dtype=np.int64)))
    data = np.concatenate((data, np.ones(len(missing))))
    kernel = csr_matrix(sparse_rows(rows, cols, data, m), shape=(m, q))
    row_sums = np.asarray(kernel.sum(axis=1)).ravel()
    if not np.all(np.isfinite(kernel.data)) or np.any(row_sums <= 0):
        raise ArithmeticError("garbling reconstruction produced an invalid row")
    # Normalize in O(nnz) without constructing an m-by-q broadcast array.
    kernel.data /= np.repeat(row_sums, np.diff(kernel.indptr))
    error = float(np.max(np.abs(joint @ kernel - target_joint)))
    if error > max(100 * atol, 1e-8):
        raise ArithmeticError(f"garbling reconstruction residual is too large: {error}")
    return kernel


def redundancy_binary_target(
    prior: ArrayLike,
    channels: Iterable[ArrayLike],
    *,
    return_channel: bool = False,
    return_garblings: bool = False,
    atol: float = 1e-12,
) -> RedundancyResult:
    """Compute redundancy from a binary-target prior and conditional channels.

    ``prior`` contains two nonnegative target weights, normalized internally.
    Each channel has shape (2,m_i), with target states as rows. A packed
    (k,2,m) array, a list of differently sized channels, or an iterator works.

    Integer channel weights must lie in [0,2**32-1], with the same positive
    row sum within each source on the positive-prior support, as in the
    binary-source solver. NumPy uint32 is recommended for compact storage.
    Floating-point channels are accepted without a collinearity tolerance;
    their rows may contain unnormalized weights and are normalized separately.
    All entries must be finite and nonnegative. Zero-prior rows may be zero,
    and zero source columns are allowed. Caller inputs are never modified.

    Binary-target redundancy is continuous, so no exact collinearity decision
    is needed. Both integer and floating inputs use floating-point hull
    arithmetic, with extended precision where available. ``atol`` controls
    numerical consistency checks (default 1e-12), not collinearity or an
    error bound on redundancy. Very small posterior atoms may be lost to
    roundoff. Each batch uses one global posterior order and a scan with
    separate source tail sums. uint32 counts avoid normalized joint arrays
    unless garblings are requested. Bounded batches retain only their
    lower hulls, combined in O(N log N) time and O(N) worst-case memory.

    Set return_channel=True to return P(Q|Y) in result.channel, together with
    posteriors and posterior weights. Independently, return_garblings=True
    reconstructs P(Q|X_i) as SciPy CSR matrices, using the inverse-transform
    martingale coupling (no linear programming). Both flags default to False;
    omitted outputs are None. Only garbling reconstruction retains all joint
    tables. Coupling construction takes O(N+k*q) additional work/storage once
    posterior laws are sorted, where q is the number of output states. Since
    q<=N, the additional worst-case bound is O(k*N); dense conversion costs
    O(N*q). SciPy is needed only when sparse garblings are requested.
    """
    if (not isinstance(atol, Real) or isinstance(atol, (bool, np.bool_))
            or not np.isfinite(atol) or atol <= 0):
        raise ValueError("atol must be finite and positive")
    prior = normalize_prior(prior)
    if len(prior) != 2:
        raise ValueError("a binary target requires a prior with two entries")
    tables = [] if return_garblings else None
    source_orders = [] if return_garblings else None
    support, weights = _posterior_meet(_channel_batches(channels, prior),
                                       prior, atol, tables, source_orders)
    posteriors = np.vstack((1 - support, support))
    garblings = None
    if return_garblings:
        target_joint = posteriors * weights
        garblings = tuple(_garbling(table, target_joint, weights, atol,
                                    support=support, source_order=order)
                          for table, order in zip(tables, source_orders))
    return make_result(prior, posteriors, weights, tables=tables, garblings=garblings,
                       return_channel=return_channel)
