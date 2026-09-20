"""Binary-target Blackwell meet via a lower convex hull of call-function knots.

This implements the classical binary-experiment meet in posterior coordinates;
see Bertschinger and Rauh (2014), Section 4, https://arxiv.org/abs/1401.3146.
The hull takes O(N log N) operations and O(N) memory. Optional sparse source
garblings use the inverse-transform martingale coupling of Jourdain and
Margheriti (2020), https://arxiv.org/abs/1808.01390. Given sorted posterior
laws, all couplings take O(N+k*q) additional operations and storage.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Iterable, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._common import Array, RedundancyResult, make_result, normalize_prior
from ._hull import lower_hull
from ._validation import validate_row_sums
from ._martingale import inverse_transform, sparse_rows

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


# Bound the arrays used for normalization, sorting, and suffix sums. Only
# each batch's lower hull is retained for the final hull of their union.
_BATCH_ENTRIES = 262144


def _channel_batches(channels, prior):
    if isinstance(channels, np.ndarray):
        if channels.ndim != 3 or channels.shape[1] != 2 or channels.shape[2] == 0:
            raise ValueError("channels must have shape (number of sources, 2, source states)")
        step = max(1, _BATCH_ENTRIES // (2 * channels.shape[2]))
        for start in range(0, len(channels), step):
            yield _normalize_channels(channels[start:start + step], prior)
        return
    pending, entries, key = [], 0, None
    for channel in channels:
        raw = np.asarray(channel)
        if raw.ndim != 2 or raw.shape[0] != 2 or raw.shape[1] == 0:
            raise ValueError("each channel must have shape (2, source states)")
        current = (raw.shape, raw.dtype)
        if pending and (current != key or entries + raw.size > _BATCH_ENTRIES):
            yield _normalize_channels(np.asarray(pending), prior)
            pending, entries = [], 0
        pending.append(raw)
        entries += raw.size
        key = current
    if pending:
        yield _normalize_channels(np.asarray(pending), prior)


def _normalize_channels(raw, prior):
    integer = raw.dtype.kind in 'iu'
    if raw.dtype.kind == 'O':
        if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) for v in raw.flat):
            raise ValueError("channels must contain real integer or floating-point weights")
        integer = all(isinstance(v, Integral) for v in raw.flat)
    elif raw.dtype.kind not in 'iuf':
        raise ValueError("channels must contain real integer or floating-point weights")
    active = prior > 0
    if integer:
        if np.any(raw < 0) or np.any(raw > np.iinfo(np.uint32).max):
            raise ValueError("integer channel weights must be in [0, 2**32-1] (uint32 range)")
        validate_row_sums(raw, active)
    try:
        weights = np.array(raw, dtype=np.longdouble, copy=True)
    except (OverflowError, ValueError) as error:
        raise ValueError("channel weights must be finite and nonnegative") from error
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("channel weights must be finite and nonnegative")
    scales = weights.max(axis=2)
    if np.any(scales[:, active] == 0):
        raise ValueError("each channel row on the positive-prior support needs positive mass")
    weights /= np.where(scales > 0, scales, 1)[:, :, None]
    totals = weights.sum(axis=2)
    weights /= np.where(totals > 0, totals, 1)[:, :, None]
    weights *= prior[None, :, None]
    return weights


def _envelope(locations, heights):
    order = np.lexsort((heights, locations))
    locations, heights = locations[order], heights[order]
    unique = np.concatenate(([True], locations[1:] != locations[:-1]))
    locations, heights = locations[unique], heights[unique]
    hull = lower_hull(locations, heights)
    return locations[hull], heights[hull]


def _posterior_meet(batches, p, atol, tables, source_orders=None):
    extended = np.longdouble
    pieces = []
    seen = False
    for batch in batches:
        seen = True
        if tables is not None:
            tables.extend(np.asarray(batch, dtype=float))
        if p == 0.0 or p == 1.0:
            if source_orders is not None:
                source_orders.extend([None] * len(batch))
            continue  # Still validate every input, even for a constant target.
        weights = batch.sum(axis=1)
        theta = np.zeros_like(weights)
        np.divide(batch[:, 1], weights, out=theta, where=weights > 0)
        order = np.argsort(theta, axis=1, kind="stable")
        if source_orders is not None:
            source_orders.extend(order)
        theta = np.take_along_axis(theta, order, axis=1)
        weights = np.take_along_axis(weights, order, axis=1)
        tail_mass = np.cumsum(weights[:, ::-1], axis=1, dtype=extended)[:, ::-1]
        tail_moment = np.cumsum((weights * theta)[:, ::-1], axis=1, dtype=extended)[:, ::-1]
        calls = np.maximum(tail_moment - theta * tail_mass, 0)
        interior = (theta > 0) & (theta < 1) & (weights > 0)
        locations = np.concatenate(([extended(0), extended(1)], theta[interior]))
        heights = np.concatenate(([extended(p), extended(0)], calls[interior]))
        # A point above a batch's lower hull cannot affect the global lower
        # hull. This reduction is optional mathematically, and saves memory.
        pieces.append(_envelope(locations, heights))
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
    roundoff. Bounded batches avoid full-size temporary matrices; their
    lower hulls are combined in O(N log N) time and O(N) worst-case memory.

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
                                       float(prior[1]), atol, tables, source_orders)
    posteriors = np.vstack((1 - support, support))
    garblings = None
    if return_garblings:
        target_joint = posteriors * weights
        garblings = tuple(_garbling(table, target_joint, weights, atol,
                                    support=support, source_order=order)
                          for table, order in zip(tables, source_orders))
    return make_result(tables, posteriors, weights, 0.0, garblings,
                       return_channel=return_channel, prior=prior)
