"""Binary-target Blackwell union from the upper envelope of source calls.

The binary-experiment join is classical; see Bertschinger and Rauh (2014),
Proposition 16, https://arxiv.org/abs/1401.3146. In posterior coordinates its
call function is the pointwise maximum of the source call functions. Pooling
their affine pieces gives an O(N log N) upper-envelope algorithm, without
optimization or posterior discretization. Optional source decoders reverse
the direction of the sparse martingale construction used for redundancy.
"""

from __future__ import annotations

from numbers import Real
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from ._common import UnionResult, make_result, normalize_prior
from ._martingale import inverse_transform, sparse_rows
from ._target_scan import count_posteriors, source_orders as recover_source_orders
from ._union_target_scan import supporting_lines, upper_envelope
from .binary_target import _channel_batches, _normalize_channels, _posterior_order


def _line_envelope(slopes, intercepts):
    order = np.argsort(slopes, kind='stable')
    slopes, intercepts = slopes[order], intercepts[order]
    groups = np.flatnonzero(np.concatenate(([True], slopes[1:] != slopes[:-1])))
    slopes = slopes[groups]
    intercepts = np.maximum.reduceat(intercepts, groups)
    indices, starts = upper_envelope(slopes, intercepts)
    return slopes[indices], intercepts[indices], starts


def _posterior_join(batches, prior, atol, tables, source_orders):
    extended = np.longdouble
    prior = np.asarray(prior, dtype=extended)
    p = prior[1]
    pieces = []
    seen = False
    source_count = 0
    state_count = 0
    constant = np.count_nonzero(prior) == 1
    for batch in batches:
        seen = True
        source_count += len(batch)
        state_count += len(batch) * batch.shape[2]
        raw = batch.dtype == np.uint32
        if tables is not None:
            joint = _normalize_channels(batch, prior) if raw else batch
            tables.extend(np.asarray(joint, dtype=float))
        if constant:
            if source_orders is not None:
                source_orders.extend([None] * len(batch))
            continue  # The input batches still validate every source.
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
        slopes, intercepts = supporting_lines(batch, theta, order, totals, multipliers)
        slopes = np.concatenate((np.array([-1., 0.], dtype=extended), slopes))
        intercepts = np.concatenate((np.array([p, 0.], dtype=extended), intercepts))
        a, b, _ = _line_envelope(slopes, intercepts)
        # A line below a batch envelope cannot be needed in the final maximum.
        pieces.append((a, b))
    if not seen:
        raise ValueError('at least one source channel is required')
    if constant:
        return np.array([p], dtype=float), np.ones(1), source_count, state_count
    slopes, _, starts = _line_envelope(
        np.concatenate([a for a, _ in pieces]),
        np.concatenate([b for _, b in pieces]))
    support = starts[1:]
    weights = np.diff(slopes)
    if (np.any(~np.isfinite(support)) or np.any(support < -10 * atol)
            or np.any(support > 1 + 10 * atol) or np.any(weights <= 0)):
        raise ArithmeticError('numerical error produced an invalid join posterior law')
    support = np.asarray(np.clip(support, 0, 1), dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    # Clipping endpoint roundoff can create coincident atoms; merging is exact.
    groups = np.flatnonzero(np.concatenate(([True], support[1:] != support[:-1])))
    support, weights = support[groups], np.add.reduceat(weights, groups)
    if abs(float(support @ weights) - p) > 10 * atol:
        raise ArithmeticError('numerical error changed the join posterior mean')
    return support, weights, source_count, state_count


def _decoder(joint, target_joint, support, weights, atol, source_order):
    """Sparse P(X_i|Q), using the join as the more informative experiment."""
    try:
        from scipy.sparse import csr_matrix
    except ImportError as exc:
        raise ImportError(
            'Sparse binary-target decoders need SciPy; install discrete-pid[garblings]'
        ) from exc
    extended = np.longdouble
    marginal = np.asarray(joint, dtype=extended).sum(axis=0)
    q, m = len(weights), len(marginal)
    if q == 1:
        return csr_matrix(np.asarray(marginal / marginal.sum(), dtype=float)[None, :])
    active = (np.flatnonzero(marginal > 0) if source_order is None
              else source_order[marginal[source_order] > 0])
    theta = np.asarray(joint[1, active], dtype=extended) / marginal[active]
    if source_order is None:
        order = np.argsort(theta, kind='stable')
        theta, active = theta[order], active[order]
    rows, columns, data = inverse_transform(
        np.ascontiguousarray(support, dtype=extended),
        np.ascontiguousarray(weights, dtype=extended),
        np.ascontiguousarray(theta), np.ascontiguousarray(marginal[active]))
    columns = active[columns]
    data = np.asarray(data / weights[rows], dtype=float)
    kernel = csr_matrix(sparse_rows(rows, columns, data, q), shape=(q, m))
    row_sums = np.asarray(kernel.sum(axis=1)).ravel()
    if not np.all(np.isfinite(kernel.data)) or np.any(row_sums <= 0):
        raise ArithmeticError('source decoder reconstruction produced an invalid row')
    kernel.data /= np.repeat(row_sums, np.diff(kernel.indptr))
    residual = float(np.max(np.abs(target_joint @ kernel - joint)))
    if residual > max(100 * atol, 1e-8):
        raise ArithmeticError(f'source decoder reconstruction residual is too large: {residual}')
    return kernel


def union_binary_target(
    prior: ArrayLike,
    channels: Iterable[ArrayLike],
    *,
    return_channel: bool = False,
    return_garblings: bool = False,
    atol: float = 1e-12,
) -> UnionResult:
    """Compute the Blackwell union and join for a binary target.

    ``prior`` contains two nonnegative target weights, normalized internally.
    Channels have shape (2,m_i); packed (k,2,m) arrays, ragged lists, and
    iterators are accepted. Input validation and normalization match
    ``redundancy_binary_target``: integer channel weights lie in [0,2**32-1]
    and have equal positive row sums on positive-prior rows, while floating
    rows are normalized separately. Zero-prior rows may be zero; zero source
    columns are allowed. Inputs are never modified.

    The join call is the pointwise maximum of source calls. Its upper envelope
    is computed in O(N log N) operations and O(N) worst-case memory, retaining
    only each bounded batch's envelope. uint32 counts bypass joint-array
    normalization unless decoders are requested. No posterior discretization
    or convex optimizer is used. Extended precision is retained where
    available; ``atol`` controls consistency checks, not a certified error
    bound. The returned optimization bounds coincide because there is no
    iterative optimization gap; they remain floating-point values.

    With ``return_channel=True``, channel[y,q] is P(Q|Y), posteriors[:,q] is
    P(Y|Q), and posterior_weights[q] is P(Q). Independently,
    ``return_garblings=True`` returns source decoders P(X_i|Q) as SciPy CSR
    matrices of shape (q,m_i), opposite to redundancy garblings. Decoder
    construction takes O(N+k*q) additional operations and sparse storage.
    Both flags default to False and omitted outputs are None. SciPy is needed
    only for decoders. The join has at most N-k+1 positive-probability outputs.
    """
    if (not isinstance(atol, Real) or isinstance(atol, (bool, np.bool_))
            or not np.isfinite(atol) or atol <= 0):
        raise ValueError('atol must be finite and positive')
    prior = normalize_prior(prior)
    if len(prior) != 2:
        raise ValueError('a binary target requires a prior with two entries')
    tables = [] if return_garblings else None
    source_orders = [] if return_garblings else None
    support, weights, sources, states = _posterior_join(
        _channel_batches(channels, prior), prior, atol, tables, source_orders)
    posteriors = np.vstack((1 - support, support))
    common = make_result(prior, posteriors, weights, return_channel=return_channel)
    garblings, residual = None, None
    if return_garblings:
        joint = posteriors * weights
        garblings = tuple(_decoder(table, joint, support, weights, atol, order)
                          for table, order in zip(tables, source_orders))
        residual = max(float(np.max(np.abs(joint @ decoder - table)))
                       for decoder, table in zip(garblings, tables))
    value = common.redundancy_nats
    return UnionResult(
        union_nats=value, target_prior=prior, lower_bound_nats=value,
        upper_bound_nats=value, gap_nats=0., feasibility_residual=residual,
        method='binary-target upper envelope',
        diagnostics={'sources': sources, 'source_states': states,
                     'join_outputs': len(weights)},
        posteriors=common.posteriors, posterior_weights=common.posterior_weights,
        channel=common.channel, garblings=garblings, max_garbling_residual=residual,
    )
