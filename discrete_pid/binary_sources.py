"""Blackwell redundancy when every source has at most two active states."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ._common import Array, RedundancyResult, make_result, validate_joints


def redundancy_binary_sources(
    joints: Iterable[Array],
    *,
    atol: float = 1e-12,
    geometry_tol: float = 1e-12,
) -> RedundancyResult:
    """Intersect binary-source posterior segments in O(k*d) operations.

    Each input is P(Y,X_i), with target-first shape (d,m_i) and at most
    two positive-probability source states. Padded zero columns are allowed.
    All inputs must have the same target marginal. The returned experiment
    has one or two outputs, and source-to-experiment garblings are included.

    ``atol`` controls input normalization and marginal checks.
    ``geometry_tol`` is an absolute tolerance for comparing directions
    normalized to have maximum absolute coordinate one. Distinct directions
    outside this tolerance give zero redundancy. Near-collinear inputs may
    be treated as collinear: the result is numerical, not a certificate of
    exact alignment. See ``max_garbling_residual`` and the README.
    """
    if not np.isfinite(geometry_tol) or geometry_tol < 0:
        raise ValueError("geometry_tol must be finite and nonnegative")
    tables, adjustment = validate_joints(joints, atol)
    prior = tables[0].sum(axis=1)
    sources = []
    for table in tables:
        marginal = table.sum(axis=0)
        active = np.flatnonzero(marginal > 0)
        if len(active) > 2:
            raise ValueError("every source must have at most two positive-probability states")
        sources.append((marginal, active, table[:, active] / marginal[active]))

    def uninformative() -> RedundancyResult:
        return make_result(
            tables, prior[:, None].copy(), np.ones(1), adjustment,
            tuple(np.ones((table.shape[1], 1)) for table in tables),
        )

    direction = None
    pivot = None
    intervals = []
    for marginal, active, posterior in sources:
        if len(active) == 1:
            return uninformative()
        delta = posterior[:, 1] - posterior[:, 0]
        scale = float(np.max(np.abs(delta)))
        # Only an exactly zero direction is discarded as uninformative;
        # geometry_tol does not erase weak but informative observations.
        if scale == 0:
            return uninformative()
        unit = delta / scale
        if direction is None:
            pivot = int(np.argmax(np.abs(unit)))
            direction = unit / unit[pivot]
        else:
            # Account for a reversed ordering of a source's two states.
            sign = 1.0 if unit[pivot] >= 0 else -1.0
            if np.max(np.abs(sign * unit - direction)) > geometry_tol:
                return uninformative()
        span = delta[pivot]
        # Barycentric weights place the prior at scalar coordinate zero.
        # This avoids subtracting nearly equal posterior and prior entries.
        probability_one = marginal[active[1]] / marginal[active].sum()
        t0 = -probability_one * span
        t1 = (1 - probability_one) * span
        intervals.append((t0, t1))

    lower = max(min(interval) for interval in intervals)
    upper = min(max(interval) for interval in intervals)
    if not lower < 0 < upper:
        return uninformative()
    weights = np.array([upper, -lower]) / (upper - lower)
    endpoints = np.array([lower, upper])

    # Evaluate endpoints as mixtures of a source attaining each boundary;
    # this preserves nonnegativity even on faces of the target simplex.
    posteriors = np.empty((len(prior), 2))
    for q, t in enumerate(endpoints):
        boundary = min if q == 0 else max
        index = next(i for i, interval in enumerate(intervals)
                     if boundary(interval) == t)
        t0, t1 = intervals[index]
        lam = np.clip((t - t0) / (t1 - t0), 0, 1)
        b = sources[index][2]
        posteriors[:, q] = (1 - lam) * b[:, 0] + lam * b[:, 1]

    garblings = []
    for (marginal, active, _), (t0, t1) in zip(sources, intervals):
        lam = np.clip((endpoints - t0) / (t1 - t0), 0, 1)
        kernel = np.tile(weights, (len(marginal), 1))
        kernel[active[0]] = weights * (1 - lam) / marginal[active[0]]
        kernel[active[1]] = weights * lam / marginal[active[1]]
        kernel /= kernel.sum(axis=1, keepdims=True)
        garblings.append(kernel)
    return make_result(tables, posteriors, weights, adjustment, tuple(garblings))
