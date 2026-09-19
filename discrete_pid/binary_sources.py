"""Blackwell redundancy when every source has at most two active states."""

from __future__ import annotations

from fractions import Fraction
from numbers import Integral, Real
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from ._common import Array, RedundancyResult, make_result, validate_joints


def redundancy_binary_sources(
    joints: Iterable[ArrayLike],
    *,
    tolerance: float | None = None,
    atol: float = 1e-12,
) -> RedundancyResult:
    """Intersect binary-source posterior segments in O(k*d) operations.

    Supply nonnegative, unnormalized INTEGER joint tables with target-first
    shape (d,m_i) and at most two positive-mass source columns. All tables
    must give exactly the same target marginal after normalization; their
    totals may differ. Python integers (including arbitrary-size integers)
    and NumPy integer arrays are supported. Padded zero columns are allowed.

    Integer inputs use exact integer collinearity checks and rational
    interval arithmetic, even when ``tolerance`` is supplied. The operation
    bound excludes integer bit costs. Returned probabilities, information,
    and source-to-experiment garblings are floating-point approximations.

    Floating-point inputs raise ValueError unless ``tolerance`` is explicitly
    set to a finite nonnegative number. This opt-in is necessary because
    arbitrarily small perturbations of collinear posterior segments can
    change positive redundancy to zero. In floating-point mode, ``tolerance``
    compares posterior directions scaled to maximum absolute coordinate one,
    allowing a sign reversal. It is NOT an error bound on the redundancy.
    Even ``tolerance=0`` does not restore exact arithmetic. See the README
    and ``max_garbling_residual``. Do not obtain counts by rounding probabilities.

    Floating-point tables may also be unnormalized. ``atol`` controls only
    their marginal reconciliation after normalization, not collinearity.
    Input arrays are never modified.
    """
    if (not isinstance(atol, Real) or isinstance(atol, (bool, np.bool_))
            or not np.isfinite(atol) or atol <= 0):
        raise ValueError("atol must be finite and positive")
    if tolerance is not None and (
        not isinstance(tolerance, Real) or isinstance(tolerance, (bool, np.bool_))
        or not np.isfinite(tolerance) or tolerance < 0
    ):
        raise ValueError("tolerance must be None or a finite nonnegative number")

    # Object arrays preserve Python integers before any NumPy coercion can
    # round large values or overflow fixed-width sums and cross-products.
    raw = [np.asarray(joint, dtype=object) for joint in joints]
    if not raw:
        raise ValueError("at least one source joint table is required")
    for index, joint in enumerate(raw):
        if joint.ndim != 2 or min(joint.shape) == 0:
            raise ValueError(f"joint {index} must be a nonempty two-dimensional table")
        if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
               for v in joint.flat):
            raise ValueError(f"joint {index} must contain real integer or floating-point values")
    if all(isinstance(v, Integral) for joint in raw for v in joint.flat):
        counts = [np.array([int(v) for v in joint.flat], dtype=object).reshape(joint.shape)
                  for joint in raw]
        return _integer_sources(counts)
    if tolerance is None:
        raise ValueError(
            "Floating-point binary-source inputs require an explicit numerical "
            "tolerance (e.g. tolerance=1e-12): collinearity is sensitive to rounding. "
            "Supply unnormalized integer joint tables for exact collinearity checks."
        )

    normalized = []
    for index, joint in enumerate(raw):
        if any(v < 0 or (not isinstance(v, Integral) and not np.isfinite(v))
               for v in joint.flat):
            raise ValueError(f"joint {index} must be finite and nonnegative")
        scale = max(joint.flat)
        if scale == 0:
            raise ValueError(f"joint {index} must have positive total mass")
        # Scale first to avoid overflowing a sum of large finite weights.
        table = np.array([float(v / scale) for v in joint.flat]).reshape(joint.shape)
        normalized.append(table / table.sum())
    tables, adjustment = validate_joints(normalized, atol)
    return _floating_sources(tables, adjustment, tolerance)


def _uninformative(tables: list[Array], adjustment: float = 0.0) -> RedundancyResult:
    return make_result(
        tables, tables[0].sum(axis=1)[:, None], np.ones(1), adjustment,
        tuple(np.ones((table.shape[1], 1)) for table in tables),
    )


def _probability(value: Fraction) -> float:
    result = float(value)
    if value > 0 and result == 0:
        raise ArithmeticError("a positive probability is too small for floating-point output")
    return result


def _integer_sources(counts: list[np.ndarray]) -> RedundancyResult:
    totals = []
    marginals = []
    active_columns = []
    common_rows = None
    for index, joint in enumerate(counts):
        if any(v < 0 for v in joint.flat):
            raise ValueError(f"joint {index} must be nonnegative")
        rows = joint.sum(axis=1).tolist()
        total = sum(rows)
        if total == 0:
            raise ValueError(f"joint {index} must have positive total mass")
        if common_rows is None:
            common_rows = rows
        elif len(rows) != len(common_rows) or any(
            row * totals[0] != reference * total
            for row, reference in zip(rows, common_rows)
        ):
            raise ValueError("integer joint tables must have exactly the same target marginal")
        marginal = joint.sum(axis=0).tolist()
        active = [x for x, mass in enumerate(marginal) if mass > 0]
        if len(active) > 2:
            raise ValueError("every source must have at most two positive-probability states")
        totals.append(total)
        marginals.append(marginal)
        active_columns.append(active)

    # These floating-point copies are used only to report results and residuals.
    tables = [np.array([_probability(Fraction(v, total)) for v in joint.flat])
              .reshape(joint.shape) for joint, total in zip(counts, totals)]
    direction = None
    pivot = None
    intervals = []
    for joint, total, marginal, active in zip(counts, totals, marginals, active_columns):
        if len(active) == 1:
            return _uninformative(tables)
        x0, x1 = active
        s0, s1 = marginal[x0], marginal[x1]
        # D / (s0*s1) is the posterior difference. All factors are Python
        # integers, so cross-products are exact even beyond int64 range.
        delta = [s0 * row[x1] - s1 * row[x0] for row in joint]
        if not any(delta):
            return _uninformative(tables)
        if direction is None:
            pivot = max(range(len(delta)), key=lambda y: abs(delta[y]))
            direction = delta
        elif any(d * direction[pivot] != r * delta[pivot]
                 for d, r in zip(delta, direction)):
            return _uninformative(tables)
        span = Fraction(delta[pivot], s0 * s1)
        probability_one = Fraction(s1, total)
        intervals.append((-probability_one * span, (1 - probability_one) * span))

    lower = max(min(interval) for interval in intervals)
    upper = min(max(interval) for interval in intervals)
    weights = [upper / (upper - lower), -lower / (upper - lower)]
    endpoints = [lower, upper]
    posteriors = np.empty((len(common_rows), 2))
    for q, t in enumerate(endpoints):
        # Every intersection endpoint is an original segment endpoint.
        index = next(i for i, interval in enumerate(intervals) if t in interval)
        x = active_columns[index][intervals[index].index(t)]
        posteriors[:, q] = [_probability(Fraction(v, marginals[index][x]))
                            for v in counts[index][:, x]]

    float_weights = np.array([_probability(w) for w in weights])
    garblings = []
    for total, marginal, active, (t0, t1) in zip(
        totals, marginals, active_columns, intervals
    ):
        kernel = np.tile(float_weights, (len(marginal), 1))
        for q, t in enumerate(endpoints):
            lam = (t - t0) / (t1 - t0)
            kernel[active[0], q] = _probability(weights[q] * (1 - lam) * total / marginal[active[0]])
            kernel[active[1], q] = _probability(weights[q] * lam * total / marginal[active[1]])
        garblings.append(kernel)
    return make_result(tables, posteriors, float_weights, 0.0, tuple(garblings))


def _floating_sources(
    tables: list[Array], adjustment: float, tolerance: float,
) -> RedundancyResult:
    prior = tables[0].sum(axis=1)
    sources = []
    for table in tables:
        marginal = table.sum(axis=0)
        active = np.flatnonzero(marginal > 0)
        if len(active) > 2:
            raise ValueError("every source must have at most two positive-probability states")
        sources.append((marginal, active, table[:, active] / marginal[active]))

    def uninformative() -> RedundancyResult:
        return _uninformative(tables, adjustment)

    direction = None
    pivot = None
    intervals = []
    for marginal, active, posterior in sources:
        if len(active) == 1:
            return uninformative()
        delta = posterior[:, 1] - posterior[:, 0]
        scale = float(np.max(np.abs(delta)))
        # Only an exactly zero direction is discarded as uninformative;
        # tolerance does not erase weak but informative observations.
        if scale == 0:
            return uninformative()
        unit = delta / scale
        if direction is None:
            pivot = int(np.argmax(np.abs(unit)))
            direction = unit / unit[pivot]
        else:
            # Account for a reversed ordering of a source's two states.
            sign = 1.0 if unit[pivot] >= 0 else -1.0
            if np.max(np.abs(sign * unit - direction)) > tolerance:
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
