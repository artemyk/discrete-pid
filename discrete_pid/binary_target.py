"""Binary-target Blackwell meet via a lower convex hull of call-function knots.

This implements the classical binary-experiment meet in posterior coordinates;
see Bertschinger and Rauh (2014), Section 4, https://arxiv.org/abs/1401.3146.
The hull takes O(N log N) operations and O(N) memory. Optional source-garbling
reconstruction uses SciPy linear programs, outside that complexity bound.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ._common import Array, RedundancyResult, make_result, validate_joints
from ._hull import lower_hull


def _posterior_meet(tables: list[Array], p: float, atol: float) -> tuple[Array, Array]:
    # Extended precision reduces cancellation when computing tail moments and
    # collinearity. It does not make the implementation exact arithmetic.
    extended = np.longdouble
    if all(table.shape == tables[0].shape for table in tables):
        # Equal source alphabets can be processed in one NumPy batch.
        batch = np.asarray(tables, dtype=extended)
        batch[:, 0] *= ((extended(1) - extended(p)) / batch[:, 0].sum(axis=1))[:, None]
        batch[:, 1] *= (extended(p) / batch[:, 1].sum(axis=1))[:, None]
        weights = batch.sum(axis=1)
        theta = np.zeros_like(weights)
        np.divide(batch[:, 1], weights, out=theta, where=weights > 0)
        order = np.argsort(theta, axis=1, kind="stable")
        theta = np.take_along_axis(theta, order, axis=1)
        weights = np.take_along_axis(weights, order, axis=1)
        tail_mass = np.cumsum(weights[:, ::-1], axis=1, dtype=extended)[:, ::-1]
        tail_moment = np.cumsum((weights * theta)[:, ::-1], axis=1, dtype=extended)[:, ::-1]
        calls = np.maximum(tail_moment - theta * tail_mass, 0)
        interior = (theta > 0) & (theta < 1) & (weights > 0)
        locations = np.concatenate(([extended(0), extended(1)], theta[interior]))
        heights = np.concatenate(([extended(p), extended(0)], calls[interior]))
    else:
        locations = [np.array([0, 1], dtype=extended)]
        heights = [np.array([p, 0], dtype=extended)]
        for table in tables:
            table_ext = table.astype(extended)
            # Match the common endpoints also in extended precision.
            table_ext[0] *= (extended(1) - extended(p)) / table_ext[0].sum()
            table_ext[1] *= extended(p) / table_ext[1].sum()
            weights = table_ext.sum(axis=0)
            positive = weights > 0
            theta = table_ext[1, positive] / weights[positive]
            weights = weights[positive]
            order = np.argsort(theta, kind="stable")
            theta, weights = theta[order], weights[order]
            tail_mass = np.cumsum(weights[::-1], dtype=extended)[::-1]
            tail_moment = np.cumsum((weights * theta)[::-1], dtype=extended)[::-1]
            calls = np.maximum(tail_moment - theta * tail_mass, 0)
            interior = (theta > 0) & (theta < 1)
            locations.append(theta[interior])
            heights.append(calls[interior])
        locations, heights = np.concatenate(locations), np.concatenate(heights)

    # NumPy sorts the coordinates in compiled code. At tied abscissae, keep
    # only the lowest point, exactly as in the original lexicographic sort.
    order = np.lexsort((heights, locations))
    locations, heights = locations[order], heights[order]
    unique = np.concatenate(([True], locations[1:] != locations[:-1]))
    locations, heights = locations[unique], heights[unique]
    hull = lower_hull(locations, heights)
    locations, heights = locations[hull], heights[hull]
    slopes = np.diff(heights) / np.diff(locations)
    # The exact hull has slopes in [-1, 0]. Very short endpoint segments can
    # amplify roundoff in the divided differences; clipping preserves the
    # monotonicity of the slopes and avoids invalid endpoint atom weights.
    slopes = np.clip(slopes, -1, 0)
    masses = np.diff(np.concatenate(([extended(-1)], slopes, [extended(0)])))
    if np.min(masses) < -atol:
        raise ArithmeticError("numerical error produced an invalid posterior law")
    masses = np.maximum(masses, 0)
    # Sub-precision slope jumps can otherwise create spurious output atoms.
    # Use machine precision here, not the user's input-normalization tolerance.
    positive = masses > 8 * np.finfo(extended).eps
    support = np.asarray(locations[positive], dtype=float)
    weights = np.asarray(masses[positive], dtype=float)
    weights /= weights.sum()
    if abs(float(support @ weights) - p) > 10 * atol:
        raise ArithmeticError("numerical error changed the optimal posterior mean")
    return support, weights


def _garbling(joint: Array, target_joint: Array, weights: Array, atol: float) -> Array:
    try:
        from scipy import sparse
        from scipy.optimize import linprog
    except ImportError as exc:
        raise ImportError(
            "Garbling reconstruction needs SciPy; install discrete-pid[garblings]"
        ) from exc
    marginal = joint.sum(axis=0)
    positive = marginal > 0
    source = joint[:, positive]
    source_count, auxiliary_count = source.shape[1], target_joint.shape[1]
    constraints = sparse.vstack(
        (
            sparse.kron(sparse.eye(source_count), np.ones((1, auxiliary_count))),
            sparse.kron(sparse.csr_matrix(source), sparse.eye(auxiliary_count)),
        ),
        format="csr",
    )
    rhs = np.concatenate((np.ones(source_count), target_joint.ravel()))
    solution = linprog(
        np.zeros(source_count * auxiliary_count),
        A_eq=constraints,
        b_eq=rhs,
        bounds=(0.0, None),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-10},
    )
    if not solution.success:
        raise RuntimeError(f"garbling reconstruction failed: {solution.message}")
    kernel = np.tile(weights, (joint.shape[1], 1))
    positive_kernel = np.maximum(solution.x.reshape(source_count, auxiliary_count), 0)
    row_sums = positive_kernel.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ArithmeticError("garbling reconstruction produced an empty row")
    kernel[positive] = positive_kernel / row_sums
    error = float(np.max(np.abs(joint @ kernel - target_joint)))
    if error > max(100 * atol, 1e-8):
        raise ArithmeticError(f"garbling reconstruction residual is too large: {error}")
    return kernel


def redundancy_binary_target(
    joints: Iterable[Array],
    *,
    return_garblings: bool = False,
    atol: float = 1e-12,
) -> RedundancyResult:
    """Compute redundancy for a binary target and arbitrary finite sources.

    Inputs are joint tables P(Y,X_i), each of shape (2,m_i), with the same
    target marginal. Zero-probability target and source states are allowed.
    The lower hull yields an optimal common experiment without posterior
    discretization or polytope-vertex enumeration. Arithmetic is floating
    point. Nats and bits are available on the returned result.

    Set return_garblings=True to also reconstruct P(Q|X_i) with SciPy linear
    programs. These LPs are not part of the O(N log N) hull bound.
    Within-atol normalization and target-marginal discrepancies are reconciled
    to the first input; the largest adjustment is reported in the result.
    """
    tables, adjustment = validate_joints(joints, atol)
    if tables[0].shape[0] != 2:
        raise ValueError("a binary target requires joint tables of shape (2,m)")
    p = float(tables[0][1].sum())
    if p == 0.0 or p == 1.0:
        support, weights = np.array([p]), np.ones(1)
    else:
        support, weights = _posterior_meet(tables, p, atol)
    posteriors = np.vstack((1 - support, support))
    garblings = None
    if return_garblings:
        target_joint = posteriors * weights
        if len(weights) == 1:
            garblings = tuple(np.ones((table.shape[1], 1)) for table in tables)
        else:
            garblings = tuple(_garbling(table, target_joint, weights, atol)
                              for table in tables)
    return make_result(tables, posteriors, weights, adjustment, garblings)
