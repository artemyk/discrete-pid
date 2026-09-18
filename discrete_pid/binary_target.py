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


def _posterior_meet(tables: list[Array], p: float, atol: float) -> tuple[Array, Array]:
    # Extended precision reduces cancellation when computing tail moments and
    # collinearity. It does not make the implementation exact arithmetic.
    extended = np.longdouble
    points = [(extended(0), extended(p)), (extended(1), extended(0))]
    for table in tables:
        table_ext = table.astype(extended)
        # Match the common endpoints also in extended precision. Float row
        # sums can otherwise introduce spurious slope changes near 0 or 1.
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
        call_values = np.maximum(tail_moment - theta * tail_mass, 0)
        # The common endpoints are known exactly from the specified prior.
        interior = (theta > 0) & (theta < 1)
        points.extend(zip(theta[interior], call_values[interior]))

    # Keeping only the lowest point at each abscissa is sufficient.
    points.sort()
    unique_points = []
    for point in points:
        if not unique_points or point[0] != unique_points[-1][0]:
            unique_points.append(point)

    hull = []
    for point in unique_points:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = ((b[0] - a[0]) * (point[1] - b[1])
                     - (b[1] - a[1]) * (point[0] - b[0]))
            if cross > 0:
                break
            hull.pop()
        hull.append(point)

    locations = np.asarray([point[0] for point in hull], dtype=extended)
    heights = np.asarray([point[1] for point in hull], dtype=extended)
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
