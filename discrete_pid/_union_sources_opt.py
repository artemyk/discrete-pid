"""Convex masters and verified source reduction for binary-source union."""

import time

import clarabel
import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull, Delaunay, QhullError
from scipy.special import xlogy

from ._union_pricing import (compatible_states, global_pricing,
                             improving_patterns, pattern_scores)


def initial_support(a):
    patterns = set()
    for row in a:
        thresholds = np.unique(np.r_[0., row, 1.])
        for lo, hi in zip(thresholds[:-1], thresholds[1:]):
            patterns.add(tuple((row >= hi).astype(np.uint8)))
    return np.asarray(sorted(patterns), dtype=np.uint8)


def reduce_sources(a, residual_tolerance=1e-12):
    """Propose a hull reduction, verifying every original decoder afterward.

    Returns retained channels, affine decoder coefficients, offsets, and the
    maximum channel reconstruction residual. If geometric reconstruction is
    inconclusive, all original constraints are retained.
    """
    d, k = a.shape
    identity = (a, None, None, 0.)
    if k <= 2:
        return identity
    points = np.vstack((a.T, 1-a.T, np.zeros(d), np.ones(d)))
    _, singular, vt = np.linalg.svd(points-points[0], full_matrices=False)
    rank = np.count_nonzero(singular > max(1e-13, singular[0]*1e-13))
    projected = (points-points[0]) @ vt[:rank].T
    try:
        vertices = (np.array([np.argmin(projected[:, 0]), np.argmax(projected[:, 0])])
                    if rank == 1 else ConvexHull(projected).vertices)
        keep = sorted(set(int(index % k) for index in vertices if index < 2*k))
        if len(keep) == k or k*(2*len(keep)+2) > 20000000:
            return identity
        retained = a[:, keep]
        h = len(keep)
        generators = np.vstack((retained.T, 1-retained.T, np.zeros(d), np.ones(d)))
        unique, unique_ids = np.unique(generators, axis=0, return_index=True)
        coordinates = (unique-points[0]) @ vt[:rank].T
        queries = (a.T-points[0]) @ vt[:rank].T
        coefficients = np.zeros((k, len(generators)))
        if rank == 1:
            left, right = int(np.argmin(coordinates[:, 0])), int(np.argmax(coordinates[:, 0]))
            weight = ((queries[:, 0]-coordinates[left, 0])
                      / (coordinates[right, 0]-coordinates[left, 0]))
            coefficients[:, unique_ids[left]] = 1-weight
            coefficients[:, unique_ids[right]] = weight
        else:
            triangulation = Delaunay(coordinates)
            simplex = triangulation.find_simplex(queries, tol=1e-11)
            if np.any(simplex < 0):
                return identity
            transform = triangulation.transform[simplex]
            weights = np.einsum('ijk,ik->ij', transform[:, :rank],
                                queries-transform[:, rank])
            weights = np.column_stack((weights, 1-weights.sum(axis=1)))
            indices = unique_ids[triangulation.simplices[simplex]]
            coefficients[np.arange(k)[:, None], indices] = weights
        if np.min(coefficients) < -residual_tolerance:
            return identity
        coefficients = np.maximum(coefficients, 0)
        coefficients /= coefficients.sum(axis=1)[:, None]
        recovered = coefficients @ generators
        residual = float(np.max(np.abs(recovered-a.T)))
        if residual > residual_tolerance:
            return identity
        matrix = (coefficients[:, :h]-coefficients[:, h:2*h]).T
        offset = coefficients[:, h:2*h].sum(axis=1)+coefficients[:, -1]
        return retained, matrix, offset, residual
    except (QhullError, np.linalg.LinAlgError, ValueError, FloatingPointError):
        return identity


def restricted_master(p, a, patterns, tolerance):
    """Relative-entropy master with incompatible boundary entries removed."""
    d, k = a.shape
    count = len(patterns)
    compatible = compatible_states(a, patterns)
    qs, ys = np.nonzero(compatible)
    n = len(qs)
    equality_count = d*(k+1)
    features = np.column_stack((np.ones(n), patterns[qs]))
    equation_rows = (ys[:, None]+d*np.arange(k+1)).ravel()
    equation_columns = np.repeat(np.arange(n), k+1)
    equations = sp.csc_matrix((features.ravel(), (equation_rows, equation_columns)),
                              shape=(equality_count, 2*n))
    equations.eliminate_zeros()
    # Cone j is (-t_j,q_j,p[y_j]*w[s_j]), where w sums only compatible q's.
    marginal = sp.csc_matrix((np.ones(n), (qs, np.arange(n))), shape=(count, n))
    third = (sp.diags(p[ys]) @ marginal[qs]).tocoo()
    cone_rows = np.r_[3*np.arange(n), 3*np.arange(n)+1, 3*third.row+2]
    cone_columns = np.r_[n+np.arange(n), np.arange(n), third.col]
    entries = np.r_[np.ones(n), -np.ones(n), -third.data]
    cone_matrix = sp.csc_matrix((entries, (cone_rows, cone_columns)), shape=(3*n, 2*n))
    matrix = sp.vstack((equations, cone_matrix), format='csc')
    rhs = np.r_[p, (p[:, None]*a).T.ravel(), np.zeros(3*n)]
    cones = ([clarabel.ZeroConeT(equality_count)]
             + [clarabel.ExponentialConeT() for _ in range(n)])
    feasibility_limit = max(1e-12, tolerance/10)
    base_precision = min(1e-9, tolerance/100)
    objective = np.r_[np.zeros(n), np.ones(n)]
    quadratic = sp.csc_matrix((2*n, 2*n))
    failures = []
    for attempt in range(3):
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = 1
        precision = (base_precision if attempt == 0
                     else max(1e-13, min(1e-11, base_precision/100)))
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = precision
        # Clarabel otherwise accepts AlmostSolved with feasibility 1e-4.
        # Its reduced termination must respect the requested numerical scale.
        settings.reduced_tol_feas = min(1e-9, feasibility_limit/10)
        settings.reduced_tol_gap_abs = settings.reduced_tol_gap_rel = tolerance/20
        settings.max_iter = 300 if attempt == 0 else 500
        if attempt == 1:
            # Equilibration occasionally stalls on masters containing many
            # nearly unused posterior columns. Re-solving the original scale
            # with tighter tolerances repairs these cases without relaxing
            # either the moment residual or the optimization-gap requirement.
            settings.equilibrate_enable = False
        elif attempt == 2:
            settings.static_regularization_constant = 1e-10
            settings.dynamic_regularization_delta = 1e-9
            settings.iterative_refinement_abstol = 1e-14
            settings.iterative_refinement_reltol = 1e-14
            settings.iterative_refinement_max_iter = 30
            settings.min_switch_step_length = 1e-4
            settings.min_terminate_step_length = 1e-8
            settings.max_step_fraction = .95
        solver = clarabel.DefaultSolver(quadratic, objective, matrix, rhs, cones, settings)
        result = solver.solve()
        if str(result.status) not in ('Solved', 'AlmostSolved'):
            failures.append(f'attempt {attempt+1}: {result.status}')
            continue
        q = np.zeros((d, count))
        q[ys, qs] = np.maximum(np.asarray(result.x[:n]), 0.)
        totals = q.sum(axis=1)
        if np.any(totals <= 0) or not np.all(np.isfinite(q)):
            failures.append(f'attempt {attempt+1}: invalid target row')
            continue
        q *= (p/totals)[:, None]  # Make the returned channel stochastic.
        residual = max(np.max(np.abs(q.sum(axis=1)-p)),
                       np.max(np.abs(q @ patterns-p[:, None]*a)))
        if residual > feasibility_limit:
            failures.append(f'attempt {attempt+1}: moment residual {residual:g}')
            continue
        alpha = -np.asarray(result.z[:d])
        B = -np.asarray(result.z[d:equality_count]).reshape(k, d).T
        dual = float(p @ alpha+np.sum(p[:, None]*a*B))
        value = float(np.sum(xlogy(q, q)-xlogy(q, p[:, None]*q.sum(axis=0))))
        # Recompute the objective and a feasible restricted dual: epigraph
        # objectives can look converged even when the actual channel is not.
        violation = float(pattern_scores(B, np.log(p)+alpha, patterns, a).max())
        master_gap = value-(dual-max(0., violation))
        roundoff = 64*np.finfo(float).eps*(1+abs(value)+abs(dual))
        if (not np.isfinite(master_gap) or master_gap < -roundoff
                or master_gap > tolerance/2):
            failures.append(f'attempt {attempt+1}: reconstructed master gap {master_gap:g}')
            continue
        return value, alpha, B, dual, q, float(residual), attempt
    raise ArithmeticError('Blackwell union master did not attain the requested precision; '
                          + '; '.join(failures))


def optimize(p, a, *, tolerance, batch_size, max_iterations, use_heuristic,
             max_pricing_nodes):
    patterns = initial_support(a)
    boundary = bool(np.any((a == 0) | (a == 1)))
    global_calls = 0
    master_retries = 0
    master_seconds = pricing_seconds = 0.
    best_lower = 0.
    for iteration in range(max_iterations):
        tick = time.perf_counter()
        upper, alpha, B, dual, q, residual, retries = restricted_master(p, a, patterns, tolerance)
        master_retries += retries
        master_seconds += time.perf_counter()-tick
        tick = time.perf_counter()
        c = np.log(p)+alpha
        global_call = boundary or not use_heuristic
        if not global_call:
            candidates, values = improving_patterns(B, c, p)
            global_call = values.max() <= tolerance/4
        if global_call:
            candidates, values, price_bound = global_pricing(
                B, c, a, limit=batch_size, tolerance=tolerance/20, max_nodes=max_pricing_nodes)
            global_calls += 1
        order = np.argsort(values)[::-1][:batch_size]
        additions = candidates[order[values[order] > tolerance/10]]
        enlarged = np.unique(np.vstack((patterns, additions)), axis=0)
        if not global_call and len(enlarged) == len(patterns):
            candidates, values, price_bound = global_pricing(
                B, c, a, limit=batch_size, tolerance=tolerance/20, max_nodes=max_pricing_nodes)
            global_calls += 1
            global_call = True
            enlarged = np.unique(np.vstack((patterns, candidates[values > tolerance/10])), axis=0)
        pricing_seconds += time.perf_counter()-tick
        if global_call:
            best_lower = max(best_lower, dual-max(0., price_bound))
            gap_roundoff = 64*np.finfo(float).eps*(1+abs(upper)+abs(best_lower))
            if upper-best_lower < -gap_roundoff:
                raise ArithmeticError('Numerical dual bound exceeds primal value')
            if upper-best_lower <= tolerance:
                lower = min(upper, best_lower)
                return q, patterns, lower, upper, dict(
                    iterations=iteration+1, feasibility_residual=residual,
                    raw_gap_nats=upper-best_lower,
                    global_pricing_calls=global_calls, support=len(patterns),
                    solver_threads=1,
                    master_retries=master_retries,
                    master_seconds=master_seconds, pricing_seconds=pricing_seconds)
        if len(enlarged) == len(patterns):
            raise ArithmeticError('Global pricing stalled above the requested numerical gap')
        patterns = enlarged
    raise RuntimeError(f'Blackwell union exceeded max_iterations={max_iterations}')
