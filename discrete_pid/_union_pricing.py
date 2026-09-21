"""Low-dimensional pricing for binary-source Blackwell union information.

The optimization is continuous/exact, with no posterior grid.  This research
implementation uses floating-point geometry, rather than certified predicates.
"""

from fractions import Fraction
import heapq
from itertools import count

import numpy as np
from scipy.special import logsumexp


def _line_endpoints(b):
    """Endpoints of b.r=0 in the three-state posterior triangle."""
    points = []
    for j in range(3):
        if b[j] == 0:
            points.append(np.eye(3)[j])
        for ell in range(j):
            if (b[j] < 0 < b[ell]) or (b[ell] < 0 < b[j]):
                r = np.zeros(3)
                r[j] = b[ell] / (b[ell] - b[j])
                r[ell] = -b[j] / (b[ell] - b[j])
                points.append(r)
    points = np.asarray(points)
    distances = np.sum((points[:, None] - points[None, :]) ** 2, axis=2)
    first, last = np.unravel_index(np.argmax(distances), distances.shape)
    return points[first], points[last]


def _rational_sweeps(B):
    """Exact cell coverage, treating finite input floats as rational numbers."""
    d, k = B.shape
    columns = [tuple(Fraction(float(x)) for x in B[:, i]) for i in range(k)]
    if d == 2:
        h0 = [b[1] for b in columns]
        slope = [b[0]-b[1] for b in columns]
        events = sorted((-h0[j]/slope[j], j) for j in range(k)
                        if slope[j] and 0 < -h0[j]/slope[j] < 1)
        t = events[0][0]/2 if events else Fraction(1, 2)
        s0 = np.array([h0[j]+t*slope[j] > 0 for j in range(k)], dtype=np.uint8)
        yield s0, np.array([j for _, j in events], dtype=int), None
        return
    for i, b in enumerate(columns):
        if not min(b) < 0 < max(b):
            continue
        endpoints = []
        for j in range(3):
            if not b[j]:
                endpoints.append(tuple(Fraction(int(j == ell)) for ell in range(3)))
            for ell in range(j):
                if b[j]*b[ell] < 0:
                    r = [Fraction(0)] * 3
                    r[j], r[ell] = b[ell]/(b[ell]-b[j]), -b[j]/(b[ell]-b[j])
                    endpoints.append(tuple(r))
        r0, r1 = tuple(dict.fromkeys(endpoints))
        h0 = [sum(x*y for x, y in zip(r0, bj)) for bj in columns]
        h1 = [sum(x*y for x, y in zip(r1, bj)) for bj in columns]
        slope = [v-u for u, v in zip(h0, h1)]
        events = sorted((-h0[j]/slope[j], j) for j in range(k)
                        if slope[j] and 0 < -h0[j]/slope[j] < 1)
        t = events[0][0]/2 if events else Fraction(1, 2)
        s0 = np.array([h0[j]+t*slope[j] > 0 for j in range(k)], dtype=np.uint8)
        alternate = np.zeros(k, dtype=int)
        pivot = next(j for j in range(3) if b[j])
        for j, bj in enumerate(columns):
            if not h0[j] and not h1[j]:
                orientation = bj[pivot]*b[pivot]
                s0[j] = orientation < 0
                alternate[j] = int(orientation > 0) - int(orientation < 0)
        yield s0, np.array([j for _, j in events], dtype=int), alternate


def positive_pricing(B, c, limit=1, geometric_tolerance=1e-12, exact_geometry=None):
    """Return best candidate bit patterns and their log-sum-exp values.

    Maximize logsumexp(c + B @ s) for s in {0,1}^k, for 1 <= d <= 3.
    Returned rows of S and corresponding values are sorted best first.  The
    first row attains the global maximum in exact arithmetic.  ``limit=None``
    returns every candidate found; finite limits avoid constructing the full
    arrangement.  Extra candidates in degenerate arrangements are harmless:
    they are still binary patterns from the original optimization domain.

    For d=3, sweep each hyperplane segment in the posterior triangle, visiting
    both adjacent cells.  Incremental objective-vector updates give O(k^2 log k)
    work for fixed ``limit``.  Coincident hyperplanes move together.  For d=2,
    a single sorted breakpoint sweep suffices.

    ``exact_geometry=True`` uses rational arithmetic for cell coverage, treating
    the finite input floats as exact rationals.  The default, None, falls back
    to that version when float geometry encounters near coincidences; False
    forces float geometry with ``geometric_tolerance``.  Objective values always
    use floating point, so this is not an interval certificate for the value.
    """
    B = np.asarray(B, dtype=float)
    c = np.asarray(c, dtype=float)
    d, k = B.shape
    if c.shape != (d,) or not 1 <= d <= 3:
        raise ValueError("Expected B of shape (d,k), c of shape (d,), 1<=d<=3")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    if not np.all(np.isfinite(B)) or not np.all(np.isfinite(c)):
        raise ValueError("Pricing coefficients must be finite")

    scales = np.max(np.abs(B), axis=0) if k else np.empty(0)
    normals = np.divide(B, scales, out=np.zeros_like(B), where=scales != 0)
    lengths = np.linalg.norm(normals, axis=0)
    normals = np.divide(normals, lengths, out=np.zeros_like(B), where=lengths != 0)

    # Cheap triggers for rational fallback: near-zero components, nearly
    # coincident planes, or nearly coincident/boundary intersection events.
    use_rational = bool(exact_geometry)
    if exact_geometry is None and d > 1 and k:
        small = np.abs(normals)
        use_rational = bool(np.any((small > 0) & (small < 1e-10)))
        if d == 3 and not use_rational:
            mixed = np.flatnonzero((normals.min(axis=0) < 0) & (normals.max(axis=0) > 0))
            for i in mixed:
                cross = np.linalg.norm(np.cross(normals[:, i], normals.T), axis=1)
                if np.any((cross < 1e-10) & (np.arange(k) != i) & (lengths > 0)):
                    use_rational = True
                    break
                r0, r1 = _line_endpoints(normals[:, i])
                h0, h1 = r0 @ normals, r1 @ normals
                slope = h1-h0
                roots = np.divide(-h0, slope, out=np.full(k, np.inf), where=slope != 0)
                roots = np.sort(roots[(np.arange(k) != i) & (roots > -1e-10) & (roots < 1+1e-10)])
                if (np.any(np.diff(roots) < 1e-10) or np.any(np.abs(roots) < 1e-10)
                        or np.any(np.abs(roots-1) < 1e-10)):
                    use_rational = True
                    break
        elif d == 2 and not use_rational:
            slope = normals[0]-normals[1]
            roots = np.divide(-normals[1], slope, out=np.full(k, np.inf), where=slope != 0)
            roots = np.sort(roots[(roots > -1e-10) & (roots < 1+1e-10)])
            use_rational = bool(np.any(np.diff(roots) < 1e-10)
                                or np.any(np.abs(roots) < 1e-10)
                                or np.any(np.abs(roots-1) < 1e-10))

    candidates = {}

    def remember(s):
        key = np.packbits(s).tobytes()
        if key not in candidates:
            candidates[key] = (s.copy(), float(logsumexp(c + B @ s)))

    def sweep(s0, crossings, alternate=None):
        # The t-th row represents the interval after t crossing events.
        changes = B[:, crossings] * (1 - 2 * s0[crossings].astype(float))
        vectors = B @ s0 + np.vstack((np.zeros(d), np.cumsum(changes.T, axis=0)))
        if alternate is None:
            scores = logsumexp(c + vectors, axis=1)
        else:
            delta = B @ alternate.astype(float)
            scores = np.concatenate((logsumexp(c + vectors, axis=1),
                                     logsumexp(c + vectors + delta, axis=1)))
        count = len(scores) if limit is None else min(limit, len(scores))
        selected = np.argpartition(scores, len(scores)-count)[-count:]
        intervals = len(crossings) + 1
        for index in selected:
            side, step = divmod(int(index), intervals)
            s = s0.copy()
            s[crossings[:step]] = 1 - s[crossings[:step]]
            if side:
                s = (s.astype(int) + alternate).astype(np.uint8)
            remember(s)

    remember((B.sum(axis=0) > 0).astype(np.uint8))
    if d == 1 or not k:
        pass
    elif use_rational:
        for s0, crossings, alternate in _rational_sweeps(B):
            sweep(s0, crossings, alternate)
    elif d == 2:
        slope = B[0] - B[1]
        roots = np.divide(-B[1], slope, out=np.full(k, np.inf), where=slope != 0)
        crossings = np.flatnonzero((roots > 0) & (roots < 1))
        crossings = crossings[np.argsort(roots[crossings])]
        t = roots[crossings[0]] / 2 if len(crossings) else .5
        sweep((B[1] + t * slope > 0).astype(np.uint8), crossings)
    else:
        for i in np.flatnonzero((normals.min(axis=0) < 0) & (normals.max(axis=0) > 0)):
            b = normals[:, i]
            r0, r1 = _line_endpoints(b)
            h0, h1 = r0 @ normals, r1 @ normals
            slope = h1 - h0
            coincident = np.linalg.norm(np.cross(b, normals.T), axis=1) <= geometric_tolerance
            coincident[i] = True
            roots = np.divide(-h0, slope, out=np.full(k, np.inf), where=slope != 0)
            crossings = np.flatnonzero((roots > 0) & (roots < 1) & ~coincident)
            crossings = crossings[np.argsort(roots[crossings])]
            t = roots[crossings[0]] / 2 if len(crossings) else .5
            s0 = (h0 + t * slope > 0).astype(np.uint8)
            # b projected onto the simplex tangent points toward b.r > 0.
            direction = (b - b.mean()) @ normals
            s0[coincident] = direction[coincident] < 0
            alternate = np.zeros(k, dtype=int)
            alternate[coincident] = np.sign(direction[coincident]).astype(int)
            sweep(s0, crossings, alternate)

    results = sorted(candidates.values(), key=lambda item: item[1], reverse=True)
    if limit is not None:
        results = results[:limit]
    return np.asarray([item[0] for item in results]), np.asarray([item[1] for item in results])


def compatible_states(a, patterns):
    """Rows y at which each deterministic decoder pattern is admissible."""
    return ~np.any(((a[None] == 0) & (patterns[:, None] == 1))
                   | ((a[None] == 1) & (patterns[:, None] == 0)), axis=2)


def pattern_scores(B, c, patterns, a=None):
    logits = c + patterns @ B.T
    if a is not None:
        logits = np.where(compatible_states(a, patterns), logits, -np.inf)
    return logsumexp(logits, axis=1)


def cube_patterns(k):
    return ((np.arange(1 << k, dtype=np.uint64)[:, None]
             >> np.arange(k, dtype=np.uint64)) & 1).astype(np.uint8)


def improving_patterns(B, c, p):
    """Cheap block maximization; returned values are never global bounds."""
    d = len(p)
    starts = np.vstack((np.eye(d), p,
                        np.random.default_rng(4916).dirichlet(np.ones(d), 20)))
    patterns = (starts @ B > 0).astype(np.uint8)
    found = set()
    for _ in range(100):
        found.update(map(tuple, patterns))
        logits = c + patterns @ B.T
        posterior = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        new = (posterior @ B > 0).astype(np.uint8)
        if np.array_equal(patterns, new):
            break
        patterns = new
    candidates = np.asarray(sorted(found), dtype=np.uint8)
    return candidates, pattern_scores(B, c, candidates)


def _branch_pricing(B, c, a, tolerance, max_nodes):
    """Box bounds give a global binary-search bound in any target dimension.

    The bound allows each target coordinate to choose its remaining bits
    independently, and respects deterministic source rows. It may be loose;
    the explicit node limit raises instead of reporting an unverified gap.
    """
    d, k = B.shape
    order = np.argsort(np.ptp(B, axis=0))[::-1]
    coefficients = B[:, order]
    channels = None if a is None else a[:, order]
    independent = np.maximum(coefficients, 0)
    if channels is not None:
        independent = np.where(channels == 0, 0,
                               np.where(channels == 1, coefficients, independent))
    remaining = np.column_stack((np.cumsum(independent[:, ::-1], axis=1)[:, ::-1],
                                 np.zeros(d)))
    seeds = [(np.sum(B, axis=0) > 0).astype(np.uint8)]
    if a is not None:
        for row in a:
            seeds.append(np.where(row == 1, 1, np.where(row == 0, 0, seeds[0])))
    seed_array = np.asarray(seeds, dtype=np.uint8)
    scores = pattern_scores(B, c, seed_array, a)
    best_index = int(np.argmax(scores))
    best, pattern = float(scores[best_index]), seed_array[best_index].copy()
    serial = count()
    initial = float(logsumexp(c + remaining[:, 0]))
    heap = [(-initial, next(serial), 0, np.zeros(d), np.ones(d, dtype=bool), ())]
    nodes = 0
    while heap and -heap[0][0] > best + tolerance:
        if nodes >= max_nodes:
            raise RuntimeError(
                f'Global pricing exceeded max_pricing_nodes={max_nodes}; '
                'no convergence gap is being reported. Increase that limit or use fewer sources.'
            )
        _, _, depth, vector, allowed, bits = heapq.heappop(heap)
        nodes += 1
        for bit in (0, 1):
            new_allowed = allowed.copy()
            if channels is not None:
                forbidden = channels[:, depth] == (1-bit)
                new_allowed &= ~forbidden
            if not np.any(new_allowed):
                continue
            new_vector = vector + bit*coefficients[:, depth]
            new_bits = bits + (bit,)
            next_depth = depth + 1
            bound = float(logsumexp(c[new_allowed] + new_vector[new_allowed]
                                   + remaining[new_allowed, next_depth]))
            if next_depth == k:
                if bound > best:
                    best = bound
                    pattern = np.empty(k, dtype=np.uint8)
                    pattern[order] = new_bits
            elif bound > best + tolerance:
                heapq.heappush(heap, (-bound, next(serial), next_depth, new_vector,
                                      new_allowed, new_bits))
    # Pruned nodes have score at most best+tolerance, including those discarded
    # before the incumbent improved. Keep this allowance in the returned bound.
    upper = best + tolerance
    if heap:
        upper = max(upper, -heap[0][0])
    return pattern[None], np.array([best]), upper


def global_pricing(B, c, a, *, limit=10, tolerance=1e-9, max_nodes=200000):
    """Return candidate patterns, their scores, and a global numerical bound.

    Binary/ternary-target cell coverage uses exact rational predicates on the
    supplied floating-point coefficients. Objective arithmetic is floating
    point. Higher dimensions use a bounded-work branch-and-bound search.
    """
    d, k = B.shape
    boundary = a is not None and np.any((a == 0) | (a == 1))
    face = a if boundary else None
    if k <= 12:
        candidates = cube_patterns(k)
        values = pattern_scores(B, c, candidates, face)
        order = np.argsort(values)[::-1][:limit]
        candidates, values = candidates[order], values[order]
        upper = float(values[0])
    elif d <= 3 and not boundary:
        candidates, values = positive_pricing(B, c, limit=limit, exact_geometry=True)
        upper = float(values[0])
    elif d <= 3:
        pool = []
        for mask in range(1, 1 << d):
            targets = np.flatnonzero([(mask >> y) & 1 for y in range(d)])
            rows = a[targets]
            zeros, ones = np.any(rows == 0, axis=0), np.any(rows == 1, axis=0)
            if np.any(zeros & ones):
                continue
            free = ~(zeros | ones)
            offset = c[targets] + B[np.ix_(targets, ones)].sum(axis=1)
            local, _ = positive_pricing(B[np.ix_(targets, free)], offset,
                                        limit=limit, exact_geometry=True)
            completed = np.tile(ones.astype(np.uint8), (len(local), 1))
            completed[:, free] = local
            pool.append(completed)
        candidates = np.unique(np.vstack(pool), axis=0)
        values = pattern_scores(B, c, candidates, face)
        order = np.argsort(values)[::-1][:limit]
        candidates, values = candidates[order], values[order]
        upper = float(values[0])
    else:
        candidates, values, upper = _branch_pricing(B, c, face, tolerance, max_nodes)
    # Conservative roundoff allowance; this is still not interval arithmetic.
    scale = 1 + np.max(np.abs(c)) + np.sum(np.max(np.abs(B), axis=0))
    allowance = 32*np.finfo(float).eps*(k+2)*scale
    return candidates, values, upper + allowance
