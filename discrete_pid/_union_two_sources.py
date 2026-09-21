"""Blackwell union for two binary sources by scalar search with convexity bounds.

This solves the continuous optimization problem, with no posterior grid.
Natural logarithms are used. Returned bounds are numerical evaluations of
exact convexity bounds, not interval-arithmetic certificates.
"""
import numpy as np
from scipy.special import xlogy


def _information(p, q):
    w = p @ q
    return float(np.sum(p[:, None] * (xlogy(q, q) - xlogy(q, w))))


def _lower_is_optimal(p, a, b):
    """Exact analytical endpoint test, evaluated in floating point."""
    c = 1.0 - a - b
    if np.all(c <= 0):
        a, b, c = 1.0 - a, 1.0 - b, -c
    elif not np.all(c >= 0):
        return False
    C = p @ c
    if C == 0:
        return True  # The channels are complements.
    ab = a * b
    if np.any((c == 0) & (ab > 0)):
        return False
    ratio_terms = np.divide(ab, c, out=np.zeros_like(ab), where=c > 0)
    return C * (p @ ratio_terms) <= (p @ a) * (p @ b)


def _conditional_tables(a, b, log_odds):
    """Columns are 00, 01, 10, 11; preserve small off-diagonal cells."""
    if log_odds < 0:
        return _conditional_tables(a, 1.0 - b, -log_odds)[:, [1, 0, 3, 2]]
    rho = np.exp(-log_odds)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    spread = hi - lo
    coefficient = spread + rho * (1.0 - spread)
    rhs = rho * lo * (1.0 - hi)
    denominator = coefficient + np.sqrt(coefficient**2 + 4.0 * (1.0 - rho) * rhs)
    delta = np.divide(2.0 * rhs, denominator, out=np.zeros_like(rhs), where=denominator > 0)
    t = lo - delta
    # Computing the two small cells directly avoids cancellation near a=b.
    q10 = np.where(a <= b, delta, spread + delta)
    q01 = np.where(a <= b, spread + delta, delta)
    return np.column_stack((1.0 - hi - delta, q01, q10, t))


def union_two_binary(p, a, b, tolerance=1e-10, max_iterations=200):
    """Return union information and a numerical primal/convexity gap.

    p is the target prior, and a,b are the two P(X=1|Y) vectors.
    Endpoint tests handle deterministic source rows as well as interior
    probabilities. At finite log odds, degenerate rows have fixed tables.
    """
    p, a, b = [np.asarray(v, dtype=float) for v in (p, a, b)]
    if p.ndim != 1 or a.shape != p.shape or b.shape != p.shape:
        raise ValueError("p, a, and b must be equal-length vectors")
    if (np.any(p < 0) or not np.isclose(p.sum(), 1.0)
            or np.any((a < 0) | (a > 1)) or np.any((b < 0) | (b > 1))):
        raise ValueError("invalid prior or channel probabilities")
    keep = p > 0
    p, a, b = p[keep], a[keep], b[keep]
    p = p / p.sum()
    lower = np.maximum(0.0, a + b - 1.0)
    upper = np.minimum(a, b)

    def endpoint(t, status):
        q = np.column_stack((1.0 - a - b + t, b - t, a - t, t))
        q = np.maximum(q, 0.0)
        value = _information(p, q)
        return dict(value=value, lower=value, upper=value, gap=0.0,
                    iterations=0, status=status, conditional=q)

    if np.all(lower == upper):
        return endpoint(lower, "unique coupling")
    if _lower_is_optimal(p, a, b):
        return endpoint(lower, "lower endpoint")
    if _lower_is_optimal(p, a, 1.0 - b):
        return endpoint(upper, "upper endpoint")

    def evaluate(lam):
        q = _conditional_tables(a, b, lam)
        w = p @ q
        if np.any(w <= 0):
            raise FloatingPointError("scalar search exceeded floating-point range")
        g = np.log(w[0]) + np.log(w[3]) - np.log(w[1]) - np.log(w[2]) - lam
        value = _information(p, q)
        t = q[:, 3]
        distance = p @ (upper - t if g >= 0 else t - lower)
        gap = max(0.0, float(abs(g) * distance))
        return g, dict(value=value, lower=value-gap, upper=value, gap=gap,
                       log_odds=lam, conditional=q, status="interior")

    g0, best = evaluate(0.0)
    if best["gap"] <= tolerance:
        best["iterations"] = 1
        return best
    lo, hi = (0.0, 1.0) if g0 > 0 else (-1.0, 0.0)
    evaluations = 1
    while True:
        candidate = hi if g0 > 0 else lo
        g, result = evaluate(candidate)
        evaluations += 1
        if result["gap"] < best["gap"]:
            best = result
        if best["gap"] <= tolerance:
            best["iterations"] = evaluations
            return best
        if (g0 > 0 and g <= 0) or (g0 < 0 and g >= 0):
            break
        if g0 > 0:
            hi *= 2
        else:
            lo *= 2
        if evaluations >= max_iterations:
            raise RuntimeError("could not bracket scalar optimum")
    while evaluations < max_iterations:
        mid = (lo + hi) / 2.0
        g, result = evaluate(mid)
        evaluations += 1
        if result["gap"] < best["gap"]:
            best = result
        if best["gap"] <= tolerance:
            best["iterations"] = evaluations
            return best
        if g > 0:
            lo = mid
        else:
            hi = mid
    raise RuntimeError("scalar search did not attain requested gap")
