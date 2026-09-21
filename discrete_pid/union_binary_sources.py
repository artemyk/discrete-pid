"""Blackwell union from a target prior and binary conditional channels."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral, Real
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from ._common import UnionResult, normalize_prior
from ._validation import integer_weights, normalize_rows, validate_row_sums


def _information(p, q):
    weights = q.sum(axis=0)
    positive = q > 0
    reference = p[:, None]*weights
    logarithm = np.zeros_like(q)
    np.log(np.divide(q, reference, out=np.ones_like(q), where=positive),
           out=logarithm, where=positive)
    return max(0., float(np.sum(q*logarithm)))


def union_binary_sources(
    prior: ArrayLike,
    channels: Iterable[ArrayLike],
    *,
    tolerance: float = 1e-7,
    batch_size: int = 10,
    max_iterations: int = 500,
    max_pricing_nodes: int = 200000,
    use_heuristic: bool = True,
    reduce_sources: bool = True,
    return_channel: bool = False,
    return_garblings: bool = False,
) -> UnionResult:
    """Minimize information among common Blackwell upper channels.

    ``channels`` has shape (k,d,2), or is an iterable of (d,2) arrays.
    Prior and channel rows are finite nonnegative weights, normalized
    internally. Rows with zero prior are ignored and may have zero mass.
    Inputs are not modified. Integer weights must fit the package's uint32
    input contract, with equal positive row totals within each source on
    positive-prior rows. Floating-point rows are normalized separately.

    Binary targets use a direct envelope; two retained sources use a scalar
    search. Other inputs use an adaptive relative-entropy program with global
    pricing before termination. An LP periodically compresses the active
    support while checking moments and information; inconclusive compression
    retains the original patterns. No posterior grid is used. The reported
    ``tolerance`` must be at least 1e-10. ``gap_nats`` is a floating-point
    primal/global-dual diagnostic, not an
    interval-arithmetic certificate. ``feasibility_residual`` measures the
    reconstruction of all ORIGINAL source-target distributions.

    Optional source reduction verifies nonnegative decoder reconstructions
    and their residuals; inconclusive geometry retains all constraints.
    Zero/one source probabilities are handled by restricting the optimization
    to compatible target-pattern pairs. Work limits raise without returning
    an unsupported convergence claim. Runtime may be exponential for large
    target alphabets or many sources that survive reduction.

    ``return_channel`` supplies P(Q|Y), posteriors, and output weights.
    ``return_garblings`` independently supplies P(X_i|Q), with shape (m,2).
    Zero-prior target rows of the returned channel use P(Q). General source
    optimization requires ``discrete-pid[union]``; dependencies are lazy.
    """
    if (not isinstance(tolerance, Real) or isinstance(tolerance, (bool, np.bool_))
            or not np.isfinite(tolerance) or tolerance < 1e-10):
        raise ValueError('tolerance must be finite and at least 1e-10 for floating-point optimization')
    for name, value in [('batch_size', batch_size), ('max_iterations', max_iterations),
                        ('max_pricing_nodes', max_pricing_nodes)]:
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    p = normalize_prior(prior)
    support = p > 0
    if isinstance(channels, np.ndarray):
        raw = channels
    else:
        items = [np.asarray(item) for item in channels]
        # Preserve each source's integer contract before mixed inputs promote
        # the combined array to floating point.
        for item in items:
            if item.shape != (len(p), 2):
                raise ValueError('each binary source channel must have shape (target states, 2)')
            if integer_weights(item):
                validate_row_sums(np.asarray(item[None], dtype=np.uint32), support)
        raw = np.asarray(items)
    if raw.ndim != 3 or raw.shape[0] == 0 or raw.shape[1:] != (len(p), 2):
        raise ValueError('channels must have shape (number of sources, target states, 2)')
    if integer_weights(raw):
        validate_row_sums(np.asarray(raw, dtype=np.uint32), support)
    probabilities = normalize_rows(raw, support)
    active_prior = p[support]
    original = probabilities[:, support, 1].T
    d, original_k = original.shape

    if d == 2:
        from .union_binary_target import union_binary_target
        result = union_binary_target(active_prior, probabilities[:, support],
                                     return_channel=return_channel,
                                     return_garblings=return_garblings)
        posterior = channel = None
        if return_channel:
            posterior = np.zeros((len(p), result.posteriors.shape[1]))
            posterior[support] = result.posteriors
            channel = np.tile(result.posterior_weights, (len(p), 1))
            channel[support] = result.channel
        diagnostics = dict(result.diagnostics or {})
        diagnostics.update(retained_sources=original_k, original_sources=original_k)
        garblings = result.garblings
        if garblings is not None:
            garblings = tuple(kernel.toarray() if hasattr(kernel, 'toarray') else kernel
                              for kernel in garblings)
        return replace(result, target_prior=p, channel=channel, posteriors=posterior,
                       garblings=garblings, diagnostics=diagnostics)

    matrix = offset = None
    a = original
    reduction_residual = 0.
    if d == 1 or np.all(original == original[0]):
        a = original[:, :0]
        matrix, offset = np.zeros((0, original_k)), original[0].copy()
    elif original_k > 2 and reduce_sources:
        try:
            from ._union_sources_opt import reduce_sources as reduce
        except ImportError as error:
            raise ImportError('Binary-source union optimization requires discrete-pid[union]') from error
        a, matrix, offset, reduction_residual = reduce(
            original, residual_tolerance=min(1e-12, tolerance/100))
    k = a.shape[1]
    diagnostics = dict(retained_sources=k, original_sources=original_k,
                       source_reduction_residual=reduction_residual)
    iterations = 0
    if k == 0:
        q, patterns = active_prior[:, None], np.empty((1, 0))
        lower = upper = 0.
        method = 'constant sources or target'
    elif k == 1:
        q = active_prior[:, None]*np.column_stack((1-a[:, 0], a[:, 0]))
        patterns = np.array([[0], [1]], dtype=np.uint8)
        lower = upper = _information(active_prior, q)
        method = 'single retained source'
    elif k == 2:
        try:
            from ._union_two_sources import union_two_binary
        except ImportError as error:
            raise ImportError('Binary-source union optimization requires discrete-pid[union]') from error
        try:
            scalar = union_two_binary(active_prior, a[:, 0], a[:, 1],
                                      tolerance=tolerance, max_iterations=max_iterations)
        except (ArithmeticError, RuntimeError):
            # Extremely large log odds may exceed floating-point range. The
            # four-pattern convex problem remains a supported fallback.
            from ._union_sources_opt import optimize
            q, patterns, lower, upper, details = optimize(
                active_prior, a, tolerance=tolerance, batch_size=4,
                max_iterations=max_iterations, use_heuristic=False,
                max_pricing_nodes=max_pricing_nodes)
            iterations = details.pop('iterations')
            diagnostics.update(details)
            method = 'two-source relative-entropy fallback'
        else:
            q = active_prior[:, None]*scalar['conditional']
            patterns = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
            lower, upper, iterations = scalar['lower'], scalar['upper'], scalar['iterations']
            method = 'two-source scalar search'
    else:
        try:
            from ._union_sources_opt import optimize
        except ImportError as error:
            raise ImportError('Binary-source union optimization requires discrete-pid[union]') from error
        q, patterns, lower, upper, details = optimize(
            active_prior, a, tolerance=tolerance, batch_size=batch_size,
            max_iterations=max_iterations, use_heuristic=use_heuristic,
            max_pricing_nodes=max_pricing_nodes)
        iterations = details.pop('iterations')
        diagnostics.update(details)
        method = 'adaptive relative-entropy optimization'

    weights = q.sum(axis=0)
    positive = weights > 0
    q, patterns, weights = q[:, positive], patterns[positive], weights[positive]
    retained_moments = q @ patterns
    if matrix is None:
        reconstructed = retained_moments
    else:
        decoder_minimum = offset + np.minimum(matrix, 0).sum(axis=0)
        decoder_maximum = offset + np.maximum(matrix, 0).sum(axis=0)
        decoder_roundoff = float(max(0., -decoder_minimum.min(), decoder_maximum.max()-1))
        if decoder_roundoff > min(1e-10, tolerance/100):
            raise ArithmeticError('Source reduction produced invalid decoder probabilities')
        reconstructed = retained_moments @ matrix + q.sum(axis=1)[:, None]*offset
    # Always verify every original constraint, even when outputs are omitted.
    residual = float(max(np.max(np.abs(q.sum(axis=1)-active_prior)),
                         np.max(np.abs(reconstructed-active_prior[:, None]*original))))
    if residual > max(1e-12, tolerance/10):
        raise ArithmeticError(f'Original source-target reconstruction residual is {residual:g}')
    value = _information(active_prior, q)
    upper = value
    lower = max(0., min(float(lower), upper))
    gap = upper-lower
    if gap > tolerance + max(1e-12, tolerance*1e-3):
        raise ArithmeticError('Returned channel exceeds the requested numerical optimization gap')
    diagnostics['support'] = len(weights)
    garblings = None
    if return_garblings:
        decoded = np.asarray(patterns, dtype=float) if matrix is None else patterns @ matrix + offset
        decoded = np.clip(decoded, 0., 1.)
        garblings = tuple(np.column_stack((1-decoded[:, i], decoded[:, i]))
                          for i in range(original_k))
    posterior = channel = None
    if return_channel:
        posterior = np.zeros((len(p), len(weights)))
        posterior[support] = q/weights
        channel = np.tile(weights, (len(p), 1))
        channel[support] = q/active_prior[:, None]
    return UnionResult(union_nats=value, target_prior=p, lower_bound_nats=lower,
                       upper_bound_nats=upper, gap_nats=gap,
                       feasibility_residual=residual, iterations=iterations,
                       method=method, diagnostics=diagnostics, posteriors=posterior,
                       posterior_weights=weights if return_channel else None,
                       channel=channel, garblings=garblings,
                       max_garbling_residual=residual if return_garblings else None)
