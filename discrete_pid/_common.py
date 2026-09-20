"""Shared input validation and representation of a common experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RedundancyResult:
    """Information and optional common-experiment outputs, in floating-point form.

    With ``return_channel=True``, ``channel[y, q]`` is P(Q=q | Y=y),
    ``posteriors[:, q]`` is P(Y | Q=q), and ``posterior_weights[q]`` is P(Q=q).
    These fields are otherwise None. On zero-prior rows, ``channel`` is set
    to P(Q), an arbitrary stochastic extension. Independently, when requested
    with ``return_garblings=True``, ``garblings[i]`` is the row-stochastic matrix
    P(Q | X_i), with shape (number of source states, number of Q states).
    ``input_adjustment`` is zero: prior and channel weights are normalized
    by definition, without reconciling separately supplied target marginals.
    Garbling residuals use the joint distributions implied by these inputs.
    """

    redundancy_nats: float
    target_prior: Array
    posteriors: Array | None = None
    posterior_weights: Array | None = None
    garblings: tuple[Array, ...] | None = None
    input_adjustment: float = 0.0
    max_garbling_residual: float | None = None
    channel: Array | None = None

    @property
    def redundancy_bits(self) -> float:
        return self.redundancy_nats / np.log(2.0)

    @property
    def target_auxiliary_joint(self) -> Array | None:
        """P(Y,Q), or None unless return_channel=True."""
        if self.posteriors is None:
            return None
        return self.posteriors * self.posterior_weights


def normalize_prior(prior):
    raw = np.asarray(prior)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in 'iufO':
        raise ValueError("prior must be a nonempty vector of real nonnegative weights")
    try:
        p = np.asarray(raw, dtype=float)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("prior weights must be finite and nonnegative") from error
    if not np.all(np.isfinite(p)) or np.any(p < 0) or p.max() == 0:
        raise ValueError("prior weights must be finite, nonnegative, and have positive mass")
    positive = p > 0
    p = p / p.max()
    p /= p.sum()
    if np.any(positive & (p == 0)):
        raise ArithmeticError("a positive prior weight is too small for floating-point output")
    return p


def make_result(
    tables: list[Array] | None,
    posteriors: Array,
    weights: Array,
    adjustment: float,
    garblings: tuple[Array, ...] | None = None,
    *,
    return_channel: bool = False,
    prior: Array | None = None,
) -> RedundancyResult:
    if prior is None:
        prior = tables[0].sum(axis=1)
    # Extended precision limits cancellation near an uninformative experiment.
    r = np.asarray(posteriors[prior > 0], dtype=np.longdouble)
    p = np.asarray(prior[prior > 0], dtype=np.longdouble)
    log_r = np.zeros_like(r)
    np.log(r, out=log_r, where=r > 0)
    divergence = np.sum(r * (log_r - np.log(p[:, None])), axis=0)
    information = float(np.asarray(weights, dtype=np.longdouble) @ divergence)
    if information < -1e-12 or not np.isfinite(information):
        raise ArithmeticError("numerical error produced invalid mutual information")
    joint = posteriors * weights if return_channel or garblings is not None else None
    residual = None
    if garblings is not None:
        if all(table.shape == tables[0].shape for table in tables):
            residual = float(np.max(np.abs(np.asarray(tables) @ np.asarray(garblings) - joint)))
        else:
            residual = max(float(np.max(np.abs(table @ kernel - joint)))
                           for table, kernel in zip(tables, garblings))
    channel = None
    if return_channel:
        channel = np.tile(weights, (len(prior), 1))
        np.divide(joint, prior[:, None], out=channel, where=prior[:, None] > 0)
    return RedundancyResult(
        redundancy_nats=max(0.0, information), target_prior=prior,
        posteriors=posteriors if return_channel else None,
        posterior_weights=weights if return_channel else None,
        garblings=garblings, input_adjustment=adjustment,
        max_garbling_residual=residual, channel=channel,
    )
