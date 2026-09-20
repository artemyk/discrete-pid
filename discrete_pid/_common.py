"""Shared input validation and representation of a common experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    ``input_adjustment`` records the largest absolute change to an input
    entry during within-tolerance normalization or marginal reconciliation.
    For binary sources this is zero: prior and channel weights are normalized
    by definition. Garbling residuals are measured against the normalized
    joint distributions implied by the inputs.
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


def validate_joints(joints: Iterable[Array], atol: float) -> tuple[list[Array], float]:
    """Copy and normalize joint tables; never mutate caller-owned arrays."""
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    joints = list(joints)
    # A batch avoids thousands of small NumPy calls when alphabets agree.
    # Keep the general path for ragged, object, and other array-like inputs.
    if joints and all(isinstance(j, np.ndarray) and j.ndim == 2
                      and j.shape == joints[0].shape and min(j.shape) > 0
                      and j.dtype.kind in "fiu" for j in joints):
        original = np.asarray(joints, dtype=float)
        if not np.all(np.isfinite(original)) or np.any(original < 0):
            raise ValueError("joint tables must be finite and nonnegative")
        totals = original.sum(axis=(1, 2))
        if np.any(~np.isfinite(totals)) or np.any(totals <= 0) or np.any(abs(totals - 1) > atol):
            raise ValueError("joint tables must sum to one within atol")
        tables = original / totals[:, None, None]
        priors = tables.sum(axis=2)
        common_prior = priors[0]
        if np.any(abs(priors - common_prior) > atol):
            raise ValueError("all joint tables must have the same target marginal")
        if np.any((priors > 0) != (common_prior > 0)):
            raise ValueError("zero-probability target states must agree across sources")
        factors = np.ones_like(priors)
        np.divide(common_prior, priors, out=factors, where=priors > 0)
        tables *= factors[:, :, None]
        adjustment = float(np.max(np.abs(tables - original)))
        return list(tables), adjustment
    tables = []
    common_prior = None
    adjustment = 0.0
    for index, joint in enumerate(joints):
        original = np.asarray(joint)
        if np.iscomplexobj(original):
            raise ValueError(f"joint {index} must be real")
        original = np.asarray(original, dtype=float)
        if original.ndim != 2 or min(original.shape) == 0:
            raise ValueError(f"joint {index} must be a nonempty two-dimensional table")
        if not np.all(np.isfinite(original)) or np.any(original < 0):
            raise ValueError(f"joint {index} must be finite and nonnegative")
        total = float(original.sum())
        if not np.isfinite(total) or total <= 0 or abs(total - 1) > atol:
            raise ValueError(f"joint {index} must sum to one within atol")
        table = original / total
        prior = table.sum(axis=1)
        if common_prior is None:
            common_prior = prior.copy()
        elif prior.shape != common_prior.shape or not np.allclose(
            prior, common_prior, rtol=0, atol=atol
        ):
            raise ValueError("all joint tables must have the same target marginal")
        if not np.array_equal(prior > 0, common_prior > 0):
            raise ValueError("zero-probability target states must agree across sources")
        active = prior > 0
        table[active] *= (common_prior[active] / prior[active])[:, None]
        adjustment = max(adjustment, float(np.max(np.abs(table - original))))
        tables.append(table)
    if not tables:
        raise ValueError("at least one source joint table is required")
    return tables, adjustment


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
