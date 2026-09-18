"""Shared input validation and representation of a common experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RedundancyResult:
    """Information and an optimal common experiment, in floating-point form.

    ``posteriors[:, q]`` is P(Y | Q=q), and ``posterior_weights[q]`` is
    P(Q=q). When present, ``garblings[i]`` is the row-stochastic matrix
    P(Q | X_i), with shape (number of source states, number of Q states).
    ``input_adjustment`` records the largest absolute change to an input
    entry during within-tolerance normalization. Garbling residuals are
    measured against the normalized inputs.
    """

    redundancy_nats: float
    target_prior: Array
    posteriors: Array
    posterior_weights: Array
    garblings: tuple[Array, ...] | None = None
    input_adjustment: float = 0.0
    max_garbling_residual: float | None = None

    @property
    def redundancy_bits(self) -> float:
        return self.redundancy_nats / np.log(2.0)

    @property
    def target_auxiliary_joint(self) -> Array:
        """P(Y,Q), in the original target-state order."""
        return self.posteriors * self.posterior_weights


def validate_joints(joints: Iterable[Array], atol: float) -> tuple[list[Array], float]:
    """Copy and normalize joint tables; never mutate caller-owned arrays."""
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
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
    tables: list[Array],
    posteriors: Array,
    weights: Array,
    adjustment: float,
    garblings: tuple[Array, ...] | None = None,
) -> RedundancyResult:
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
    joint = posteriors * weights
    residual = None
    if garblings is not None:
        residual = max(float(np.max(np.abs(table @ kernel - joint)))
                       for table, kernel in zip(tables, garblings))
    return RedundancyResult(
        max(0.0, information), prior, posteriors, weights,
        garblings, adjustment, residual,
    )
