"""Construction-time validation (spec §10.1 / §14). Raises at Python/trace-construction time
with an actionable message — never inside a jitted step. Keeps the contract's promises
(runtime ≤ caps; seed pools disjoint) enforceable before any compute runs.
"""
from __future__ import annotations

from typing import Dict, Set

from .typing import StaticWorldParams, WorldParams


def validate_params(static: StaticWorldParams, params: WorldParams) -> None:
    """Reject a (static, runtime) pair that violates the two-tier contract.

    Runtime dimensions must fit the compile caps; ranges must be positive; rules must be a
    tuple (immutable ordered — spec §14/C8). Raises :class:`ValueError` on the first problem.
    """
    caps = (("h", params.h, static.h_max), ("w", params.w, static.w_max),
            ("n", params.n, static.n_max), ("m", params.m, static.m_max))
    for name, v, cap in caps:
        if v > cap:
            raise ValueError(f"runtime {name}={v} exceeds cap {name}_max={cap}")

    positive = (("h", params.h), ("w", params.w), ("n", params.n),
                ("sense_r", params.sense_r))
    for name, v in positive:
        if v <= 0:
            raise ValueError(f"{name} must be positive, got {v}")
    if params.m < 0:
        raise ValueError(f"m must be non-negative, got {params.m}")

    if not isinstance(static.rules, tuple):
        raise ValueError("StaticWorldParams.rules must be an immutable tuple (no mutable registry)")


def validate_seed_pools(pools: Dict[str, Set[int]]) -> None:
    """Reject seed pools (e.g. train/test/eval) that share any seed (spec §14, G13) —
    the generalization split must be disjoint by construction, not by convention."""
    names = list(pools)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = pools[a] & pools[b]
            if shared:
                raise ValueError(f"seed pools overlap: {a} ∩ {b} = {sorted(shared)}")
