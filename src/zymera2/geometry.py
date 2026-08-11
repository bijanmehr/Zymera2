"""Geometry queries (spec §8): the ONLY world surface the bridge may read.

``pairwise_dist`` — Chebyshev distance between alive agents (dead rows → +inf).
``wall_crossings`` — walls on the digital line between agent pairs, for occlusion.

Pinned ``-v0`` line algorithm: the digital line — cells ``round(a + (b-a)·t/L)`` for
``t = 1..L-1`` with ``L = max(cheby, 1)`` — NOT full supercover (deviation from the
spec's first choice, recorded here: corner-cutting cases differ; this one is simpler,
exact under jit, and fixed as the v0 behavior its golden files pin).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .typing import StaticWorldParams, World


def pairwise_dist(state: World) -> jax.Array:
    """Chebyshev distance [N,N] between agents; +inf to/from dead rows (so padded
    entities can never create adjacency — the alive mask is IN the geometry)."""
    d = jnp.abs(state.agent_pos[:, None, :] - state.agent_pos[None, :, :]).max(-1)
    alive = state.agent_alive
    inf = jnp.float32(jnp.inf)
    return jnp.where(alive[:, None] & alive[None, :], d.astype(jnp.float32), inf)


def wall_crossings(static: StaticWorldParams, state: World) -> jax.Array:
    """[N,N] count of interior digital-line cells that are walls, agent i → agent j.

    Fixed T = h_max + w_max samples (jit-safe); sample t/L positions along the segment,
    rounded to cells, deduplicated by counting only the first sample landing in each
    new cell. Symmetric in practice for Chebyshev lines; dead rows → 0.
    """
    T = static.h_max + static.w_max
    a = state.agent_pos.astype(jnp.float32)                       # [N,2]
    diff = a[None, :, :] - a[:, None, :]                          # [N,N,2]
    L = jnp.maximum(jnp.abs(diff).max(-1), 1.0)                   # [N,N]
    ts = jnp.arange(1, T + 1, dtype=jnp.float32)                  # [T]
    frac = ts[None, None, :] / (L[..., None] + 1e-9)              # [N,N,T]
    valid = frac < 1.0                                            # interior samples only
    pts = a[:, None, None, :] + diff[:, :, None, :] * frac[..., None]
    cells = jnp.round(pts).astype(jnp.int32)                      # [N,N,T,2]
    cells = jnp.clip(cells, 0, jnp.array([static.h_max - 1, static.w_max - 1]))
    hit = state.wall[cells[..., 0], cells[..., 1]] & valid        # [N,N,T]
    # count first-entry into each cell: a sample counts if it differs from the previous one
    prev = jnp.concatenate([jnp.full_like(cells[:, :, :1], -1), cells[:, :, :-1]], 2)
    new_cell = (cells != prev).any(-1)
    k = (hit & new_cell).sum(-1)
    alive = state.agent_alive
    return jnp.where(alive[:, None] & alive[None, :], k, 0)
