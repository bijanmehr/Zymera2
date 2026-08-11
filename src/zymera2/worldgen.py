"""World generation (spec §5): parameterized, jit-safe, padded to caps.

Terrains: ``open`` (no walls), ``clutter`` (scattered obstacle cells), ``rooms``
(four rooms with doors). Spawns are distinct free cells. The terrain string and
counts live in :class:`GenConfig` (static — changing terrain recompiles); the
runtime sizes come from the ``WorldParams`` it carries.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .typing import StaticWorldParams, World, WorldParams


@dataclass(frozen=True)
class GenConfig:
    """Static generation recipe. ``params`` are the runtime sizes the world is built at."""
    terrain: str = "open"          # open | clutter | rooms
    n_obstacles: int = 0           # clutter only
    door: int = 1                  # rooms: half-width of door gaps


def _rooms_walls(static: StaticWorldParams, params: WorldParams, door: int) -> jax.Array:
    """Four-room partition: one vertical + one horizontal wall with centered door gaps."""
    rr = jnp.arange(static.h_max)[:, None]
    cc = jnp.arange(static.w_max)[None, :]
    mid_r, mid_c = params.h // 2, params.w // 2
    vwall = (cc == mid_c) & (rr < params.h)
    hwall = (rr == mid_r) & (cc < params.w)
    q1, q3 = params.h // 4, (3 * params.h) // 4
    c1, c3 = params.w // 4, (3 * params.w) // 4
    vdoor = (jnp.abs(rr - q1) <= door) | (jnp.abs(rr - q3) <= door)
    hdoor = (jnp.abs(cc - c1) <= door) | (jnp.abs(cc - c3) <= door)
    return (vwall & ~vdoor) | (hwall & ~hdoor)


def generate(gcfg: GenConfig, static: StaticWorldParams, params: WorldParams,
             key: jax.Array) -> World:
    """Build a fresh :class:`World` (Generator-conforming when partially applied).

    Deterministic in ``key``; every random draw uses ``fold_in`` (spec §14/C6).
    """
    rr = jnp.arange(static.h_max)[:, None]
    cc = jnp.arange(static.w_max)[None, :]
    arena = (rr < params.h) & (cc < params.w)

    if gcfg.terrain == "open":
        wall = jnp.zeros((static.h_max, static.w_max), bool)
    elif gcfg.terrain == "rooms":
        wall = _rooms_walls(static, params, gcfg.door) & arena
    elif gcfg.terrain == "clutter":
        score = jax.random.uniform(jax.random.fold_in(key, 1),
                                   (static.h_max, static.w_max))
        score = jnp.where(arena, score, -jnp.inf)
        k = min(gcfg.n_obstacles, static.h_max * static.w_max)
        thresh = jax.lax.top_k(score.ravel(), k)[0][-1] if k > 0 else jnp.inf
        wall = (score >= thresh) & arena if k > 0 else jnp.zeros_like(arena)
    else:
        raise ValueError(f"unknown terrain {gcfg.terrain!r}")

    # Distinct free spawn cells: rank free cells by random score, take the top E.
    E = static.n_max + static.m_max
    sscore = jax.random.uniform(jax.random.fold_in(key, 2),
                                (static.h_max, static.w_max))
    sscore = jnp.where(arena & ~wall, sscore, -jnp.inf)
    _, flat = jax.lax.top_k(sscore.ravel(), E)
    spawn = jnp.stack([flat // static.w_max, flat % static.w_max], -1).astype(jnp.int32)

    return World(
        agent_pos=spawn[: static.n_max],
        agent_alive=jnp.arange(static.n_max) < params.n,
        body_pos=spawn[static.n_max:],
        body_alive=jnp.arange(static.m_max) < params.m,
        body_kind=jnp.zeros((static.m_max,), jnp.int32),
        wall=wall, arena=arena,
        step_count=jnp.zeros((), jnp.int32))
