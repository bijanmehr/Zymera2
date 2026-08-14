"""World generation (spec §5): parameterized, jit-safe, padded to caps.

Terrains: ``open`` (no walls), ``clutter`` (scattered obstacle cells), ``rooms``
(four rooms with doors), ``mixed`` (rooms + clutter — corridors AND debris). Spawns are distinct free cells. The terrain string and
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
    terrain: str = "open"          # open | clutter | rooms | mixed
    n_obstacles: int = 0           # clutter/mixed
    door: int = 1                  # rooms/mixed: half-width of door gaps
    spawn: str = "scatter"         # scatter | cluster (random anchor) | corner (v0 tests:
                                   #   bunched contiguous block at one side of the world)
    spawn_radius: int = 2          # cluster/corner: distance-penalty scale (v0 convention)
    corner: str = "tl"             # corner mode: tl | tr | bl | br


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

    def _clutter(sub):
        score = jax.random.uniform(jax.random.fold_in(key, sub),
                                   (static.h_max, static.w_max))
        score = jnp.where(arena, score, -jnp.inf)
        k = min(gcfg.n_obstacles, static.h_max * static.w_max)
        if k == 0:
            return jnp.zeros_like(arena)
        thresh = jax.lax.top_k(score.ravel(), k)[0][-1]
        return (score >= thresh) & arena

    if gcfg.terrain == "open":
        wall = jnp.zeros((static.h_max, static.w_max), bool)
    elif gcfg.terrain == "rooms":
        wall = _rooms_walls(static, params, gcfg.door) & arena
    elif gcfg.terrain == "clutter":
        wall = _clutter(1)
    elif gcfg.terrain == "mixed":
        rooms = _rooms_walls(static, params, gcfg.door) & arena
        rr2 = jnp.arange(static.h_max)[:, None]
        cc2 = jnp.arange(static.w_max)[None, :]
        # keep clutter off the door lines so doors are never sealed shut
        q1, q3 = params.h // 4, (3 * params.h) // 4
        c1, c3 = params.w // 4, (3 * params.w) // 4
        door_zone = ((jnp.abs(rr2 - q1) <= gcfg.door) | (jnp.abs(rr2 - q3) <= gcfg.door)
                     | (jnp.abs(cc2 - c1) <= gcfg.door) | (jnp.abs(cc2 - c3) <= gcfg.door))
        wall = rooms | (_clutter(1) & ~door_zone)
    else:
        raise ValueError(f"unknown terrain {gcfg.terrain!r}")

    # Distinct free spawn cells: rank free cells by score, take the top E.
    # scatter: uniform random score. cluster: nearest-to-anchor (random free anchor),
    # noise-tiebroken — the team STARTS connected (v0's spawn_radius convention).
    E = static.n_max + static.m_max
    noise = jax.random.uniform(jax.random.fold_in(key, 2),
                               (static.h_max, static.w_max))
    free = arena & ~wall
    if gcfg.spawn in ("cluster", "corner"):
        if gcfg.spawn == "corner":
            ar = jnp.where(gcfg.corner[0] == "t", 1, params.h - 2)
            ac = jnp.where(gcfg.corner[1] == "l", 1, params.w - 2)
        else:
            a_score = jnp.where(free, noise, -jnp.inf)
            a_flat = jnp.argmax(a_score)
            ar, ac = a_flat // static.w_max, a_flat % static.w_max
        rr2 = jnp.arange(static.h_max)[:, None]
        cc2 = jnp.arange(static.w_max)[None, :]
        d = jnp.maximum(jnp.abs(rr2 - ar), jnp.abs(cc2 - ac)).astype(jnp.float32)
        sscore = jnp.where(free, -d + noise / jnp.float32(max(gcfg.spawn_radius, 1)),
                           -jnp.inf)
    else:
        sscore = jnp.where(free, noise, -jnp.inf)
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
