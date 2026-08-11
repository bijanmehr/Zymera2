"""Sensing (spec §6, corrected §14/C4): pure readouts over the world — truth strictly
within range, never a step side-effect. Returns ``(readout, SenseEvent)``; the HARNESS
owns event-log concatenation (``world_step`` never emits sense events).

The radius is a static Python int (spec §14.5): patch sizes must be trace-constant,
so the tick is built per sense radius — a world-family constant, not a swept knob.
"""
from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from .typing import StaticWorldParams, World, WorldParams


@chex.dataclass(frozen=True)
class SenseEvent:
    """Observer-only: which agent performed a sensing readout this tick."""
    sensed: chex.Array       # bool[N_max]


def sense_occupancy(static: StaticWorldParams, params: WorldParams, state: World,
                    r: int) -> tuple[jax.Array, jax.Array, SenseEvent]:
    """Ground-truth local windows for ALL agents: ``(walls, valid, event)``.

    ``walls[i]``  — bool[2r+1, 2r+1], the wall layer around agent i (truth in range).
    ``valid[i]``  — bool[2r+1, 2r+1], cell is inside the runtime arena (False ⇒ the
                    matching walls cell is padding, not world truth).
    Cells beyond the runtime bounds read False/invalid; dead agents read all-invalid.
    """
    side = 2 * r + 1
    wall_p = jnp.pad(state.wall, r)
    rr = jnp.arange(static.h_max + 2 * r)[:, None]
    cc = jnp.arange(static.w_max + 2 * r)[None, :]
    inb = ((rr >= r) & (rr < params.h + r) & (cc >= r) & (cc < params.w + r))

    def one(i):
        p = state.agent_pos[i]
        w = jax.lax.dynamic_slice(wall_p, (p[0], p[1]), (side, side))
        v = jax.lax.dynamic_slice(inb, (p[0], p[1]), (side, side))
        v = v & state.agent_alive[i]
        return w & v, v

    walls, valid = jax.vmap(one)(jnp.arange(static.n_max))
    return walls, valid, SenseEvent(sensed=state.agent_alive)


def sense_entities(static: StaticWorldParams, params: WorldParams, state: World,
                   r: int) -> jax.Array:
    """bool[N_max, 2r+1, 2r+1] — an ALIVE entity (agent or body, excluding self)
    occupies the cell, within agent i's window. Truth strictly within range."""
    side = 2 * r + 1
    occ = jnp.zeros((static.h_max, static.w_max), jnp.int32)
    apos, bpos = state.agent_pos, state.body_pos
    occ = occ.at[apos[:, 0], apos[:, 1]].add(state.agent_alive.astype(jnp.int32))
    occ = occ.at[bpos[:, 0], bpos[:, 1]].add(state.body_alive.astype(jnp.int32))
    occ_p = jnp.pad(occ, r)

    def one(i):
        p = state.agent_pos[i]
        w = jax.lax.dynamic_slice(occ_p, (p[0], p[1]), (side, side))
        w = w.at[r, r].add(-state.agent_alive[i].astype(jnp.int32))   # exclude self
        return (w > 0) & state.agent_alive[i]

    return jax.vmap(one)(jnp.arange(static.n_max))
