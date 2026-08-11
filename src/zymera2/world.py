"""The world step (spec §4, corrected §14/C1): unified movement + conflict pass over
agents AND bodies, then the interaction-rule tuple. Pure, deterministic, jit/scan-safe.

Conflict semantics (part of ``-v0`` behavior):
  * An entity's move is refused if it targets a wall, leaves the runtime arena, or
    the entity is dead (dead entities never move and never block).
  * Cell contests: an entity staying on (or reverted to) its own cell always keeps it;
    among movers contesting a cell, the LOWEST global entity index wins (agents occupy
    indices ``0..N_max-1``, bodies ``N_max..E_max-1`` — agents outrank bodies).
  * Swap-through (A→B's cell while B→A's cell) is refused for both.
  * Losers revert and may displace nobody; resolution iterates to a fixpoint
    (≤ E iterations — a loser becomes a stayer, and stayers never lose).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .typing import (ACTION_DELTAS, ActionId, KernelEvents, N_ACTIONS,
                     StaticWorldParams, World, WorldParams)


def world_step(static: StaticWorldParams, params: WorldParams, state: World,
               actions: jax.Array, body_actions: jax.Array,
               key: jax.Array) -> tuple[World, KernelEvents]:
    """Advance the world one tick. Conforms to :data:`zymera2.typing.WorldStepFn`."""
    E = static.n_max + static.m_max
    deltas = jnp.array(ACTION_DELTAS, jnp.int32)

    pos = jnp.concatenate([state.agent_pos, state.body_pos])            # [E,2]
    alive = jnp.concatenate([state.agent_alive, state.body_alive])      # [E]
    acts = jnp.concatenate([actions, body_actions]).astype(jnp.int32)
    acts = jnp.clip(acts, 0, N_ACTIONS - 1)
    # Padding-invariance guard: dead entities act as STAY at their own cell.
    acts = jnp.where(alive, acts, ActionId.STAY)
    pos = jnp.clip(pos, 0, jnp.array([static.h_max - 1, static.w_max - 1]))

    prop = pos + deltas[acts]
    inb = ((prop[:, 0] >= 0) & (prop[:, 1] >= 0)
           & (prop[:, 0] < params.h) & (prop[:, 1] < params.w))
    gp = jnp.clip(prop, 0, jnp.array([static.h_max - 1, static.w_max - 1]))
    ok = alive & inb & ~state.wall[gp[:, 0], gp[:, 1]] & state.arena[gp[:, 0], gp[:, 1]]
    intended = jnp.where(ok[:, None], gp, pos)

    lower = jnp.tril(jnp.ones((E, E), bool), -1)        # lower[i,j] ⇔ j < i
    eye = jnp.eye(E, dtype=bool)
    pair_alive = alive[:, None] & alive[None, :] & ~eye

    def resolve(carry, _):
        cur, conflict = carry
        same = (cur[:, None, :] == cur[None, :, :]).all(-1) & pair_alive
        stay = (cur == pos).all(-1)                                    # [E]
        # i loses a contested cell to j if: j is a stayer and i is not, or both move and j<i.
        lose_cell = (same & (stay[None, :] & ~stay[:, None])).any(1) \
            | (same & (~stay[None, :] & ~stay[:, None]) & lower).any(1)
        # swap-through: both refused
        swap = ((cur[:, None, :] == pos[None, :, :]).all(-1)
                & (cur[None, :, :] == pos[:, None, :]).all(-1)
                & pair_alive & ~stay[:, None] & ~stay[None, :])
        lose = lose_cell | swap.any(1)
        conflict = conflict | (same & ~eye) | swap
        cur = jnp.where(lose[:, None], pos, cur)
        return (cur, conflict), None

    (final, conflict), _ = jax.lax.scan(
        resolve, (intended, jnp.zeros((E, E), bool)), None, length=E)

    moved = (final != pos).any(-1)
    blocked = (acts != ActionId.STAY) & alive & ~moved
    events = KernelEvents(moved=moved, blocked=blocked, conflict=conflict,
                          captured=jnp.zeros((static.m_max,), bool))

    new = state.replace(agent_pos=final[: static.n_max],
                        body_pos=final[static.n_max:],
                        step_count=state.step_count + 1)
    # Interaction rules: static ordered tuple (spec §14/C8) — unrolled at trace time.
    for rule in static.rules:
        new, events = rule.apply(static, params, new, events)
    return new, events
