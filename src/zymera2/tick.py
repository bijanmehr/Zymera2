"""The normative tick (spec §14/C5): the composed order and key tree live HERE, in the
package — two harnesses may not disagree about either. ``run_episode`` is the reference
harness (rollout); anything reimplementing the loop must match its golden test.

Tick t (state S_t, mail/adjacency from t-1):
  1. obs_i  = assemble(sense(S_t), mail_i, delivered_row_i, statics)   [the ONLY agent input]
  2. (action_i, outbox_i, astate_i') = policy(obs_i, astate_i, k_agent(t,i))
  3. (S_{t+1}, kernel_events) = world_step(S_t, actions, body_actions, k_world(t))
  4. geom   = {dist, crossings, potential}(S_{t+1})
  5. (mail', delivered', cstate', delivery_events) = channel.deliver(geom, outboxes, k_chan(t))

Key tree (spec §14/C6 — fold_in only, no sequential splits):
  k_agent(t, i) = fold_in(fold_in(fold_in(base, 1), t), i)
  k_world(t)    = fold_in(fold_in(base, 2), t)
  k_chan(t)     = fold_in(fold_in(base, 3), t)   (channel folds its own tick again)

Observation contract (G8): obs_i is EXACTLY {patch, patch_valid, entities, mail,
adj_row, statics} with statics = [h, w] — the whitelist test pins these keys.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from .sensing import sense_entities, sense_occupancy
from .geometry import pairwise_dist, wall_crossings
from .typing import StaticWorldParams, World, WorldParams
from .world import world_step

OBS_KEYS = ("patch", "patch_valid", "entities", "mail", "adj_row", "statics")


class TickCarry(NamedTuple):
    state: World
    astates: Any
    mail: jax.Array
    delivered: jax.Array
    cstate: Any


def assemble_obs(static: StaticWorldParams, params: WorldParams, state: World,
                 mail: jax.Array, delivered: jax.Array, sense_r: int) -> dict:
    """The four ingredients, for all agents (dict of stacked arrays, G8-whitelisted)."""
    walls, valid, sev = sense_occupancy(static, params, state, sense_r)
    ents = sense_entities(static, params, state, sense_r)
    statics = jnp.stack([jnp.int32(params.h), jnp.int32(params.w)])
    n = static.n_max
    return {"patch": walls, "patch_valid": valid, "entities": ents,
            "mail": mail, "adj_row": delivered,
            "statics": jnp.broadcast_to(statics, (n, 2))}, sev


def run_episode(static: StaticWorldParams, params: WorldParams, state0: World,
                policy: Callable, astates0: Any, topology, channel, payload_spec,
                body_controller: Callable, horizon: int, base_key: jax.Array,
                sense_r: int):
    """Reference harness: scan the normative tick for ``horizon`` steps.

    ``policy(obs_i, astate_i, key) -> (action, outbox, astate_i')`` (PolicyFn, vmapped).
    ``body_controller(state, t) -> i32[M_max]`` (task-layer NPC drive; observer of state).
    Returns (final TickCarry, trajectory dict of stacked per-tick records).
    """
    n = static.n_max
    cstate0 = channel.init(topology, static, payload_spec)
    mail0 = jnp.zeros((n, n) + tuple(payload_spec.shape), payload_spec.dtype)
    delivered0 = jnp.zeros((n, n), bool)
    k_agents = jax.random.fold_in(base_key, 1)
    k_world = jax.random.fold_in(base_key, 2)
    k_chan = jax.random.fold_in(base_key, 3)

    def tick(carry: TickCarry, t):
        obs, sev = assemble_obs(static, params, carry.state, carry.mail,
                                carry.delivered, sense_r)
        kt = jax.random.fold_in(k_agents, t)
        akeys = jax.vmap(lambda i: jax.random.fold_in(kt, i))(jnp.arange(n))
        actions, outboxes, astates = jax.vmap(policy)(obs, carry.astates, akeys)
        bacts = body_controller(carry.state, t)
        state, kev = world_step(static, params, carry.state, actions, bacts,
                                jax.random.fold_in(k_world, t))
        geom = {"dist": pairwise_dist(state),
                "crossings": wall_crossings(static, state)}
        geom["potential"] = topology.adjacency(geom)
        mail, delivered, cstate, dev = channel.deliver(
            geom, outboxes, carry.cstate, jax.random.fold_in(k_chan, t))
        rec = {"agent_pos": state.agent_pos, "body_pos": state.body_pos,
               "moved": kev.moved, "blocked": kev.blocked,
               "captured": kev.captured, "sensed": sev.sensed,
               "delivered": delivered, "potential": geom["potential"]}
        return TickCarry(state, astates, mail, delivered, cstate), rec

    carry0 = TickCarry(state0, astates0, mail0, delivered0, cstate0)
    return jax.lax.scan(tick, carry0, jnp.arange(horizon))
