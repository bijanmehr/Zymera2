"""The bridge (spec §8, corrected §14/C2–C3, §14.2): topology + a delayed, lossy,
occlusion-aware channel. Transports opaque-but-schematized payloads; never reads them.

* ``DiskTopology`` — who COULD talk: effective distance ``d_eff = d + c·crossings ≤ r``
  (soft occlusion; ``c=0`` disables). Symmetric by construction; self-loops excluded.
* ``DelayDropoutChannel`` — who DID talk: payloads enqueue at ``t``, pop after ``delay``
  ticks, and are gated by adjacency AND per-edge dropout at DELIVERY time (spec §14.2).
  Per-edge keys via ``fold_in`` (C6); dropout is symmetric per undirected edge.
  ``mail[i, j] = payload authored by j, delay ticks ago, iff delivered[i, j]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

import chex

from .typing import PayloadSpec, StaticWorldParams


@chex.dataclass(frozen=True)
class DeliveryEvents:
    """Observer-only record of what the channel did this tick (spec §14/C4: delivery
    events ride the channel's own return, not the world's kernel record)."""
    delivered: chex.Array    # bool[N,N] — receiver i got sender j's payload


@dataclass(frozen=True)
class DiskTopology:
    comm_r: float = 5.0
    occlusion_c: float = 0.0

    def adjacency(self, geom: dict) -> jax.Array:
        d_eff = geom["dist"] + self.occlusion_c * geom["crossings"]
        n = d_eff.shape[0]
        return (d_eff <= self.comm_r) & ~jnp.eye(n, dtype=bool)


@dataclass(frozen=True)
class ChannelState:
    ring: jax.Array          # [delay+1, N, N?, *leaf] queued payloads (per sender)
    t: jax.Array             # i32 scalar — channel clock


@dataclass(frozen=True)
class DelayDropoutChannel:
    delay: int = 0           # ticks between authoring and delivery
    dropout: float = 0.0     # per-undirected-edge loss probability at delivery time

    def init(self, topology, static: StaticWorldParams,
             payload_spec: PayloadSpec) -> ChannelState:
        ring = jnp.zeros((self.delay + 1, static.n_max) + tuple(payload_spec.shape),
                         payload_spec.dtype)
        return ChannelState(ring=ring, t=jnp.zeros((), jnp.int32))

    def deliver(self, geom: dict, payloads: jax.Array, cstate: ChannelState,
                key: jax.Array):
        """-> (mail [N,N,*leaf], delivered bool[N,N], cstate', events). Payload contents
        are never branched on (opacity, G5) — they are stored, popped, and masked only."""
        n = payloads.shape[0]
        D = self.delay + 1
        slot = cstate.t % D
        ring = jax.lax.dynamic_update_index_in_dim(cstate.ring, payloads, slot, 0)
        pop = jax.lax.dynamic_index_in_dim(ring, (cstate.t - self.delay) % D, 0,
                                           keepdims=False)          # [N,*leaf]
        ready = cstate.t >= self.delay                               # ring warm-up
        potential = geom["potential"]                                # bool[N,N], from Topology
        # per-undirected-edge dropout, delivery-time key: fold_in(edge_id ∘ tick)
        ek = jax.random.fold_in(key, cstate.t)
        iu = jnp.triu_indices(n, 1)
        edge_ids = jnp.arange(n * n).reshape(n, n)
        draw = jax.vmap(lambda e: jax.random.bernoulli(jax.random.fold_in(ek, e),
                                                       self.dropout))(edge_ids[iu])
        drop = jnp.zeros((n, n), bool).at[iu].set(draw)
        drop = drop | drop.T
        delivered = potential & ~drop & ready
        leaf_dims = pop.ndim - 1                                     # payload leaf rank
        mask = delivered.reshape(delivered.shape + (1,) * leaf_dims)
        mail = jnp.where(mask, pop[None, ...], 0)                    # [rcv, snd, *leaf]
        events = DeliveryEvents(delivered=delivered)
        return mail, delivered, ChannelState(ring=ring, t=cstate.t + 1), events
