"""zymera2 contract — the single source of every signature, shape, and dtype.

This module encodes the corrected world contract (design spec §14.1). Nothing else in the
package (or the docs) restates these signatures — they are imported from here. No behavior
lives here: P0 is the contract + validation only; implementations arrive in P1 (world core),
P3 (comms bridge). Protocol method bodies are ``...``.

Conventions
-----------
* Two-tier params: :class:`StaticWorldParams` = hashable compile-time caps (recompile across
  size tiers); :class:`WorldParams` = runtime values passed per call (vmappable).
* Arrays are padded to the static caps and alive-masked; ``arena`` masks non-rectangular shapes.
* No RNG, rewards, done, or comms state live in :class:`World` (spec §3).
* RNG discipline (spec §14/C6): per-entity / per-edge keys via ``jax.random.fold_in(base, id)``,
  never sequential ``split`` — so a counterfactual edit outside an agent's cone cannot perturb
  unrelated draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Protocol, Tuple, runtime_checkable

import chex


# =============================================================================
# Action space (spec §14.5) — fixed, so ``3`` is never ambiguous.
# =============================================================================
class ActionId(IntEnum):
    STAY = 0
    N = 1
    E = 2
    S = 3
    W = 4


N_ACTIONS = 5
# Row/col deltas indexed by ActionId (row = axis 0, down positive).
ACTION_DELTAS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))


# =============================================================================
# World state (spec §3) — entity arrays + property layers; a pure JAX pytree.
# =============================================================================
@chex.dataclass(frozen=True)
class World:
    """The simulated world. Immutable pytree, padded to static caps, alive-masked.

    No beliefs, no rewards, no done, no comms state, no RNG — those live above the world.
    """
    agent_pos: chex.Array    # i32[N_max, 2] (row, col)
    agent_alive: chex.Array  # bool[N_max]
    body_pos: chex.Array     # i32[M_max, 2] — non-agent entities (lead, evaders, …)
    body_alive: chex.Array   # bool[M_max]
    body_kind: chex.Array    # i32[M_max] — task-defined kind tag; the world only compares kinds
    wall: chex.Array         # bool[H_max, W_max] — obstacle layer
    arena: chex.Array        # bool[H_max, W_max] — validity mask → non-rectangular arenas
    step_count: chex.Array   # i32 scalar


# =============================================================================
# Parameters — two-tier (spec §4).
# =============================================================================
@dataclass(frozen=True)
class StaticWorldParams:
    """Compile-time caps; hashable → changing these recompiles. ``rules`` is an immutable
    ordered tuple of :class:`InteractionRule` (spec §14/C8 — no mutable registry)."""
    h_max: int
    w_max: int
    n_max: int
    m_max: int
    rules: tuple = ()


@dataclass(frozen=True)
class WorldParams:
    """Runtime values passed per ``world_step`` call — vmappable, sweepable without recompile.

    Holds world-side quantities only: dimensions and the sensing radius. The communication
    radius is a *bridge* parameter (Topology config, P3) — comms is not a world property.
    """
    h: int
    w: int
    n: int
    m: int
    sense_r: int


@dataclass(frozen=True)
class PayloadSpec:
    """dm_env-style spec for an opaque comm payload (spec §14/C2): the bridge transports
    ``[N_max, *shape]`` arrays of ``dtype`` without interpreting their contents. Declaring a
    payload's *schema* is static (a re-baseline event); the *values* are agents' business."""
    shape: tuple
    dtype: Any


# =============================================================================
# Events — the stable observer-only kernel (spec §14/C4). Dense masked arrays, never a
# count-capped stream. Per-rule event extensions are bundled with each rule, not here.
# =============================================================================
@chex.dataclass(frozen=True)
class KernelEvents:
    """Typed, fixed-shape record of what the world did this tick. Observer-only — agents
    never receive events; the task layer computes rewards/metrics/ledger from them.

    Entity index convention (matches the unified conflict pass, C1): agents occupy global
    indices ``0..N_max-1``, bodies ``N_max..E_max-1``, with ``E_max = N_max + M_max``.
    Sensing events are NOT emitted here — sensing is a harness-side readout returning its
    own ``SenseEvent`` (P2), because ``world_step`` never performs sensing (spec §14/C4).
    Delivery events ride ``Channel.deliver``'s own return, not this kernel record.
    """
    moved: chex.Array     # bool[E_max]        — entity e took a non-STAY move that landed
    blocked: chex.Array   # bool[E_max]        — entity e's intended move was refused
    conflict: chex.Array  # bool[E_max, E_max] — e,f contested the same cell (e yielded)
    captured: chex.Array  # bool[M_max]        — body j flipped captured this tick


# Channel-owned pytree (ring buffers, stamps). Opaque to the world; concrete type in P3.
ChannelState = Any


# =============================================================================
# Protocols — the swappable seams. Bodies are ``...``; implementations arrive later.
# =============================================================================
@runtime_checkable
class Generator(Protocol):
    """Builds a fresh world instance from a config + key (spec §5; parameterized, jit-safe)."""
    def __call__(self, gparams: Any, key: chex.Array) -> Tuple[WorldParams, World]: ...


@runtime_checkable
class InteractionRule(Protocol):
    """Pure contact mechanic run inside ``world_step`` after movement (spec §4b). Reads
    positions/kinds only — never payloads or beliefs. Returns updated state + kernel events;
    a rule may also emit its own bundled event extension (P1)."""
    def apply(self, static: StaticWorldParams, params: WorldParams,
              state: World, events: KernelEvents) -> Tuple[World, KernelEvents]: ...


@runtime_checkable
class Topology(Protocol):
    """Who COULD talk this tick, from world geometry only (spec §8). ``geom`` is the
    geometry-query bundle; returns ``bool[N_max, N_max]`` potential adjacency."""
    def adjacency(self, geom: Any) -> chex.Array: ...


@runtime_checkable
class Channel(Protocol):
    """The radio (spec §8, corrected §14/C2–C3). Transports opaque payloads under delay,
    per-edge dropout, and occlusion; delivery-time gating (spec §14.2)."""
    def init(self, topology: Topology, static: StaticWorldParams,
             payload_spec: PayloadSpec) -> ChannelState: ...

    def deliver(self, geom: Any, payloads: Any, cstate: ChannelState,
                key: chex.Array) -> Tuple[Any, chex.Array, ChannelState, KernelEvents]:
        """-> (mail: [N_max,N_max,*leaf], delivered: bool[N_max,N_max], cstate', events)."""
        ...


@runtime_checkable
class PolicyFn(Protocol):
    """The agent contract (spec §14/C7) — versioned in-package though implemented above.
    ``obs`` is the harness-assembled observation; ``state`` is the agent's carried pytree."""
    def __call__(self, obs: Any, state: Any, key: chex.Array) -> Tuple[Any, Any, Any]: ...


# The world step (spec §14/C1) — bodies actuate via ``body_actions``; unified conflict pass.
WorldStepFn = Callable[
    [StaticWorldParams, WorldParams, World, chex.Array, chex.Array, chex.Array],
    Tuple[World, KernelEvents],
]


__all__ = [
    "ActionId", "N_ACTIONS", "ACTION_DELTAS",
    "World", "StaticWorldParams", "WorldParams", "PayloadSpec",
    "KernelEvents", "ChannelState",
    "Generator", "InteractionRule", "Topology", "Channel", "PolicyFn", "WorldStepFn",
]
