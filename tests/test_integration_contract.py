"""Contract integration dry-run: does the P0 contract COMPOSE under real usage?

Wires the exact architecture-shaped pattern the lab will use — PolicyFn agents
(``(obs, state, key) -> (action, outbox, state')``), a WorldStepFn-conforming step,
Topology/Channel stubs per the Protocols, obs assembled from the four ingredients —
under ``jit`` + ``lax.scan`` + ``vmap`` with the fold_in key discipline, at padded
caps with runtime sizes below them. No physics correctness is claimed here (that is
P1); this test exists to surface contract errors: pytree registration, static/runtime
param handling, shape stability, determinism, recompile behavior.
"""
import jax
import jax.numpy as jnp
import pytest

from zymera2 import typing as zt
from zymera2.validate import validate_params

CAPS = zt.StaticWorldParams(h_max=32, w_max=32, n_max=10, m_max=4, rules=())
E_MAX = CAPS.n_max + CAPS.m_max
PAYLOAD = zt.PayloadSpec(shape=(4,), dtype=jnp.float32)
T = 8  # ticks per episode


# ---------------------------------------------------------------- world stubs
def _gen(params: zt.WorldParams, key: jax.Array) -> zt.World:
    """Minimal Generator-conforming builder: padded arrays, runtime region active."""
    wall = jnp.zeros((CAPS.h_max, CAPS.w_max), bool)
    wall = wall.at[5, :10].set(True)  # a wall segment inside the runtime region
    rr = jnp.arange(CAPS.h_max)[:, None]
    cc = jnp.arange(CAPS.w_max)[None, :]
    arena = (rr < params.h) & (cc < params.w)
    k1, k2 = jax.random.fold_in(key, 1), jax.random.fold_in(key, 2)
    apos = jnp.stack([jax.random.randint(k1, (CAPS.n_max,), 0, params.h),
                      jax.random.randint(k1, (CAPS.n_max,), 0, params.w)], -1).astype(jnp.int32)
    bpos = jnp.stack([jax.random.randint(k2, (CAPS.m_max,), 0, params.h),
                      jax.random.randint(k2, (CAPS.m_max,), 0, params.w)], -1).astype(jnp.int32)
    return zt.World(
        agent_pos=apos, agent_alive=jnp.arange(CAPS.n_max) < params.n,
        body_pos=bpos, body_alive=jnp.arange(CAPS.m_max) < params.m,
        body_kind=jnp.zeros((CAPS.m_max,), jnp.int32),
        wall=wall, arena=arena, step_count=jnp.zeros((), jnp.int32))


def _step(static: zt.StaticWorldParams, params: zt.WorldParams, state: zt.World,
          actions: jax.Array, body_actions: jax.Array, key: jax.Array):
    """WorldStepFn-conforming stub: clip-move agents+bodies, refuse walls, emit events."""
    deltas = jnp.array(zt.ACTION_DELTAS, jnp.int32)

    def move(pos, alive, acts):
        prop = pos + deltas[acts]
        prop = jnp.clip(prop, 0, jnp.array([static.h_max - 1, static.w_max - 1]))
        inside = (prop[:, 0] < params.h) & (prop[:, 1] < params.w)
        ok = inside & ~state.wall[prop[:, 0], prop[:, 1]] & alive
        return jnp.where(ok[:, None], prop, pos), ok

    apos, aok = move(state.agent_pos, state.agent_alive, actions)
    bpos, bok = move(state.body_pos, state.body_alive, body_actions)
    moved = jnp.concatenate([aok & (actions != zt.ActionId.STAY),
                             bok & (body_actions != zt.ActionId.STAY)])
    blocked = jnp.concatenate([~aok & state.agent_alive, ~bok & state.body_alive])
    events = zt.KernelEvents(
        moved=moved, blocked=blocked,
        conflict=jnp.zeros((E_MAX, E_MAX), bool),
        captured=jnp.zeros((CAPS.m_max,), bool))
    new = state.replace(agent_pos=apos, body_pos=bpos, step_count=state.step_count + 1)
    return new, events


# ---------------------------------------------------------------- bridge stubs
def _adjacency(state: zt.World, comm_r: int) -> jax.Array:
    d = jnp.abs(state.agent_pos[:, None, :] - state.agent_pos[None, :, :]).max(-1)
    alive = state.agent_alive
    return (d <= comm_r) & alive[:, None] & alive[None, :]


def _deliver(potential: jax.Array, payloads: jax.Array, key: jax.Array):
    """Channel-conforming stub: per-edge dropout, delivered ⊆ potential, opaque mail."""
    drop = jax.random.bernoulli(key, 0.2, potential.shape)
    delivered = potential & ~drop
    mail = jnp.where(delivered[:, :, None], payloads[None, :, :], 0.0)  # [rcv, snd, *leaf]
    return mail, delivered


# ---------------------------------------------------------------- agent (PolicyFn shape)
def _policy(obs, astate, key):
    """PolicyFn-conforming random agent: consumes the 4-ingredient obs, authors an
    outbox of PAYLOAD spec, carries a state. Mirrors the L3 policy contract exactly."""
    a = jax.random.randint(key, (), 0, zt.N_ACTIONS)
    outbox = jnp.full(PAYLOAD.shape, obs["patch"].sum(), PAYLOAD.dtype)
    return a, outbox, astate + 1


def _obs(state: zt.World, mail: jax.Array, delivered: jax.Array, i: jax.Array):
    """The four ingredients, per agent: sensed patch, mail row, delivered row, statics."""
    r, c = state.agent_pos[i, 0], state.agent_pos[i, 1]
    patch = jax.lax.dynamic_slice(
        jnp.pad(state.wall, 1), (r, c), (3, 3))  # 3x3 truth window (sense_r=1)
    return {"patch": patch, "mail": mail[i], "adj_row": delivered[i],
            "statics": jnp.array([CAPS.h_max, CAPS.w_max], jnp.int32)}


# ---------------------------------------------------------------- the composed tick
def _episode(params: zt.WorldParams, key: jax.Array):
    state0 = _gen(params, jax.random.fold_in(key, 0))
    astates0 = jnp.zeros((CAPS.n_max,), jnp.int32)
    mail0 = jnp.zeros((CAPS.n_max, CAPS.n_max) + PAYLOAD.shape, PAYLOAD.dtype)
    delivered0 = jnp.zeros((CAPS.n_max, CAPS.n_max), bool)

    def tick(carry, t):
        state, astates, mail, delivered = carry
        kt = jax.random.fold_in(key, t + 1)
        # agents decide (vmapped, per-agent fold_in keys) from the assembled obs
        ids = jnp.arange(CAPS.n_max)
        obs = jax.vmap(lambda i: _obs(state, mail, delivered, i))(ids)
        akeys = jax.vmap(lambda i: jax.random.fold_in(kt, i))(ids)
        actions, outboxes, astates = jax.vmap(_policy)(obs, astates, akeys)
        body_actions = jnp.full((CAPS.m_max,), int(zt.ActionId.STAY), jnp.int32)
        # world step → geometry → topology → deliver
        state, events = _step(CAPS, params, state, actions, body_actions,
                              jax.random.fold_in(kt, 1000))
        potential = _adjacency(state, comm_r=3)
        mail, delivered = _deliver(potential, outboxes, jax.random.fold_in(kt, 2000))
        ok = jnp.all(~delivered | potential)  # delivered ⊆ potential, every tick
        return (state, astates, mail, delivered), (events.moved.any(), ok)

    (state, astates, mail, delivered), (any_moved, ok) = jax.lax.scan(
        tick, (state0, astates0, mail0, delivered0), jnp.arange(T))
    return state, astates, ok.all(), any_moved.any()


def test_episode_composes_under_jit_scan_vmap():
    """The architecture-shaped loop runs end-to-end without raising, with params passed
    as a RUNTIME argument (the spec's 'sweepable without recompile' claim)."""
    params = zt.WorldParams(h=16, w=16, n=4, m=1, sense_r=1)
    validate_params(CAPS, params)
    run = jax.jit(_episode)
    state, astates, subset_ok, moved = run(params, jax.random.PRNGKey(0))
    assert int(state.step_count) == T
    assert bool(subset_ok), "delivered ⊄ potential"
    assert bool(moved), "no agent ever moved — step stub or key plumbing broken"
    assert int(astates[0]) == T  # agent state carried through every tick


def test_bit_exact_determinism_cpu():
    params = zt.WorldParams(h=16, w=16, n=4, m=1, sense_r=1)
    run = jax.jit(_episode)
    s1, a1, _, _ = run(params, jax.random.PRNGKey(7))
    s2, a2, _, _ = run(params, jax.random.PRNGKey(7))
    assert (s1.agent_pos == s2.agent_pos).all() and (a1 == a2).all()


def test_size_sweep_does_not_recompile():
    """Two runtime sizes under one set of caps must reuse one compilation — the
    two-tier params claim. Skipped if the private cache-size API is unavailable."""
    run = jax.jit(_episode)
    if not hasattr(run, "_cache_size"):
        pytest.skip("jit cache-size API unavailable on this JAX version")
    run(zt.WorldParams(h=16, w=16, n=4, m=1, sense_r=1), jax.random.PRNGKey(0))
    run(zt.WorldParams(h=24, w=20, n=7, m=2, sense_r=1), jax.random.PRNGKey(0))
    assert run._cache_size() == 1, f"recompiled across runtime sizes: {run._cache_size()}"
