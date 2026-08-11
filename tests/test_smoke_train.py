"""Fitness-for-training smoke test (the ladder's last rung): gradients flow through
obs -> policy across the full composed tick, and REINFORCE improves team coverage.

This certifies the STACK is trainable — it is deliberately NOT an env-correctness
test (RL improving proves little about semantics; the property/oracle/differential
suites carry correctness). Reward is computed OBSERVER-side from trajectory
positions — task-layer style; no reward exists inside the world.
"""
import jax
import jax.numpy as jnp
import numpy as np

from zymera2 import typing as zt
from zymera2.comms import DelayDropoutChannel, DiskTopology
from zymera2.tick import run_episode
from zymera2.worldgen import GenConfig, generate

STATIC = zt.StaticWorldParams(h_max=12, w_max=12, n_max=4, m_max=1, rules=())
PARAMS = zt.WorldParams(h=12, w=12, n=4, m=0, sense_r=1)
SPEC = zt.PayloadSpec(shape=(1,), dtype=jnp.float32)
TOPO = DiskTopology(comm_r=5.0)
CHAN = DelayDropoutChannel(delay=1, dropout=0.2)
HORIZON = 30
BATCH = 8
FEATS = 9 + 9 + STATIC.n_max + 1          # patch + entities + adj_row + bias


def _features(obs):
    return jnp.concatenate([
        obs["patch"].reshape(-1).astype(jnp.float32),
        obs["entities"].reshape(-1).astype(jnp.float32),
        obs["adj_row"].astype(jnp.float32),
        jnp.ones((1,), jnp.float32)])


def _make_policy(W):
    """PolicyFn: linear logits over the whitelisted obs; logp accumulates in AgentState."""
    def policy(obs, astate, key):
        logits = _features(obs) @ W                                   # [5]
        a = jax.random.categorical(key, logits)
        logp = jax.nn.log_softmax(logits)[a]
        return a, jnp.zeros(SPEC.shape, SPEC.dtype), astate + logp
    return policy


def _coverage(rec):
    """Team coverage over the episode — observer-side scatter, differentiation-free."""
    pos = rec["agent_pos"]                                            # [T,N,2]
    vis = jnp.zeros((STATIC.h_max, STATIC.w_max), bool)
    vis = vis.at[pos[..., 0].reshape(-1), pos[..., 1].reshape(-1)].set(True)
    return vis.sum().astype(jnp.float32)


def _loss(W, world0, keys):
    def one(key):
        carry, rec = run_episode(STATIC, PARAMS, world0, _make_policy(W),
                                 jnp.zeros((STATIC.n_max,), jnp.float32), TOPO, CHAN,
                                 SPEC, lambda s, t: jnp.zeros((STATIC.m_max,), jnp.int32),
                                 HORIZON, key, 1)
        R = _coverage(rec)
        logp = carry.astates.sum()
        return R, logp
    R, logp = jax.vmap(one)(keys)
    adv = jax.lax.stop_gradient(R - R.mean())
    return -(adv * logp).mean() / HORIZON, R.mean()


def test_reinforce_improves_coverage():
    world0 = generate(GenConfig("open"), STATIC, PARAMS, jax.random.PRNGKey(0))
    W = jnp.zeros((FEATS, zt.N_ACTIONS))                              # uniform policy
    grad = jax.jit(jax.value_and_grad(_loss, has_aux=True))
    lr = 0.08
    means = []
    key = jax.random.PRNGKey(123)
    for it in range(120):
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(key, it), i)
                        )(jnp.arange(BATCH))
        (loss, mR), g = grad(W, world0, keys)
        W = W - lr * jnp.clip(g, -1.0, 1.0)
        means.append(float(mR))
    first, last = np.mean(means[:10]), np.mean(means[-10:])
    print(f"\nsmoke-train coverage: first10={first:.1f} last10={last:.1f} "
          f"(of {PARAMS.h * PARAMS.w} cells, horizon {HORIZON})")
    assert np.isfinite(loss), "loss went non-finite — gradient path broken"
    assert last > first + 5.0, (
        f"REINFORCE failed to improve coverage: {first:.1f} -> {last:.1f}")
