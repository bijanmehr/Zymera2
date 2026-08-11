"""The leakage suite (spec §10.2 / §14.3–14.4): T1–T5 + planted-leak mutation tests.

Scope of the claim (three-row taxonomy, spec §14.3): these certify EXECUTION-TIME
transport — each agent's decision-time inputs are measurable w.r.t. its information
cone. Training-time privilege (CTDE) is out of scope by design and never certified here.
"""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from zymera2 import typing as zt
from zymera2.comms import DelayDropoutChannel, DiskTopology
from zymera2.tick import OBS_KEYS, assemble_obs, run_episode
from zymera2.worldgen import GenConfig, generate

STATIC = zt.StaticWorldParams(h_max=16, w_max=16, n_max=4, m_max=1, rules=())
PARAMS = zt.WorldParams(h=12, w=12, n=4, m=1, sense_r=1)
SPEC = zt.PayloadSpec(shape=(3,), dtype=jnp.float32)
TOPO = DiskTopology(comm_r=4.0)
CHAN = DelayDropoutChannel(delay=1, dropout=0.3)
R = 1
HORIZON = 5


def _policy(obs, astate, key):
    a = jax.random.randint(key, (), 0, zt.N_ACTIONS)
    outbox = jnp.array([obs["patch"].sum(), obs["adj_row"].sum(), 1.0], SPEC.dtype)
    return a, outbox, astate + 1


def _bodies(state, t):
    return jnp.zeros((STATIC.m_max,), jnp.int32)


def _clustered_world():
    """Agents pinned to the top-left 4x4 block; the far corner is outside every cone."""
    w = generate(GenConfig("open"), STATIC, PARAMS, jax.random.PRNGKey(5))
    return w.replace(
        agent_pos=jnp.asarray([[1, 1], [1, 3], [3, 1], [3, 3]], jnp.int32),
        body_pos=jnp.asarray([[2, 2]], jnp.int32))


def _episode(state0, key=jax.random.PRNGKey(11)):
    return run_episode(STATIC, PARAMS, state0, _policy,
                       jnp.zeros((STATIC.n_max,), jnp.int32), TOPO, CHAN, SPEC,
                       _bodies, HORIZON, key, R)


# ------------------------------------------------------------------ T1 cone-nullity
def test_T1_cone_nullity_of_decision_time_obs():
    """A wall edit outside every agent's sense window leaves tick-0 obs bit-identical."""
    w = _clustered_world()
    far = w.replace(wall=w.wall.at[11, 11].set(True))
    mail0 = jnp.zeros((4, 4) + SPEC.shape, SPEC.dtype)
    d0 = jnp.zeros((4, 4), bool)
    o1, _ = assemble_obs(STATIC, PARAMS, w, mail0, d0, R)
    o2, _ = assemble_obs(STATIC, PARAMS, far, mail0, d0, R)
    for k in OBS_KEYS:
        assert (o1[k] == o2[k]).all(), f"obs[{k}] changed from an out-of-cone wall edit"


# ------------------------------------- T2 counterfactual non-interference (bit-exact)
def test_T2_counterfactual_non_interference_full_episode():
    """Flip a wall no rollout can reach (dist > horizon + sense_r from every spawn):
    the ENTIRE trajectory — positions, deliveries, records — must be bit-identical."""
    w = _clustered_world()
    far = w.replace(wall=w.wall.at[11, 11].set(True))
    (_, _, m1, d1, _), r1 = _episode(w)
    (_, _, m2, d2, _), r2 = _episode(far)
    for k in r1:
        assert (np.asarray(r1[k]) == np.asarray(r2[k])).all(), f"trajectory[{k}] diverged"
    assert (np.asarray(m1) == np.asarray(m2)).all() and (np.asarray(d1) == np.asarray(d2)).all()


# ------------------------------------------------------------------ T3 channel ablation
def test_T3_channel_ablation_dropout_one():
    """dropout=1 ⇒ nothing is ever delivered and all mail is zeros; the world runs on."""
    chan = DelayDropoutChannel(delay=1, dropout=1.0)
    (_, _, mail, delivered, _), rec = run_episode(
        STATIC, PARAMS, _clustered_world(), _policy,
        jnp.zeros((STATIC.n_max,), jnp.int32), TOPO, chan, SPEC,
        _bodies, HORIZON, jax.random.PRNGKey(11), R)
    assert not np.asarray(rec["delivered"]).any()
    assert (np.asarray(mail) == 0).all()
    assert np.asarray(rec["moved"]).any()          # the world itself keeps working


# ------------------------------------------------------------- T4 delivered ⊆ potential
def test_T4_delivered_subset_of_potential_every_tick():
    _, rec = _episode(_clustered_world())
    d = np.asarray(rec["delivered"]); p = np.asarray(rec["potential"])
    assert (d <= p).all(), "a delivery happened on a non-potential edge"


# ------------------------------------------------------------------ T5 whitelist guard
def test_T5_obs_whitelist_and_no_global_leaf():
    """obs has EXACTLY the whitelisted keys; no leaf carries a global-map plane."""
    w = _clustered_world()
    obs, _ = assemble_obs(STATIC, PARAMS, w,
                          jnp.zeros((4, 4) + SPEC.shape, SPEC.dtype),
                          jnp.zeros((4, 4), bool), R)
    assert tuple(sorted(obs.keys())) == tuple(sorted(OBS_KEYS))
    for k, v in obs.items():
        assert (STATIC.h_max, STATIC.w_max) != v.shape[-2:], \
            f"obs[{k}] smuggles a full-map plane"
    assert obs["patch"].shape == (4, 3, 3) and obs["statics"].shape == (4, 2)


# ================================================================== MUTATION TESTS
# The suite must CATCH planted leaks (spec §14.3) — each leak below is a known v1-style
# violation; the paired detector predicate must flag it. A suite that never caught a
# planted leak certifies nothing.

def _detector_T1(assemble_fn):
    w = _clustered_world()
    far = w.replace(wall=w.wall.at[3, 5].set(True))   # dist 2 from agent (3,3): outside r=1
    m0 = jnp.zeros((4, 4) + SPEC.shape, SPEC.dtype); d0 = jnp.zeros((4, 4), bool)
    o1 = assemble_fn(w, m0, d0); o2 = assemble_fn(far, m0, d0)
    return all(bool((o1[k] == o2[k]).all()) for k in o1)


def test_mutation_1_statics_smuggles_map_is_caught():
    def leaky(w, m, d):
        obs, _ = assemble_obs(STATIC, PARAMS, w, m, d, R)
        obs["statics"] = jnp.broadcast_to(w.wall, (4,) + w.wall.shape)   # smuggle
        return obs
    obs = leaky(_clustered_world(), jnp.zeros((4, 4) + SPEC.shape, SPEC.dtype),
                jnp.zeros((4, 4), bool))
    caught = any((STATIC.h_max, STATIC.w_max) == v.shape[-2:] for v in obs.values())
    assert caught, "T5 shape scan failed to catch a smuggled global map"


def test_mutation_2_godview_patch_is_caught():
    def leaky(w, m, d):
        obs, _ = assemble_obs(STATIC, PARAMS, w, m, d, R)
        # god-view: fold far-map information into an in-shape channel
        obs = dict(obs); obs["patch"] = obs["patch"] | w.wall.any()
        return obs
    assert not _detector_T1(leaky), "T1 failed to catch a god-view patch"


def test_mutation_3_potential_for_delivered_is_caught():
    """v1 leak L3: adjacency shown to agents = POTENTIAL instead of DELIVERED."""
    chan = DelayDropoutChannel(delay=1, dropout=1.0)
    _, rec = run_episode(STATIC, PARAMS, _clustered_world(), _policy,
                         jnp.zeros((STATIC.n_max,), jnp.int32), TOPO, chan, SPEC,
                         _bodies, HORIZON, jax.random.PRNGKey(11), R)
    leaky_adj = rec["potential"]                      # the swap
    honest_adj = rec["delivered"]
    # detector: under total dropout the agent-visible adjacency must be empty
    assert np.asarray(honest_adj).sum() == 0
    assert np.asarray(leaky_adj).sum() > 0            # the leak IS detectable
    caught = np.asarray(leaky_adj).sum() != np.asarray(honest_adj).sum()
    assert caught, "T3-based detector failed to catch potential-for-delivered"


def test_mutation_4_delay_bypass_is_caught():
    """Mail built from CURRENT outboxes (skipping the ring) breaks delay exactness."""
    n = STATIC.n_max
    potential = jnp.ones((n, n), bool) & ~jnp.eye(n, dtype=bool)
    chan = DelayDropoutChannel(delay=2, dropout=0.0)
    cs = chan.init(TOPO, STATIC, SPEC)
    stamps = []
    for t in range(4):
        payloads = jnp.full((n,) + SPEC.shape, float(t + 1), SPEC.dtype)
        mail, delivered, cs, _ = chan.deliver({"potential": potential}, payloads, cs,
                                              jax.random.PRNGKey(3))
        leaky_mail = jnp.where(delivered[:, :, None], payloads[None, :, :], 0)  # bypass
        stamps.append((float(mail[0, 1, 0]) if bool(delivered[0, 1]) else None,
                       float(leaky_mail[0, 1, 0]) if bool(delivered[0, 1]) else None))
    honest = [s for s, _ in stamps if s is not None]
    leaked = [s for _, s in stamps if s is not None]
    assert honest and honest[0] == 1.0                # first delivery carries t=0's stamp
    assert leaked[0] != 1.0, "delay-bypass mutation not detectable?"
    assert honest != leaked, "delay detector failed to catch the ring bypass"


def test_mutation_5_mail_mask_bypass_is_caught():
    """Mail present on undelivered edges must be flagged (mask must be `delivered`)."""
    n = STATIC.n_max
    potential = jnp.ones((n, n), bool) & ~jnp.eye(n, dtype=bool)
    chan = DelayDropoutChannel(delay=0, dropout=0.5)
    cs = chan.init(TOPO, STATIC, SPEC)
    payloads = jnp.ones((n,) + SPEC.shape, SPEC.dtype)
    mail, delivered, cs, _ = chan.deliver({"potential": potential}, payloads, cs,
                                          jax.random.PRNGKey(4))
    leaky_mail = jnp.where(potential[:, :, None], payloads[None, :, :], 0)  # wrong mask
    honest_bad = np.asarray((mail != 0).any(-1) & ~np.asarray(delivered)).sum()
    leaky_bad = np.asarray((leaky_mail != 0).any(-1) & ~np.asarray(delivered)).sum()
    assert honest_bad == 0
    assert leaky_bad > 0, "mask-bypass mutation not detectable?"


# ------------------------------------------------------------------ composed-tick golden
def _traj_digest(rec) -> str:
    h = hashlib.md5()
    for k in sorted(rec):
        h.update(k.encode())
        h.update(np.ascontiguousarray(np.asarray(rec[k])).tobytes())
    return h.hexdigest()

GOLDEN_V0 = "6d9e0d24a03ee828afbb27464807984f"   # jaxlib 0.11.0 / CPU (goldens are
# per-JAX-version artifacts — regenerate + record on any pinned-version change, §14/C6)


def test_composed_tick_golden_v0():
    """Pins the -v0 composed tick (order + keys + semantics). If this breaks, either
    the behavior version bumps (-v1 + new golden) or the change is a bug."""
    _, rec = _episode(_clustered_world())
    assert _traj_digest(rec) == GOLDEN_V0
