import jax
import jax.numpy as jnp
import numpy as np

from zymera2 import typing as zt
from zymera2.comms import DelayDropoutChannel, DiskTopology
from zymera2.geometry import pairwise_dist, wall_crossings
from zymera2.worldgen import GenConfig, generate

STATIC = zt.StaticWorldParams(h_max=16, w_max=16, n_max=6, m_max=2, rules=())
PARAMS = zt.WorldParams(h=12, w=12, n=4, m=0, sense_r=1)
SPEC = zt.PayloadSpec(shape=(3,), dtype=jnp.float32)


def _world(agent_pos):
    w = generate(GenConfig("open"), STATIC, PARAMS, jax.random.PRNGKey(0))
    return w.replace(agent_pos=jnp.asarray(agent_pos, jnp.int32))


POS = [[2, 2], [2, 5], [9, 2], [9, 9], [0, 0], [0, 0]]   # last two are dead pad rows


def test_pairwise_dist_matches_numpy_oracle():
    d = np.asarray(pairwise_dist(_world(POS)))
    p = np.array(POS)
    for i in range(4):
        for j in range(4):
            assert d[i, j] == np.abs(p[i] - p[j]).max()
    assert np.isinf(d[0, 4]) and np.isinf(d[5, 5])       # dead rows unreachable


def test_wall_crossings_zero_without_walls_and_counts_with():
    w = _world(POS)
    k0 = np.asarray(wall_crossings(STATIC, w))
    assert (k0[:4, :4] == 0).all()
    # wall column between agent 0 (2,2) and agent 1 (2,5): cells (2,3),(2,4)
    w2 = w.replace(wall=w.wall.at[2, 3].set(True).at[2, 4].set(True))
    k = np.asarray(wall_crossings(STATIC, w2))
    assert k[0, 1] == 2 and k[1, 0] == 2
    assert k[0, 4] == 0                                   # dead row → 0


def test_topology_symmetric_no_self_and_occlusion_cuts():
    w = _world(POS)
    geom = {"dist": pairwise_dist(w), "crossings": wall_crossings(STATIC, w)}
    adj = np.asarray(DiskTopology(comm_r=4.0).adjacency(geom))
    assert adj[0, 1] and adj[1, 0] and not adj.diagonal().any()
    assert not adj[0, 3]                                  # far pair out of range
    # occlusion: the 2-wall cut pushes d_eff over r
    w2 = w.replace(wall=w.wall.at[2, 3].set(True).at[2, 4].set(True))
    geom2 = {"dist": pairwise_dist(w2), "crossings": wall_crossings(STATIC, w2)}
    adj2 = np.asarray(DiskTopology(comm_r=4.0, occlusion_c=1.0).adjacency(geom2))
    assert not adj2[0, 1], "occlusion should cut the 0-1 link (d_eff = 3 + 2 > 4)"


def _run_channel(ch, n_ticks, payload_fn, potential):
    cs = ch.init(None, STATIC, SPEC)
    out = []
    for t in range(n_ticks):
        payloads = payload_fn(t)
        mail, delivered, cs, ev = ch.deliver({"potential": potential}, payloads, cs,
                                             jax.random.PRNGKey(42))
        out.append((np.asarray(mail), np.asarray(delivered)))
    return out


def test_delay_exactness():
    """A payload authored at tick t arrives exactly at t+delay — never earlier/later."""
    n = STATIC.n_max
    potential = jnp.ones((n, n), bool) & ~jnp.eye(n, dtype=bool)
    ch = DelayDropoutChannel(delay=2, dropout=0.0)
    stamp = lambda t: jnp.full((n,) + SPEC.shape, float(t + 1), SPEC.dtype)
    out = _run_channel(ch, 5, stamp, potential)
    for t, (mail, delivered) in enumerate(out):
        if t < 2:
            assert not delivered.any(), f"delivered during warm-up at t={t}"
        else:
            assert delivered.any()
            got = mail[0, 1, 0]                       # receiver 0 ← sender 1
            assert got == float(t - 2 + 1), f"t={t}: got stamp {got}, want {t-1}"


def test_delivered_subset_of_potential_under_dropout():
    n = STATIC.n_max
    potential = jnp.zeros((n, n), bool).at[0, 1].set(True).at[1, 0].set(True)
    ch = DelayDropoutChannel(delay=0, dropout=0.5)
    for t, (mail, delivered) in enumerate(_run_channel(
            ch, 20, lambda t: jnp.ones((n,) + SPEC.shape, SPEC.dtype), potential)):
        assert (delivered <= np.asarray(potential)).all()
        assert (mail[~delivered] == 0).all()          # mail strictly masked by delivered


def test_dropout_rate_matches_closed_form():
    n = STATIC.n_max
    potential = jnp.ones((n, n), bool) & ~jnp.eye(n, dtype=bool)
    ch = DelayDropoutChannel(delay=0, dropout=0.3)
    cs = ch.init(None, STATIC, SPEC)
    kept = total = 0
    for t in range(400):
        _, delivered, cs, _ = ch.deliver({"potential": potential},
                                         jnp.ones((n,) + SPEC.shape, SPEC.dtype),
                                         cs, jax.random.PRNGKey(9))
        d = np.asarray(delivered)
        iu = np.triu_indices(n, 1)
        kept += d[iu].sum(); total += len(iu[0])
    rate = kept / total
    assert abs(rate - 0.7) < 0.03, f"delivery rate {rate:.3f} vs closed-form 0.7"


def test_dropout_symmetric_per_edge():
    n = STATIC.n_max
    potential = jnp.ones((n, n), bool) & ~jnp.eye(n, dtype=bool)
    ch = DelayDropoutChannel(delay=0, dropout=0.5)
    cs = ch.init(None, STATIC, SPEC)
    for t in range(10):
        _, delivered, cs, _ = ch.deliver({"potential": potential},
                                         jnp.ones((n,) + SPEC.shape, SPEC.dtype),
                                         cs, jax.random.PRNGKey(1))
        d = np.asarray(delivered)
        assert (d == d.T).all(), "per-undirected-edge dropout must be symmetric"


def test_opacity_content_swap():
    """G5: scrambling payload VALUES must not change the delivered graph."""
    n = STATIC.n_max
    potential = jnp.ones((n, n), bool) & ~jnp.eye(n, dtype=bool)
    ch = DelayDropoutChannel(delay=1, dropout=0.4)
    a = _run_channel(ch, 6, lambda t: jnp.full((n,) + SPEC.shape, 1.0, SPEC.dtype), potential)
    b = _run_channel(ch, 6, lambda t: jnp.full((n,) + SPEC.shape, -9e9, SPEC.dtype), potential)
    for (_, da), (_, db) in zip(a, b):
        assert (da == db).all(), "delivery branched on payload content"
