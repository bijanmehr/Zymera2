import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, strategies as st

from zymera2 import typing as zt
from zymera2.world import world_step
from zymera2.worldgen import GenConfig, generate

STATIC = zt.StaticWorldParams(h_max=16, w_max=16, n_max=6, m_max=2, rules=())
PARAMS = zt.WorldParams(h=12, w=12, n=4, m=1, sense_r=1)
KEY = jax.random.PRNGKey(0)


def _world():
    return generate(GenConfig("open"), STATIC, PARAMS, KEY)


def _step(state, acts, bacts):
    return world_step(STATIC, PARAMS, state,
                      jnp.asarray(acts, jnp.int32), jnp.asarray(bacts, jnp.int32), KEY)


def _place(state, agent_pos, body_pos=None):
    s = state.replace(agent_pos=jnp.asarray(agent_pos, jnp.int32))
    if body_pos is not None:
        s = s.replace(body_pos=jnp.asarray(body_pos, jnp.int32))
    return s


def test_simple_moves_and_bounds():
    st_ = _place(_world(), [[5, 5], [0, 0], [11, 11], [7, 7], [9, 9], [9, 9]])
    acts = [zt.ActionId.N, zt.ActionId.N, zt.ActionId.S, zt.ActionId.E, 0, 0]
    new, ev = _step(st_, acts, [0, 0])
    assert tuple(np.asarray(new.agent_pos[0])) == (4, 5)      # N moves up
    assert tuple(np.asarray(new.agent_pos[1])) == (0, 0)      # blocked at edge
    assert tuple(np.asarray(new.agent_pos[2])) == (11, 11)    # blocked at runtime bound (h=12)
    assert tuple(np.asarray(new.agent_pos[3])) == (7, 8)      # E moves right
    assert bool(ev.moved[0]) and bool(ev.blocked[1]) and bool(ev.blocked[2])
    assert int(new.step_count) == 1


def test_wall_refusal():
    s = _world()
    s = s.replace(wall=s.wall.at[5, 6].set(True))
    s = _place(s, [[5, 5], [0, 0], [1, 1], [2, 2], [9, 9], [9, 9]])
    new, ev = _step(s, [zt.ActionId.E, 0, 0, 0, 0, 0], [0, 0])
    assert tuple(np.asarray(new.agent_pos[0])) == (5, 5)
    assert bool(ev.blocked[0])


def test_same_target_lowest_index_wins():
    s = _place(_world(), [[5, 4], [5, 6], [0, 0], [1, 1], [9, 9], [9, 9]])
    new, ev = _step(s, [zt.ActionId.E, zt.ActionId.W, 0, 0, 0, 0], [0, 0])
    assert tuple(np.asarray(new.agent_pos[0])) == (5, 5)      # index 0 wins
    assert tuple(np.asarray(new.agent_pos[1])) == (5, 6)      # index 1 reverted
    assert bool(ev.conflict[0, 1]) and bool(ev.conflict[1, 0])


def test_stayer_beats_mover():
    s = _place(_world(), [[5, 5], [5, 6], [0, 0], [1, 1], [9, 9], [9, 9]])
    new, _ = _step(s, [zt.ActionId.STAY, zt.ActionId.W, 0, 0, 0, 0], [0, 0])
    assert tuple(np.asarray(new.agent_pos[0])) == (5, 5)      # stayer keeps cell
    assert tuple(np.asarray(new.agent_pos[1])) == (5, 6)      # mover refused despite… nothing


def test_swap_through_refused():
    s = _place(_world(), [[5, 5], [5, 6], [0, 0], [1, 1], [9, 9], [9, 9]])
    new, _ = _step(s, [zt.ActionId.E, zt.ActionId.W, 0, 0, 0, 0], [0, 0])
    assert tuple(np.asarray(new.agent_pos[0])) == (5, 5)
    assert tuple(np.asarray(new.agent_pos[1])) == (5, 6)


def test_agent_outranks_body_on_contest():
    s = _place(_world(), [[5, 4], [0, 0], [1, 1], [2, 2], [9, 9], [9, 9]],
               body_pos=[[5, 6], [10, 10]])
    new, _ = _step(s, [zt.ActionId.E, 0, 0, 0, 0, 0], [zt.ActionId.W, 0])
    assert tuple(np.asarray(new.agent_pos[0])) == (5, 5)      # agent (lower global idx) wins
    assert tuple(np.asarray(new.body_pos[0])) == (5, 6)       # body reverted


def test_determinism_bit_exact():
    s = _world()
    a = [1, 2, 3, 4, 0, 0]
    n1, e1 = _step(s, a, [1, 0])
    n2, e2 = _step(s, a, [1, 0])
    assert (n1.agent_pos == n2.agent_pos).all() and (e1.moved == e2.moved).all()


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 2**31 - 1), st.integers(0, 2**31 - 1))
def test_property_collision_freedom(seed, aseed):
    """No two ALIVE entities ever share a cell after a step, from any generated world
    under random joint actions (walls + rooms + clutter)."""
    gk = jax.random.PRNGKey(seed)
    terr = ["open", "rooms", "clutter"][seed % 3]
    w = generate(GenConfig(terr, n_obstacles=20), STATIC, PARAMS, gk)
    acts = jax.random.randint(jax.random.PRNGKey(aseed), (STATIC.n_max,), 0, zt.N_ACTIONS)
    bacts = jax.random.randint(jax.random.PRNGKey(aseed + 1), (STATIC.m_max,), 0, zt.N_ACTIONS)
    new, _ = world_step(STATIC, PARAMS, w, acts, bacts, gk)
    pos = np.concatenate([np.asarray(new.agent_pos), np.asarray(new.body_pos)])
    alive = np.concatenate([np.asarray(new.agent_alive), np.asarray(new.body_alive)])
    cells = {tuple(p) for p in pos[alive]}
    assert len(cells) == alive.sum(), f"collision among alive entities: {pos[alive]}"


def test_padding_invariance():
    """Garbage in DEAD entity rows must not change any alive entity's outcome."""
    s = _world()
    a = [1, 2, 3, 4, 0, 0]
    garbage = s.replace(
        agent_pos=s.agent_pos.at[4:].set(jnp.int32(9999)),
        body_pos=s.body_pos.at[1:].set(jnp.int32(-7)))
    n1, e1 = _step(s, a, [1, 0])
    n2, e2 = _step(garbage, a, [1, 0])
    assert (n1.agent_pos[:4] == n2.agent_pos[:4]).all()
    assert (n1.body_pos[:1] == n2.body_pos[:1]).all()
    assert (e1.moved[:4] == e2.moved[:4]).all() and (e1.blocked[:4] == e2.blocked[:4]).all()


class _TagRule:
    """Test-only InteractionRule: bodies within Chebyshev r of any alive agent are captured."""
    def __init__(self, r):
        self.r = r

    def apply(self, static, params, state, events):
        d = jnp.abs(state.agent_pos[:, None, :] - state.body_pos[None, :, :]).max(-1)
        near = (d <= self.r) & state.agent_alive[:, None] & state.body_alive[None, :]
        captured = near.any(0)
        return (state.replace(body_alive=state.body_alive & ~captured),
                events.replace(captured=events.captured | captured))


def test_interaction_rule_seam():
    static = zt.StaticWorldParams(h_max=16, w_max=16, n_max=6, m_max=2,
                                  rules=(_TagRule(1),))
    assert isinstance(_TagRule(1), zt.InteractionRule)          # Protocol conformance
    s = _place(_world(), [[5, 5], [0, 0], [1, 1], [2, 2], [9, 9], [9, 9]],
               body_pos=[[5, 7], [10, 10]])
    new, ev = world_step(static, PARAMS, s,
                         jnp.asarray([zt.ActionId.E, 0, 0, 0, 0, 0], jnp.int32),
                         jnp.asarray([0, 0], jnp.int32), KEY)
    assert bool(ev.captured[0])            # agent stepped to (5,6), body at (5,7) within r=1
    assert not bool(new.body_alive[0])
    assert not bool(new.body_alive[1])     # index 1 is the dead padding row (m=1) — stays dead


def test_worldgen_shapes_and_distinct_spawns():
    for terr, nob in (("open", 0), ("rooms", 0), ("clutter", 25)):
        w = generate(GenConfig(terr, n_obstacles=nob), STATIC, PARAMS, jax.random.PRNGKey(3))
        pos = np.concatenate([np.asarray(w.agent_pos), np.asarray(w.body_pos)])
        assert len({tuple(p) for p in pos}) == len(pos), f"spawn collision on {terr}"
        wall = np.asarray(w.wall)
        assert not wall[pos[:, 0], pos[:, 1]].any(), f"spawned in wall on {terr}"
        assert np.asarray(w.arena)[:12, :12].all() and not np.asarray(w.arena)[12:, :].any()
