import jax.numpy as jnp
import chex
from zymera2 import typing as zt


def test_action_ids():
    assert (zt.ActionId.STAY, zt.ActionId.N, zt.ActionId.E, zt.ActionId.S, zt.ActionId.W) == (0, 1, 2, 3, 4)
    assert zt.N_ACTIONS == 5
    assert len(zt.ACTION_DELTAS) == 5


def test_action_deltas_semantics():
    # (row, col), row axis 0 with down positive: N up, S down, E right, W left.
    assert zt.ACTION_DELTAS[zt.ActionId.STAY] == (0, 0)
    assert zt.ACTION_DELTAS[zt.ActionId.N] == (-1, 0)
    assert zt.ACTION_DELTAS[zt.ActionId.S] == (1, 0)
    assert zt.ACTION_DELTAS[zt.ActionId.E] == (0, 1)
    assert zt.ACTION_DELTAS[zt.ActionId.W] == (0, -1)


def test_world_constructs_with_declared_shapes():
    N, M, H, W = 4, 2, 8, 8
    w = zt.World(
        agent_pos=jnp.zeros((N, 2), jnp.int32), agent_alive=jnp.ones((N,), bool),
        body_pos=jnp.zeros((M, 2), jnp.int32), body_alive=jnp.ones((M,), bool),
        body_kind=jnp.zeros((M,), jnp.int32),
        wall=jnp.zeros((H, W), bool), arena=jnp.ones((H, W), bool),
        step_count=jnp.zeros((), jnp.int32))
    chex.assert_shape(w.agent_pos, (N, 2))
    chex.assert_shape(w.wall, (H, W))
    assert w.agent_pos.dtype == jnp.int32


def test_world_has_no_forbidden_fields():
    forbidden = {"reward", "done", "channel", "comm_graph", "key", "rng", "belief"}
    assert forbidden.isdisjoint(set(zt.World.__dataclass_fields__))


def test_kernel_events_unified_entity_index():
    # events cover the unified conflict pass: E = N + M (agents then bodies, C1)
    N, M = 4, 2
    E = N + M
    e = zt.KernelEvents(
        moved=jnp.zeros((E,), bool), blocked=jnp.zeros((E,), bool),
        conflict=jnp.zeros((E, E), bool), captured=jnp.zeros((M,), bool))
    chex.assert_shape(e.conflict, (E, E))


def test_kernel_events_have_no_sense_field():
    # sensing is a harness-side readout with its own SenseEvent (spec §14/C4);
    # world_step cannot emit "sensed"
    assert "sensed" not in zt.KernelEvents.__dataclass_fields__


def test_protocols_present_and_payloadspec():
    for name in ("Generator", "InteractionRule", "Topology", "Channel", "PolicyFn"):
        assert hasattr(zt, name)
    ps = zt.PayloadSpec(shape=(1,), dtype=jnp.float32)
    assert isinstance(ps, zt.PayloadSpec) and ps.shape == (1,)


def test_rules_are_a_tuple_on_static_params():
    sp = zt.StaticWorldParams(h_max=8, w_max=8, n_max=4, m_max=2, rules=())
    assert isinstance(sp.rules, tuple)  # immutable ordered — no registry (C8)


def test_world_step_fn_alias_exists():
    assert zt.WorldStepFn is not None
