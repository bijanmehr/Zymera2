import pytest
from zymera2 import typing as zt
from zymera2.validate import validate_params, validate_seed_pools


def _static():
    return zt.StaticWorldParams(h_max=8, w_max=8, n_max=4, m_max=2, rules=())


def _params(**kw):
    d = dict(h=8, w=8, n=4, m=2, sense_r=1)
    d.update(kw)
    return zt.WorldParams(**d)


def test_valid_params_pass():
    validate_params(_static(), _params())  # no raise


def test_runtime_exceeds_caps_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        validate_params(_static(), _params(h=16))
    with pytest.raises(ValueError, match="exceeds"):
        validate_params(_static(), _params(n=8))


def test_nonpositive_rejected():
    with pytest.raises(ValueError):
        validate_params(_static(), _params(sense_r=0))
    with pytest.raises(ValueError):
        validate_params(_static(), _params(h=0))


def test_zero_bodies_is_valid():
    validate_params(_static(), _params(m=0))  # coverage missions have no bodies


def test_comms_params_do_not_live_in_world_params():
    # comms is a bridge property, not a world property (ruling 2026-08-10; spec §8)
    assert "comm_r" not in zt.WorldParams.__dataclass_fields__


def test_disjoint_seed_pools_ok_and_overlap_rejected():
    validate_seed_pools({"train": {1, 2, 3}, "test": {4, 5}, "eval": {6}})
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_pools({"train": {1, 2}, "test": {2, 3}})
