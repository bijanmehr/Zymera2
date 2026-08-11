"""Differential fuzz: the JAX world_step vs the independent NumPy reference."""
import jax
import jax.numpy as jnp
import numpy as np

from zymera2 import typing as zt
from zymera2.world import world_step
from zymera2.worldgen import GenConfig, generate
from np_reference import np_world_step

STATIC = zt.StaticWorldParams(h_max=16, w_max=16, n_max=6, m_max=2, rules=())
PARAMS = zt.WorldParams(h=12, w=12, n=5, m=2, sense_r=1)


def test_differential_fuzz_500_cases():
    mismatches = 0
    for case in range(500):
        gk = jax.random.PRNGKey(case)
        terr = ["open", "rooms", "clutter"][case % 3]
        w = generate(GenConfig(terr, n_obstacles=25), STATIC, PARAMS, gk)
        acts = jax.random.randint(jax.random.fold_in(gk, 1), (STATIC.n_max,), 0, 5)
        bacts = jax.random.randint(jax.random.fold_in(gk, 2), (STATIC.m_max,), 0, 5)

        new, ev = world_step(STATIC, PARAMS, w, acts, bacts, gk)

        pos = np.concatenate([np.asarray(w.agent_pos), np.asarray(w.body_pos)])
        alive = np.concatenate([np.asarray(w.agent_alive), np.asarray(w.body_alive)])
        a = np.concatenate([np.asarray(acts), np.asarray(bacts)])
        rnew, rmoved, rblocked, _ = np_world_step(
            STATIC.h_max, STATIC.w_max, PARAMS.h, PARAMS.w,
            np.asarray(w.wall), np.asarray(w.arena), pos, alive, a)

        jnew = np.concatenate([np.asarray(new.agent_pos), np.asarray(new.body_pos)])
        ok = ((jnew[alive] == rnew[alive]).all()
              and (np.asarray(ev.moved)[alive] == rmoved[alive]).all()
              and (np.asarray(ev.blocked)[alive] == rblocked[alive]).all())
        if not ok:
            mismatches += 1
            if mismatches <= 3:
                print(f"case {case} ({terr}): jax={jnew[alive].tolist()} "
                      f"ref={rnew[alive].tolist()}")
    assert mismatches == 0, f"{mismatches}/500 differential mismatches"
