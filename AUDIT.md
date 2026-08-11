# zymera2 — audit cards (v0, 2026-08-11)

Per-module: claimed semantics → what verifies them. This file ships with the package
(spec §10.3). Claims are scoped per the three-row leakage taxonomy (§14.3): the suite
certifies **execution-time transport**; training is centralized and privileged (CTDE)
and is NOT certified leak-free — trained weights are a channel.

| Module | Claim | Verified by |
|---|---|---|
| `typing.py` | single-source contract; `World` holds no reward/done/comms/RNG/belief; runtime params are a pytree (no recompile across sizes); rules immutable tuple | `test_contract.py` (8) · `test_integration_contract.py` (jit-cache size = 1 across sizes; bit-exact CPU determinism) |
| `validate.py` | runtime ≤ caps; positivity; disjoint seed pools; comms params excluded from world params | `test_validate.py` (6) |
| `worldgen.py` | open/rooms/clutter; distinct free spawns; arena validity; fold_in-only randomness | `test_world_step.py::test_worldgen_*` |
| `world.py` | unified agents+bodies conflict pass: stayer > mover, lowest global index, swap refusal, fixpoint; dead rows inert; rules applied in static order | unit tests (7) · property: collision-freedom (hypothesis, 60 random cases × 3 terrains) · padding-invariance · **differential: 500/500 vs NumPy reference** (`np_reference.py` — independent computation path, same-author caveat recorded) |
| `geometry.py` | Chebyshev dist, +inf dead rows (padding can never create adjacency); digital-line wall crossings (pinned v0 algorithm — NOT supercover, deviation recorded) | `test_geometry_comms.py` vs hand-computed oracles |
| `comms.py` | delivered ⊆ potential; exact delay (delivery-time gating, warm-up); per-undirected-edge fold_in dropout; payload opacity (never branched on) | delay-exactness · subset-under-dropout · closed-form rate 0.7±0.03 · edge symmetry · occlusion cut · **G5 content-swap** |
| `tick.py` | normative order + fold_in key tree in-package; obs = exactly {patch, patch_valid, entities, mail, adj_row, statics} | leakage suite + composed-tick **golden v0** (`6d9e0d24…`, per-jaxlib artifact) |
| leakage | T1 cone-nullity · T2 full-episode bit-exact counterfactual · T3 dropout=1 ablation · T4 delivered⊆potential · T5 whitelist + no-global-leaf | `test_tick_leakage.py` — **suite sensitivity proven: 5/5 planted v1-style leaks caught** (statics-smuggled map, god-view patch, potential-for-delivered, delay bypass, mail-mask bypass) |
| trainability | gradients flow through the composed tick; REINFORCE improves team coverage (49→59 of 144 cells @ horizon 30) | `test_smoke_train.py` — fitness check, **not** a correctness claim |

**Substituted rung.** Bit-parity vs v1 was dropped as meaningless: v0 semantics deliberately
correct v1 (co-location forbidden, delivered-vs-potential adjacency, delayed lossy channel),
so trajectories legitimately diverge. Its role is carried by the NumPy differential + the
pinned golden. Metric-level v1 comparisons belong to the migration study, not this audit.

**Known scope limits (v0).** One payload leaf per PayloadSpec; sense radius static per tick
build; digital-line (not supercover) crossings; no capture rule shipped in-package (the
rule seam is tested with a test-only TagRule); GPU determinism tier untested (no GPU).
