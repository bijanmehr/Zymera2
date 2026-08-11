# zymera2

A pure-JAX **discrete grid-world simulator** — a *world only*: it generates maps (arbitrary sizes,
arena shapes, obstacle layouts), moves entities, resolves collisions and contact rules, answers sensing
queries, and reports typed events. It has **no rewards, no episodes, no missions, no communication, and
no agent machinery** — those compose above it (a task layer for missions/scoring; a separate
`zymera2.comms` bridge for the radio; agents in their own layer).

Status: **early build (P0 — foundation).** Design + binding corrections:
`../Project.Zymera/.../docs/design/2026-08-10-zymera2-world-design.md` (§14 is authoritative).
Build is intentionally **lean** — correct contract + full test suite as the definition of "solid";
publish ceremony (CI, license, DOI, docs site, PyPI) is deferred behind a go/no-go gate.
