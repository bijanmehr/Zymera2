"""zymera2 — a pure-JAX discrete grid-world simulator (world-only core).

The world owns maps, entities, movement, collision, sensing, and events. It has no
rewards, no episodes, no missions, no communication (that is the ``zymera2.comms`` bridge),
and no agent machinery. See docs/design/2026-08-10-zymera2-world-design.md (esp. §14 —
the binding corrections adopted from adversarial review).
"""
__version__ = "0.1.0.dev0"
