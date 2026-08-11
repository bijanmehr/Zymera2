"""NumPy reference implementation of the world step (spec §14.4, external-validation rung).

Written procedurally from the SPEC TEXT (per-entity loops, explicit fixpoint), not
translated from the vectorized JAX code — the differential test compares two
independent computation paths. Same-author caveat applies and is recorded in AUDIT.md.
"""
import numpy as np

DELTAS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]


def np_world_step(h_max, w_max, h, w, wall, arena, pos, alive, acts):
    """pos [E,2] int, alive [E] bool, acts [E] int -> (new_pos, moved, blocked, conflict).

    Semantics (-v0): dead entities are inert; refuse wall/out-of-runtime/out-of-arena
    targets; stayers keep their cell; contested cells go to the lowest global index
    among movers; swap-throughs are refused pairwise; losers revert; iterate to fixpoint.
    """
    E = len(pos)
    pos = np.clip(np.array(pos, int), 0, [h_max - 1, w_max - 1])
    acts = [a if alive[e] else 0 for e, a in enumerate(np.clip(acts, 0, 4))]

    target = []
    for e in range(E):
        dr, dc = DELTAS[acts[e]]
        t = (pos[e][0] + dr, pos[e][1] + dc)
        okay = (alive[e] and 0 <= t[0] < h and 0 <= t[1] < w
                and not wall[t] and arena[t])
        target.append(t if okay else tuple(pos[e]))

    conflict = np.zeros((E, E), bool)
    for _ in range(E):                                    # fixpoint
        changed = False
        for i in range(E):
            if not alive[i] or target[i] == tuple(pos[i]):
                continue
            for j in range(E):
                if j == i or not alive[j]:
                    continue
                if target[j] == target[i]:
                    conflict[i, j] = conflict[j, i] = True
                    j_stays = target[j] == tuple(pos[j])
                    if j_stays or j < i:                  # stayer wins, else lower index
                        target[i] = tuple(pos[i]); changed = True; break
                if (target[i] == tuple(pos[j]) and target[j] == tuple(pos[i])
                        and target[j] != tuple(pos[j])):  # swap-through: both refused
                    conflict[i, j] = conflict[j, i] = True
                    target[i] = tuple(pos[i]); target[j] = tuple(pos[j])
                    changed = True; break
        if not changed:
            break

    new = np.array(target, int)
    moved = (new != pos).any(1)
    blocked = np.array([acts[e] != 0 and alive[e] and not moved[e] for e in range(E)])
    return new, moved, blocked, conflict
