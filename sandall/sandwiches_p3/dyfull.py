#!/usr/bin/env python3
r"""
dyfull.py --- the deep Skryabin algebra dy(10; \One) in full: the complete Cartan
prolong of the non-positive part transcribed in dy3.py, built one (degree,
weight) block at a time, and then cubed element by element.

dy3.py settles the 3-structure through the Lemma (a prolong is closed under p-th
powers as soon as its non-positive part is); this engine checks the conclusion on
every basis element of the algebra, which is the independent confirmation.

Also: the derived series of each of the four small Skryabin algebras, run to
stabilization.

Output: runs/dyfull.log
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402
from dpvect import P, DPVect                                          # noqa: E402
from gvect import rref, nullspace                                     # noqa: E402
from skry3 import cube, is_derivation                                 # noqa: E402
import dy3                                                            # noqa: E402

LOG = os.path.join(HERE, "runs", "dyfull.log")


def blocks_of(V):
    out = {}
    for k in V.basis_keys():
        out.setdefault((V.deg(k), V.wt(k)), []).append(k)
    return out


def prolong(V, layers, bykey, say, dmax=200):
    gm1 = layers[-1]
    d = 1
    while d <= dmax:
        prev = layers.get(d - 1, [])
        if not prev:
            break
        prevby = {}
        for D in prev:
            prevby.setdefault(V.fwt(D), []).append(D)
        cur = []
        for (dg, w), src in bykey.items():
            if dg != d:
                continue
            rows = []
            for E in gm1:
                tgtw = tuple(a + b for a, b in zip(w, V.fwt(E)))
                tgt = bykey.get((d - 1, tgtw), [])
                tpos = {k: i for i, k in enumerate(tgt)}
                sub = prevby.get(tgtw, [])
                Msub = np.zeros((len(sub), len(tgt)), dtype=np.int64)
                for r, D in enumerate(sub):
                    for k, c in D.items():
                        Msub[r, tpos[k]] = c % P
                A = nullspace(Msub, len(tgt)) if tgt else None
                col = np.zeros((len(src), len(tgt)), dtype=np.int64)
                bad = []
                for i, s in enumerate(src):
                    for k, c in V.bracket({s: 1}, E).items():
                        if k in tpos:
                            col[i, tpos[k]] = c % P
                        else:
                            bad.append(i)
                if tgt:
                    rows.append((A @ col.T) % P)
                if bad:
                    R = np.zeros((len(set(bad)), len(src)), dtype=np.int64)
                    for r, i in enumerate(sorted(set(bad))):
                        R[r, i] = 1
                    rows.append(R)
            rows = [r for r in rows if r.size]
            Mall = np.concatenate(rows, axis=0) if rows else \
                np.zeros((0, len(src)), dtype=np.int64)
            for v in nullspace(Mall, len(src)):
                D = {s: int(c) % P for s, c in zip(src, v) if c % P}
                if D:
                    cur.append(D)
        if not cur:
            break
        layers[d] = cur
        say("   g_%-3d dim %d" % (d, len(cur)))
        d += 1
    return layers


def in_block(V, fields, D):
    """is the homogeneous field D in the span of `fields` (same block)?"""
    keys = sorted({k for E in fields for k in E} | set(D))
    kpos = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(fields), len(keys)), dtype=np.int64)
    for r, E in enumerate(fields):
        for k, c in E.items():
            M[r, kpos[k]] = c % P
    v = np.zeros(len(keys), dtype=np.int64)
    for k, c in D.items():
        v[kpos[k]] = c % P
    R, piv = rref(M)
    w = v.copy()
    for i, c in enumerate(piv):
        if w[c]:
            w = (w - w[c] * R[i]) % P
    return not w.any()


def derived_chain(name, say):
    """the derived series of one of the four small Skryabin algebras."""
    from prolong import make
    from skry3 import Model
    g = make(name, verbose=False)
    m = Model(g.V, g.basis, g.deg, g.wt, name)
    dims = [m.n]
    cur = m
    while True:
        rows = []
        for i in range(cur.n):
            for j in range(i + 1, cur.n):
                co = cur.coords(cur.V.bracket(cur.basis[i], cur.basis[j]))
                if co is not None and co.any():
                    rows.append(co)
        R, piv = rref(np.array(rows, dtype=np.int64))
        if len(R) == cur.n:
            break
        basis = [cur.field(v) for v in R]
        cur = Model(cur.V, basis, [cur.V.fdeg(D) for D in basis],
                    [cur.V.fwt(D) for D in basis], name + "'")
        dims.append(cur.n)
    say("  %-4s: derived series %s (stabilizes, and the last term is perfect)"
        % (name, " -> ".join(str(d) for d in dims)))
    return cur


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(m):
        print(m)
        log.write(m + "\n")

    if "--derived" in sys.argv:
        say("the derived series of the small Skryabin algebras (un = One):")
        for name in ("my", "smy", "by", "sby"):
            derived_chain(name, say)

    say("")
    say("dy(10; One): the complete prolong of the non-positive part of dy3.py")
    V = DPVect(dy3.DEG, dy3.WT, [1] * 10, dy3.NAMES)
    layers = {-1: dy3.GM[0:3], -2: dy3.GM[3:6], -3: dy3.GM[6:7],
              -4: dy3.GM[7:10], 0: dy3.G0}
    say("   dim O(10; One) = %d" % len(V.mons))
    bykey = blocks_of(V)
    say("   %d (degree, weight) blocks, largest %d"
        % (len(bykey), max(len(v) for v in bykey.values())))
    prolong(V, layers, bykey, say)
    total = sum(len(v) for v in layers.values())
    say("   dim dy(10; One) = %d, degrees %d..%d"
        % (total, min(layers), max(layers)))

    say("")
    say("   cubing every basis element:")
    byblock = {}
    for dg, fields in layers.items():
        for D in fields:
            byblock.setdefault((dg, V.fwt(D)), []).append(D)
    nonzero, outside = [], []
    for dg, fields in sorted(layers.items()):
        for D in fields:
            C = cube(V, D)
            if not C:
                continue
            assert is_derivation(V, C)
            key = (3 * dg, tuple(3 * a for a in V.fwt(D)))
            tgt = byblock.get(key, [])
            if tgt and in_block(V, tgt, C):
                nonzero.append((dg, D, C))
            else:
                outside.append((dg, D, C))
    say("   non-zero cubes: %d, all of them inside dy: %s"
        % (len(nonzero), not outside))
    for dg, D, C in nonzero:
        say("     (%s)^[3] = %s   [degree %d]" % (V.show(D), V.show(C), dg))
    if outside:
        for dg, D, C in outside:
            say("     OUTSIDE: (%s)^3 = %s   [degree %d]"
                % (V.show(D), V.show(C), dg))

    # the derived series of dy, degree-wise dimensions
    say("")
    say("   dim dy(10;One)_k: " + ", ".join(
        "%d:%d" % (k, len(v)) for k, v in sorted(layers.items())))
    log.close()


if __name__ == "__main__":
    main()
