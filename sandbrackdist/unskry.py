#!/usr/bin/env python3
r"""
unskry.py --- the middle Skryabin algebra my(6; \un) for \un =/= \One: the
obstruction to a 3-structure.

my(6; \un) = (E^2(V), V, gl(V))_{*,\un} with dim V = 3, realized in vect(6; \un):
the coordinates x_1, x_2, x_3 of degree 1 (heights 1, they are the constrained
ones) and y_1, y_2, y_3 of degree 2 (heights N_4, N_5, N_6).  Its depth is 2 and
its g_{-2} is spanned by the PURE partials d_{y_k}.

Since (d_{y_k})^3 = d_{y_k}^{(3)} as an operator on O(6;\un) as soon as N_k >= 2,
and since an element f with ad_f = (ad_{d_{y_k}})^3 would have to be homogeneous
of degree -6 < -2, the algebra is restricted only if (ad_{d_{y_k}})^3 = 0 on it,
i.e. only if no coefficient occurring in it has y_k-degree >= 3.  This engine
prolongs degree by degree (one (degree, weight) block at a time) and looks.

Output: runs/unskry.log
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402
from dpvect import P, DPVect                                          # noqa: E402
from gvect import rref, nullspace                                     # noqa: E402

LOG = os.path.join(HERE, "runs", "unskry.log")

EPS = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1,
       (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}


def build(heights):
    """g_- and g_0 of my(6; un); the fields do not depend on the heights."""
    degs = [1, 1, 1, 2, 2, 2]
    e = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    three = (1, 1, 1)
    wts = [tuple(-a for a in e[i]) for i in range(3)] + \
          [tuple(-(three[t] - e[i][t]) for t in range(3)) for i in range(3)]
    names = ["x1", "x2", "x3", "y1", "y2", "y3"]
    V = DPVect(degs, wts, heights, names)
    Z = tuple([0] * 6)

    def mon(**kw):
        m = [0] * 6
        for k, v in kw.items():
            m[names.index(k)] = v
        return tuple(m)

    X = []
    for i in range(3):
        D = {(Z, i): 1}
        for j in range(3):
            for k in range(3):
                s = EPS.get((i, j, k), 0)
                if s:
                    key = (mon(**{"x%d" % (j + 1): 1}), 3 + k)
                    D[key] = (D.get(key, 0) + s) % P
        X.append({k: v for k, v in D.items() if v})
    Y = [{(Z, 3 + k): 1} for k in range(3)]
    gm = X + Y

    # g_0: the degree-0 fields normalizing g_-  (W_0 does not depend on \un)
    src = [k for k in V.basis_keys() if V.deg(k) == 0]
    tgt = [k for k in V.basis_keys() if V.deg(k) in (-1, -2)]
    tpos = {k: i for i, k in enumerate(tgt)}
    M = np.zeros((len(gm), len(tgt)), dtype=np.int64)
    for r, D in enumerate(gm):
        for k, c in D.items():
            M[r, tpos[k]] = c % P
    A = nullspace(M, len(tgt))
    rows = []
    for E in gm:
        col = np.zeros((len(src), len(tgt)), dtype=np.int64)
        for i, s in enumerate(src):
            for k, c in V.bracket({s: 1}, E).items():
                col[i, tpos[k]] = c % P
        rows.append((A @ col.T) % P)
    Mall = np.concatenate(rows, axis=0)
    NS = nullspace(Mall, len(src))
    g0 = [{s: int(c) for s, c in zip(src, v) if c} for v in NS]
    return V, X, Y, g0


def prolong(V, layers, dmax, say):
    """the complete prolong, one (degree, weight) block at a time."""
    gm1 = layers[-1]
    bykey = {}
    for k in V.basis_keys():
        bykey.setdefault((V.deg(k), V.wt(k)), []).append(k)
    d = 1
    while d <= dmax:
        prev = layers.get(d - 1, [])
        if not prev:
            break
        prevby = {}
        for D in prev:
            prevby.setdefault(V.fwt(D), []).append(D)
        cur = []
        for (dg, w), src in sorted(bykey.items()):
            if dg != d:
                continue
            # the condition [D, Y] in g_{d-1} for every Y in g_{-1}
            rows = []
            for E in gm1:
                we = V.fwt(E)
                tgtw = tuple(a + b for a, b in zip(w, we))
                tgt = bykey.get((d - 1, tgtw), [])
                if not tgt:
                    # the bracket must vanish identically
                    tgt = []
                tpos = {k: i for i, k in enumerate(tgt)}
                sub = prevby.get(tgtw, [])
                Msub = np.zeros((len(sub), len(tgt)), dtype=np.int64)
                for r, D in enumerate(sub):
                    for k, c in D.items():
                        Msub[r, tpos[k]] = c % P
                A = nullspace(Msub, len(tgt)) if len(tgt) else \
                    np.zeros((0, 0), dtype=np.int64)
                col = np.zeros((len(src), len(tgt)), dtype=np.int64)
                bad = np.zeros((len(src),), dtype=np.int64)
                for i, s in enumerate(src):
                    br = V.bracket({s: 1}, E)
                    for k, c in br.items():
                        if k in tpos:
                            col[i, tpos[k]] = c % P
                        else:
                            bad[i] = 1
                if len(tgt):
                    rows.append((A @ col.T) % P)
                if bad.any():
                    rows.append(np.diag(bad)[bad.astype(bool)] if False else
                                np.array([[1 if bad[i] and i == j else 0
                                           for j in range(len(src))]
                                          for i in range(len(src))
                                          if bad[i]], dtype=np.int64))
            Mall = np.concatenate([r for r in rows if r.size],
                                  axis=0) if rows else \
                np.zeros((0, len(src)), dtype=np.int64)
            NS = nullspace(Mall, len(src))
            for v in NS:
                D = {s: int(c) % P for s, c in zip(src, v) if c % P}
                if D:
                    cur.append(D)
        if not cur:
            break
        layers[d] = cur
        say("   g_%-3d dim %d" % (d, len(cur)))
        d += 1
    return layers


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(m):
        print(m)
        log.write(m + "\n")

    for heights, dmax in (((1, 1, 1, 1, 1, 1), 12), ((1, 1, 1, 2, 1, 1), 9)):
        say("")
        say("=" * 70)
        say("my(6; un) with un = %s" % (heights,))
        V, X, Y, g0 = build(heights)
        say("   dim O(6; un) = %d,  dim g_0 = %d" % (len(V.mons), len(g0)))
        layers = {-1: X, -2: Y, 0: g0}
        prolong(V, layers, dmax, say)
        total = sum(len(v) for v in layers.values())
        say("   dim (computed up to degree %d) = %d, top degree %d"
            % (dmax, total, max(layers)))

        # where does the height show?
        worst = 0
        witness = None
        for dg, fields in sorted(layers.items()):
            for D in fields:
                for (m, i) in D:
                    if m[3] > worst:
                        worst, witness = m[3], (dg, D)
        say("   largest y_1-degree of a coefficient: %d" % worst)
        if witness:
            say("   witness in g_%d: %s" % (witness[0], V.show(witness[1])))

        # (ad_{d_{y_1}})^3 on the algebra
        dy1 = Y[0]
        nz = 0
        first = None
        for dg, fields in sorted(layers.items()):
            for D in fields:
                B = V.bracket(dy1, V.bracket(dy1, V.bracket(dy1, D)))
                if B:
                    nz += 1
                    if first is None:
                        first = (dg, D, B)
        say("   (ad_{d_y1})^3 =/= 0 on %d of the %d basis fields" % (nz, total))
        if first:
            say("      e.g. (ad_{d_y1})^3(%s) = %s   [degree %d -> %d]"
                % (V.show(first[1]), V.show(first[2]), first[0], first[0] - 6))
            say("      an f with ad_f = (ad_{d_y1})^3 would be of degree -6 < -2,"
                " so f = 0: NOT RESTRICTED")
        else:
            say("      (ad_{d_y1})^3 = 0 throughout")

        # the cube of d_{y_1} as an operator on O(6; un)
        op = V.power_operator(dy1, P)
        fld = V.as_field(op)
        say("   (d_y1)^3 as an operator on O(6;un) is %s"
            % ("the field %s" % V.show(fld) if fld is not None
               else "NOT a vector field: it is d_{y1}^{(3)}"))
    log.close()


if __name__ == "__main__":
    main()
