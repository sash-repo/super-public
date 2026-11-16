#!/usr/bin/env python3
r"""H^2(g; g)_0 for a transitive depth-1 graded Lie superalgebra g, exactly over Q.

If the grading operator E lies in g_0, then L_E = d i_E + i_E d acts on the
degree-k part of H^.(g;g) as the scalar k and, being homotopic to zero, as zero;
hence H^.(g;g) = H^.(g;g)_0 and it suffices to compute the degree-0 part.

PROPAGATION.  Let c be a 2-cocycle of degree 0, so c(g_i,g_j) subset g_{i+j}.
If c vanishes on every pair of total degree <= -1, then c = 0.  (Induction on
T = i+j: in the cocycle identity for (x,y,z) with x in g_{-1} every summand but
[x,c(y,z)] is a value of c on a pair of total degree T-1, so [g_{-1},c(y,z)] = 0
with c(y,z) in g_T, T >= 0, and c(y,z) = 0 by transitivity.)  For depth 1 the
pairs of total degree <= -1 are (g_{-1},g_{-1}), whose target g_{-2} is 0, and
(g_{-1},g_0).  So a degree-0 cocycle is determined by the single component

    c : g_{-1} (x) g_0 --> g_{-1}.

BOUNDING THAT COMPONENT.  The cocycle identity on the triples (x,X,Y) with
x in g_{-1} and X,Y in g_0 expresses [x, c(X,Y)] through that component alone;
since g_0 acts faithfully on g_{-1}, the resulting map x |-> [x,c(X,Y)] must lie
in the image of g_0 in Hom(g_{-1},g_{-1}).  That is a necessary condition, hence
an upper bound for the space of the components that extend to a cocycle.

FINALLY, only the cochains of torus weight 0 matter: for h in a torus of g_0 the
operator L_h acts on the weight-mu part as mu(h), so every subcomplex of a
non-zero weight is acyclic.
"""
import sys
from fractions import Fraction

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rref, nullspace, rank                          # noqa: E402

from excalg import echelon, key_wt                                  # noqa: E402


def coord_map(S, basis, keys=None):
    """Return (keys, solve) where solve(D) gives the coordinates of D in `basis`."""
    if keys is None:
        keys = sorted({k for D in basis for k in D})
    pos = {k: i for i, k in enumerate(keys)}
    M = []
    for D in basis:
        v = [Fraction(0)] * len(keys)
        for k, c in D.items():
            v[pos[k]] = Fraction(c)
        M.append(v)
    R, piv = rref([row[:] for row in M])
    # express: coordinates in `basis` of a vector v.  Solve  lambda * M = v.
    MT = [[M[i][j] for i in range(len(M))] for j in range(len(keys))]

    from exactlin import solve as _solve

    def solve_coords(D):
        v = [Fraction(0)] * len(keys)
        for k, c in D.items():
            if k not in pos:
                return None
            v[pos[k]] = Fraction(c)
        return _solve(MT, v)

    return keys, solve_coords


def rigid_degree0(Sp, comp, name='', extra=True, verbose=True):
    """Return (dim Zbound, dim B) for H^2(g;g)_0 of a depth-1 prolong."""
    gm1, g0 = comp[-1], comp[0]
    n1, n0 = len(gm1), len(g0)
    p1 = [Sp.vf_par(D) for D in gm1]
    p0 = [Sp.vf_par(D) for D in g0]
    w1 = [Sp.vf_wt(D) for D in gm1]
    w0 = [Sp.vf_wt(D) for D in g0]

    keys1, co1 = coord_map(Sp, gm1)
    keys0, co0 = coord_map(Sp, g0)

    # brackets, precomputed
    br_10 = {}          # [e_a, X_m] in g_{-1} coordinates
    for a in range(n1):
        for m in range(n0):
            v = co1(Sp.bracket(gm1[a], g0[m]))
            br_10[(a, m)] = v if v else [Fraction(0)] * n1
    br_00 = {}          # [X_m, X_n] in g_0 coordinates
    for m in range(n0):
        for l in range(n0):
            v = co0(Sp.bracket(g0[m], g0[l]))
            br_00[(m, l)] = v if v else [Fraction(0)] * n0
    # The image of g_0 in Hom(g_{-1},g_{-1}) via Z |-> (e_a |-> [e_a,Z]), weight
    # by weight: an operator of weight w has entries only at the (a,b) with
    # wt(e_b) - wt(e_a) = w, and it can only come from g_0 of the same weight.
    homblk = {}
    for a in range(n1):
        for b in range(n1):
            w = tuple(x - y for x, y in zip(w1[b], w1[a]))
            homblk.setdefault(w, []).append((a, b))
    annw = {}
    for w, cells in homblk.items():
        rows_w = []
        for m in range(n0):
            if w0[m] != w:
                continue
            rows_w.append([br_10[(a, m)][b] for (a, b) in cells])
        annw[w] = (cells, nullspace(rows_w) if rows_w else
                   [[Fraction(1) if i == j else Fraction(0) for j in range(len(cells))]
                    for i in range(len(cells))])

    results = {}
    for pc in (0, 1):
        var = {}
        for a in range(n1):
            for m in range(n0):
                for b in range(n1):
                    if w1[b] != tuple(x + y for x, y in zip(w1[a], w0[m])):
                        continue
                    if (p1[b] - p1[a] - p0[m] - pc) % 2:
                        continue
                    var[(a, m, b)] = len(var)
        nv = len(var)

        def cval(a, m):
            """c(e_a, X_m) as a list of linear forms (dicts var->coeff) on g_{-1}."""
            return [({var[(a, m, b)]: Fraction(1)} if (a, m, b) in var else {})
                    for b in range(n1)]

        def add(dst, src, f):
            for k, c in src.items():
                dst[k] = dst.get(k, Fraction(0)) + f * c
            return dst

        def c_vec_g1(vec, m, f):
            """f * c(v, X_m) for v given by coordinates `vec` in g_{-1}."""
            out = [dict() for _ in range(n1)]
            for a in range(n1):
                if not vec[a]:
                    continue
                cc = cval(a, m)
                for b in range(n1):
                    add(out[b], cc[b], f * vec[a])
            return out

        rows = []
        for m in range(n0):
            for l in range(m, n0):
                if m == l and p0[m] == 0:
                    continue
                wpair0 = tuple(x + y for x, y in zip(w0[m], w0[l]))
                if wpair0 not in homblk:
                    continue
                # the map  e_a |-> [e_a, c(X_m, X_l)],  from the identity on (e_a,X_m,X_l)
                phi = [[dict() for _ in range(n1)] for _ in range(n1)]
                for a in range(n1):
                    p_x, pm, pl = p1[a], p0[m], p0[l]
                    tot = [dict() for _ in range(n1)]
                    # T1 = (-1)^{1+pm(pc+p_x)} [X_m, c(e_a, X_l)]
                    s = -1 if (pm * (pc + p_x)) % 2 == 0 else 1
                    cc = cval(a, l)
                    for b in range(n1):
                        if not cc[b]:
                            continue
                        # [X_m, e_b] = -(-1)^{p(X_m) p(e_b)} [e_b, X_m]
                        sb = -1 if (pm * p1[b]) % 2 == 0 else 1
                        v = br_10[(b, m)]
                        for b2 in range(n1):
                            if v[b2]:
                                add(tot[b2], cc[b], s * sb * v[b2])
                    # T2 = (-1)^{pl(pc+p_x+pm)} [X_l, c(e_a, X_m)]
                    s = 1 if (pl * (pc + p_x + pm)) % 2 == 0 else -1
                    cc = cval(a, m)
                    for b in range(n1):
                        if not cc[b]:
                            continue
                        sb = -1 if (pl * p1[b]) % 2 == 0 else 1
                        v = br_10[(b, l)]
                        for b2 in range(n1):
                            if v[b2]:
                                add(tot[b2], cc[b], s * sb * v[b2])
                    # T01 = - c([e_a, X_m], X_l)
                    cc = c_vec_g1(br_10[(a, m)], l, Fraction(-1))
                    for b in range(n1):
                        add(tot[b], cc[b], Fraction(1))
                    # T02 = (-1)^{pl pm} c([e_a, X_l], X_m)
                    s = 1 if (pl * pm) % 2 == 0 else -1
                    cc = c_vec_g1(br_10[(a, l)], m, Fraction(s))
                    for b in range(n1):
                        add(tot[b], cc[b], Fraction(1))
                    # T12 = (-1)^{1+p_x(pm+pl)} c([X_m,X_l], e_a)
                    #     = (-1)^{1+p_x(pm+pl)} * (-(-1)^{p_x(pm+pl)}) c(e_a,[X_m,X_l])
                    #     = c(e_a, [X_m, X_l])
                    v = br_00[(m, l)]
                    for mm in range(n0):
                        if not v[mm]:
                            continue
                        cc = cval(a, mm)
                        for b in range(n1):
                            add(tot[b], cc[b], v[mm])
                    # T0 = (-1)^{p_x pc}[e_a, c(X_m,X_l)] = -(T1+T2+T01+T02+T12)
                    s = 1 if (p_x * pc) % 2 == 0 else -1
                    for b in range(n1):
                        phi[a][b] = {k: -s * cf for k, cf in tot[b].items() if cf}
                # require phi to lie in ad(g_0); only its own weight block matters
                wpair = tuple(x + y for x, y in zip(w0[m], w0[l]))
                if wpair not in annw:
                    continue
                cells, qs = annw[wpair]
                for q in qs:
                    row = [Fraction(0)] * nv
                    for f, (a, b) in zip(q, cells):
                        if not f:
                            continue
                        for k, cf in phi[a][b].items():
                            row[k] += f * cf
                    if any(row):
                        rows.append(row)
        ker = nullspace(rows) if rows else [[Fraction(1) if i == j else Fraction(0)
                                             for j in range(nv)] for i in range(nv)]
        dimZ = len(ker)

        # ---- coboundaries: phi_{-1} in End(g_{-1}), phi_0 in End(g_0), weight 0
        cob = []
        pairs = []
        for a in range(n1):
            for b in range(n1):
                if w1[a] == w1[b] and (p1[b] - p1[a] - pc) % 2 == 0:
                    pairs.append(('m', a, b))
        for m in range(n0):
            for l in range(n0):
                if w0[m] == w0[l] and (p0[l] - p0[m] - pc) % 2 == 0:
                    pairs.append(('z', m, l))
        for tag, i, j in pairs:
            row = [Fraction(0)] * nv
            for a in range(n1):
                for m in range(n0):
                    # (d phi)(e_a, X_m) = (-1)^{p_a pc}[e_a, phi(X_m)]
                    #      - (-1)^{pm(pc+p_a)}[X_m, phi(e_a)] - phi([e_a, X_m])
                    val = [Fraction(0)] * n1
                    if tag == 'z' and i == m:
                        s = 1 if (p1[a] * pc) % 2 == 0 else -1
                        v = br_10[(a, j)]
                        for b in range(n1):
                            val[b] += s * v[b]
                    if tag == 'm' and i == a:
                        s = -1 if (p0[m] * (pc + p1[a])) % 2 == 0 else 1
                        sb = -1 if (p0[m] * p1[j]) % 2 == 0 else 1
                        v = br_10[(j, m)]
                        for b in range(n1):
                            val[b] += s * sb * v[b]    # [X_m,e_j]=-(-1)^{pm pj}[e_j,X_m]
                    if tag == 'm':
                        v = br_10[(a, m)]
                        if v[i]:
                            val[j] -= v[i]
                    for b in range(n1):
                        if val[b] and (a, m, b) in var:
                            row[var[(a, m, b)]] += val[b]
                        elif val[b]:
                            # a coboundary component outside the weight-0 block:
                            # impossible, since phi is weight 0
                            raise AssertionError('weight leak in the coboundary')
            if any(row):
                cob.append(row)
        dimB = rank(cob) if cob else 0
        # every coboundary must satisfy every equation
        ok = True
        for v in cob:
            for r in rows:
                if sum(a * b for a, b in zip(r, v)):
                    ok = False
                    break
            if not ok:
                break
        results[pc] = (nv, len(rows), dimZ, dimB, ok)
        if verbose:
            print('  %-14s parity %d: unknowns %4d, equations %5d, '
                  'dim Z <= %3d, dim B = %3d, B in Z: %s'
                  % (name, pc, nv, len(rows), dimZ, dimB, ok))
    return results
