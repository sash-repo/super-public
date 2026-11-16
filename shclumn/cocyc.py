#!/usr/bin/env python3
r"""H^2(g;g)_0 for a W-graded transitive vectorial Lie superalgebra, exactly over Q.

Generic version: unknowns for a chosen set of bidegrees, equations from the
cocycle identity on every triple all of whose terms stay inside that set.  Any
subset of the identities is a necessary condition, so the space it cuts out is
an UPPER BOUND for the space of the restrictions of genuine cocycles to the
components that determine them (propagation lemma).  When that upper bound has
the dimension of the space of coboundaries, H^2(g;g)_0 = 0.
"""
import sys
from fractions import Fraction

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rref, nullspace, rank, solve as _solve          # noqa: E402


class Graded:
    """A W-graded Lie superalgebra given by bases of its components."""

    def __init__(self, Sp, comp):
        self.S, self.comp = Sp, comp
        self.deg = sorted(comp)
        self.par = {i: [Sp.vf_par(D) for D in comp[i]] for i in comp}
        self.wt = {i: [Sp.vf_wt(D) for D in comp[i]] for i in comp}
        self._solver = {}
        self.br = {}

    def _sv(self, i):
        if i not in self._solver:
            basis = self.comp[i]
            keys = sorted({k for D in basis for k in D})
            pos = {k: j for j, k in enumerate(keys)}
            M = [[Fraction(0)] * len(keys) for _ in basis]
            for r, D in enumerate(basis):
                for k, c in D.items():
                    M[r][pos[k]] = Fraction(c)
            MT = [[M[r][c] for r in range(len(basis))] for c in range(len(keys))]
            self._solver[i] = (pos, MT, len(basis))
        return self._solver[i]

    def coords(self, D, i):
        if i not in self.comp:
            if D:
                raise AssertionError('bracket left the range of degrees')
            return []
        pos, MT, n = self._sv(i)
        v = [Fraction(0)] * len(pos)
        for k, c in D.items():
            if k not in pos:
                raise AssertionError('vector outside the span of g_%d' % i)
            v[pos[k]] = Fraction(c)
        if not any(v):
            return [Fraction(0)] * n
        x = _solve(MT, v)
        if x is None:
            raise AssertionError('vector outside the span of g_%d' % i)
        return x

    def bracket(self, i, a, j, b):
        """[e^i_a, e^j_b] in the coordinates of g_{i+j} (empty if that is 0)."""
        key = (i, a, j, b)
        if key in self.br:
            return self.br[key]
        D = self.S.bracket(self.comp[i][a], self.comp[j][b])
        if i + j not in self.comp:
            if D:
                raise AssertionError('bracket into a missing degree %d' % (i + j))
            out = []
        else:
            out = self.coords(D, i + j) if D else [Fraction(0)] * len(self.comp[i + j])
        self.br[key] = out
        return out


def wsum(*ws):
    out = None
    for w in ws:
        out = list(w) if out is None else [x + y for x, y in zip(out, w)]
    return tuple(out)


class Cocycle:
    """The degree-0 2-cochains of a given parity on a chosen set of bidegrees."""

    def __init__(self, G, allowed, pc):
        self.G, self.allowed, self.pc = G, set(allowed), pc
        self.var = {}
        for (i, j) in sorted(self.allowed):
            if i > j:
                continue
            tgt = i + j
            if tgt not in G.comp:
                continue
            for a in range(len(G.comp[i])):
                brange = range(a, len(G.comp[j])) if i == j else range(len(G.comp[j]))
                for b in brange:
                    if i == j and a == b and G.par[i][a] == 0:
                        continue
                    for t in range(len(G.comp[tgt])):
                        if G.wt[tgt][t] != wsum(G.wt[i][a], G.wt[j][b]):
                            continue
                        if (G.par[tgt][t] - G.par[i][a] - G.par[j][b] - pc) % 2:
                            continue
                        self.var[(i, a, j, b, t)] = len(self.var)
        self.n = len(self.var)

    def val(self, i, a, j, b):
        """c(e^i_a, e^j_b) as a list of linear forms in the coordinates of g_{i+j}."""
        G = self.G
        tgt = i + j
        if tgt not in G.comp:
            return []
        m = len(G.comp[tgt])
        if (min(i, j), max(i, j)) not in self.allowed:
            return None
        if (i > j) or (i == j and a > b):
            inner = self.val(j, b, i, a)
            if inner is None:
                return None
            s = -1 if (G.par[i][a] * G.par[j][b]) % 2 == 0 else 1
            return [{k: s * c for k, c in f.items()} for f in inner]
        if i == j and a == b and G.par[i][a] == 0:
            return [dict() for _ in range(m)]
        return [({self.var[(i, a, j, b, t)]: Fraction(1)}
                 if (i, a, j, b, t) in self.var else {}) for t in range(m)]


def _addbr(G, dst, i, a, m, form, factor):
    """dst += factor * [e^i_a, (form, a linear form valued in g_m)]."""
    if not form:
        return True
    tgt = i + m
    if tgt not in G.comp:
        for f in form:
            if f:
                return False
        return True
    for b, f in enumerate(form):
        if not f:
            continue
        v = G.bracket(i, a, m, b)
        for t, co in enumerate(v):
            if not co:
                continue
            for k, c in f.items():
                dst[t][k] = dst[t].get(k, Fraction(0)) + factor * co * c
    return True


def equations(G, C, triples, verbose=False):
    rows = []
    for (d0, d1, d2) in triples:
        tot = d0 + d1 + d2
        if tot not in G.comp:
            continue
        nt = len(G.comp[tot])
        for a0 in range(len(G.comp[d0])):
            r1 = range(a0, len(G.comp[d1])) if d1 == d0 else range(len(G.comp[d1]))
            for a1 in r1:
                r2 = range(a1, len(G.comp[d2])) if d2 == d1 else range(len(G.comp[d2]))
                for a2 in r2:
                    p0, p1, p2 = G.par[d0][a0], G.par[d1][a1], G.par[d2][a2]
                    w = wsum(G.wt[d0][a0], G.wt[d1][a1], G.wt[d2][a2])
                    acc = [dict() for _ in range(nt)]
                    ok = True
                    # [x0, c(x1,x2)]
                    f = C.val(d1, a1, d2, a2)
                    if f is None:
                        continue
                    s = 1 if (p0 * C.pc) % 2 == 0 else -1
                    ok &= _addbr(G, acc, d0, a0, d1 + d2, f, s)
                    # [x1, c(x0,x2)]
                    f = C.val(d0, a0, d2, a2)
                    if f is None:
                        continue
                    s = -1 if (p1 * (C.pc + p0)) % 2 == 0 else 1
                    ok &= _addbr(G, acc, d1, a1, d0 + d2, f, s)
                    # [x2, c(x0,x1)]
                    f = C.val(d0, a0, d1, a1)
                    if f is None:
                        continue
                    s = 1 if (p2 * (C.pc + p0 + p1)) % 2 == 0 else -1
                    ok &= _addbr(G, acc, d2, a2, d0 + d1, f, s)
                    if not ok:
                        continue
                    # - c([x0,x1], x2) + (-1)^{p2p1} c([x0,x2],x1)
                    #                  + (-1)^{1+p0(p1+p2)} c([x1,x2],x0)
                    for (i, ai, j, aj, k, ak, sg) in (
                            (d0, a0, d1, a1, d2, a2, Fraction(-1)),
                            (d0, a0, d2, a2, d1, a1, Fraction(1 if (p2 * p1) % 2 == 0 else -1)),
                            (d1, a1, d2, a2, d0, a0,
                             Fraction(-1 if (p0 * (p1 + p2)) % 2 == 0 else 1))):
                        v = G.bracket(i, ai, j, aj)
                        if not v or not any(v):
                            continue
                        dd = i + j
                        if (min(dd, k), max(dd, k)) not in C.allowed:
                            ok = False
                            break
                        for b, co in enumerate(v):
                            if not co:
                                continue
                            f = C.val(dd, b, k, ak)
                            if f is None:
                                ok = False
                                break
                            for t in range(nt):
                                for kk, c in f[t].items():
                                    acc[t][kk] = acc[t].get(kk, Fraction(0)) + sg * co * c
                        if not ok:
                            break
                    if not ok:
                        continue
                    for t in range(nt):
                        if G.wt[tot][t] != w:
                            continue
                        if acc[t]:
                            row = [Fraction(0)] * C.n
                            for k, c in acc[t].items():
                                row[k] += c
                            if any(row):
                                rows.append(row)
    return rows


def coboundaries(G, C, det, degs):
    """The images of the weight-0 1-cochains of degree 0, in the coordinates of `det`."""
    out = []
    for i in degs:
        if i not in G.comp:
            continue
        for a in range(len(G.comp[i])):
            for b in range(len(G.comp[i])):
                if G.wt[i][a] != G.wt[i][b]:
                    continue
                if (G.par[i][b] - G.par[i][a] - C.pc) % 2:
                    continue
                row = [Fraction(0)] * C.n
                # phi = E_{a->b} on g_i, zero elsewhere
                for (p, q) in sorted(C.allowed):
                    tgt = p + q
                    if tgt not in G.comp:
                        continue
                    for x in range(len(G.comp[p])):
                        ys = range(x, len(G.comp[q])) if p == q else range(len(G.comp[q]))
                        for y in ys:
                            val = [Fraction(0)] * len(G.comp[tgt])
                            px, py = G.par[p][x], G.par[q][y]
                            # (-1)^{px pc} [x, phi(y)]
                            if q == i and y == a:
                                s = 1 if (px * C.pc) % 2 == 0 else -1
                                v = G.bracket(p, x, q, b)
                                for t, co in enumerate(v):
                                    val[t] += s * co
                            # -(-1)^{py(pc+px)} [y, phi(x)]
                            if p == i and x == a:
                                s = -1 if (py * (C.pc + px)) % 2 == 0 else 1
                                v = G.bracket(q, y, p, b)
                                for t, co in enumerate(v):
                                    val[t] += s * co
                            # - phi([x,y])
                            if tgt == i:
                                v = G.bracket(p, x, q, y)
                                if v and v[a]:
                                    val[b] -= v[a]
                            for t in range(len(val)):
                                if not val[t]:
                                    continue
                                key = (p, x, q, y, t)
                                if key in C.var:
                                    row[C.var[key]] += val[t]
                                else:
                                    # reduce by skew-symmetry / vanishing
                                    key2 = (q, y, p, x, t)
                                    if key2 in C.var:
                                        s = -1 if (px * py) % 2 == 0 else 1
                                        row[C.var[key2]] += s * val[t]
                                    else:
                                        raise AssertionError(
                                            'coboundary outside the unknowns %s' % (key,))
                if any(row):
                    out.append(row)
    return out
