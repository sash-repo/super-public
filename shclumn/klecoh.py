#!/usr/bin/env python3
r"""H^2(g;g)_k for a W-graded transitive Lie superalgebra, exactly over Q.

Same scheme as cocyc.py (degree 0) but for an arbitrary degree k: a 2-cochain
of degree k sends g_i (x) g_j to g_{i+j+k}.  Unknowns are its components on a
chosen set of bidegrees; the equations are the cocycle identities on every
triple all of whose terms stay inside that set.  Any subset of the identities
is a necessary condition, so what is cut out is an UPPER BOUND for the space of
the restrictions of genuine cocycles to the components that determine them
(the propagation lemma).  When that bound has the dimension of the space of the
restrictions of the coboundaries, H^2(g;g)_k = 0.
"""
import sys
from fractions import Fraction

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, solve as _solve                            # noqa: E402


class Graded:
    def __init__(self, Sp, comp):
        self.S, self.comp = Sp, comp
        self.par = {i: [Sp.par(D) for D in comp[i]] for i in comp}
        self.wt = {i: [Sp.wt(D) for D in comp[i]] for i in comp}
        self._solver, self.br = {}, {}

    def _sv(self, i):
        if i not in self._solver:
            basis = self.comp[i]
            keys = sorted({k for D in basis for k in D}, key=repr)
            pos = {k: j for j, k in enumerate(keys)}
            M = [[Fraction(0)] * len(keys) for _ in basis]
            for r, D in enumerate(basis):
                for k, c in D.items():
                    M[r][pos[k]] = Fraction(c)
            MT = [[M[r][c] for r in range(len(basis))] for c in range(len(keys))]
            self._solver[i] = (pos, MT, len(basis))
        return self._solver[i]

    def coords(self, D, i):
        if i not in self.comp or not self.comp[i]:
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
    """The weight-0 2-cochains of degree k on a chosen set of bidegrees."""

    def __init__(self, G, allowed, k):
        self.G, self.allowed, self.k = G, set(allowed), k
        self.pc = k % 2
        self.var = {}
        for (i, j) in sorted(self.allowed):
            if i > j:
                continue
            tgt = i + j + k
            if tgt not in G.comp or not G.comp[tgt]:
                continue
            for a in range(len(G.comp[i])):
                brange = range(a, len(G.comp[j])) if i == j else range(len(G.comp[j]))
                for b in brange:
                    if i == j and a == b and G.par[i][a] == 0:
                        continue
                    for t in range(len(G.comp[tgt])):
                        if G.wt[tgt][t] != wsum(G.wt[i][a], G.wt[j][b]):
                            continue
                        if (G.par[tgt][t] - G.par[i][a] - G.par[j][b] - self.pc) % 2:
                            continue
                        self.var[(i, a, j, b, t)] = len(self.var)
        self.n = len(self.var)

    def val(self, i, a, j, b):
        G, tgt = self.G, i + j + self.k
        if tgt not in G.comp or not G.comp[tgt]:
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
    if tgt not in G.comp or not G.comp[tgt]:
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


def equations(G, C, triples):
    k, rows = C.k, []
    for (d0, d1, d2) in triples:
        tot = d0 + d1 + d2 + k
        if tot not in G.comp or not G.comp[tot]:
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
                    f = C.val(d1, a1, d2, a2)
                    if f is None:
                        continue
                    s = 1 if (p0 * C.pc) % 2 == 0 else -1
                    ok &= _addbr(G, acc, d0, a0, d1 + d2 + k, f, s)
                    f = C.val(d0, a0, d2, a2)
                    if f is None:
                        continue
                    s = -1 if (p1 * (C.pc + p0)) % 2 == 0 else 1
                    ok &= _addbr(G, acc, d1, a1, d0 + d2 + k, f, s)
                    f = C.val(d0, a0, d1, a1)
                    if f is None:
                        continue
                    s = 1 if (p2 * (C.pc + p0 + p1)) % 2 == 0 else -1
                    ok &= _addbr(G, acc, d2, a2, d0 + d1 + k, f, s)
                    if not ok:
                        continue
                    for (i, ai, j, aj, kk, ak, sg) in (
                            (d0, a0, d1, a1, d2, a2, Fraction(-1)),
                            (d0, a0, d2, a2, d1, a1,
                             Fraction(1 if (p2 * p1) % 2 == 0 else -1)),
                            (d1, a1, d2, a2, d0, a0,
                             Fraction(-1 if (p0 * (p1 + p2)) % 2 == 0 else 1))):
                        v = G.bracket(i, ai, j, aj)
                        if not v or not any(v):
                            continue
                        dd = i + j
                        if (min(dd, kk), max(dd, kk)) not in C.allowed:
                            ok = False
                            break
                        for b, co in enumerate(v):
                            if not co:
                                continue
                            f = C.val(dd, b, kk, ak)
                            if f is None:
                                ok = False
                                break
                            for t in range(nt):
                                for k2, c in f[t].items():
                                    acc[t][k2] = acc[t].get(k2, Fraction(0)) + sg * co * c
                        if not ok:
                            break
                    if not ok:
                        continue
                    for t in range(nt):
                        if G.wt[tot][t] != w:
                            continue
                        if acc[t]:
                            row = [Fraction(0)] * C.n
                            for k2, c in acc[t].items():
                                row[k2] += c
                            if any(row):
                                rows.append(row)
    return rows


def coboundaries(G, C, degs):
    """The restrictions of d(phi) for the weight-0 1-cochains phi of degree k."""
    k, out = C.k, []
    for i in sorted(degs):
        if i not in G.comp or i + k not in G.comp:
            continue
        if not G.comp[i] or not G.comp[i + k]:
            continue
        for a in range(len(G.comp[i])):
            for b in range(len(G.comp[i + k])):
                if G.wt[i][a] != G.wt[i + k][b]:
                    continue
                if (G.par[i + k][b] - G.par[i][a] - C.pc) % 2:
                    continue
                row = [Fraction(0)] * C.n
                bad = False
                for (p, q) in sorted(C.allowed):
                    tgt = p + q + k
                    if tgt not in G.comp or not G.comp[tgt]:
                        continue
                    for x in range(len(G.comp[p])):
                        ys = range(x, len(G.comp[q])) if p == q else range(len(G.comp[q]))
                        for y in ys:
                            val = [Fraction(0)] * len(G.comp[tgt])
                            px, py = G.par[p][x], G.par[q][y]
                            if q == i and y == a:
                                s = 1 if (px * C.pc) % 2 == 0 else -1
                                for t, co in enumerate(G.bracket(p, x, i + k, b)):
                                    val[t] += s * co
                            if p == i and x == a:
                                s = -1 if (py * (C.pc + px)) % 2 == 0 else 1
                                for t, co in enumerate(G.bracket(q, y, i + k, b)):
                                    val[t] += s * co
                            if p + q == i:
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
                                    key2 = (q, y, p, x, t)
                                    if key2 in C.var:
                                        s = -1 if (px * py) % 2 == 0 else 1
                                        row[C.var[key2]] += s * val[t]
                                    else:
                                        raise AssertionError(
                                            'coboundary outside the unknowns %s' % (key,))
                if any(row) and not bad:
                    out.append(row)
    return out


def window(k, depth):
    """The bidegrees that determine a 2-cocycle of degree k (propagation lemma)."""
    out = []
    for T in range(-depth - k, -k):
        for i in range(-depth, T + depth + 1):
            j = T - i
            if j < -depth or i > j:
                continue
            out.append((i, j))
    return out
