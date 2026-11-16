#!/usr/bin/env python3
r"""An exact model of kle(9|6;K) = (id_{sl(5)}, Pi(Lambda^2(id^*)), sl(5))^k_*.

Realization (Kac's E(5,10)):  g_ev = svect(5) = the divergence-free polynomial
vector fields in x_1..x_5,  g_od = Omega^2_cl(5) = the closed polynomial 2-forms.
Brackets:
    [X,Y]     the bracket of vector fields;
    [X,w]     = L_X w = d(iota_X w)  (w closed);
    [w,w']    = the vector field Z with  iota_Z vol = w /\ w'.
Grading:  deg(f d_i) = 2 deg f - 2,   deg(f dx_i/\dx_j) = 2 deg f - 1,
so g_{-2} = <d_i> (5|0), g_{-1} = <dx_i/\dx_j> (0|10), g_0 = sl(5) (24|0).

Elements are dicts.  Keys:
    ('v', m, alpha)  ->  x^alpha d_m           (even)
    ('f', i, j, alpha) -> x^alpha dx_i/\dx_j   (odd, i<j)
alpha is a 5-tuple of exponents.
"""
import sys
from fractions import Fraction
from itertools import combinations

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace                                        # noqa: E402

N = 5
RANGE = range(N)


def monomials(n):
    """The 5-variable monomials of total degree n, as exponent tuples."""
    if n < 0:
        return []
    out = []

    def rec(pos, left, cur):
        if pos == N - 1:
            out.append(tuple(cur + [left]))
            return
        for e in range(left + 1):
            rec(pos + 1, left - e, cur + [e])
    rec(0, n, [])
    return out


def _add(D, key, c):
    if not c:
        return
    v = D.get(key, Fraction(0)) + c
    if v:
        D[key] = v
    elif key in D:
        del D[key]


def _clean(D):
    return {k: v for k, v in D.items() if v}


# ---------------------------------------------------------------- brackets
def _vv(m1, a1, m2, a2):
    """[x^a1 d_m1, x^a2 d_m2]."""
    out = {}
    if a2[m1]:
        b = list(a2)
        b[m1] -= 1
        _add(out, ('v', m2, tuple(x + y for x, y in zip(a1, b))), Fraction(a2[m1]))
    if a1[m2]:
        b = list(a1)
        b[m2] -= 1
        _add(out, ('v', m1, tuple(x + y for x, y in zip(a2, b))), Fraction(-a1[m2]))
    return out


def _vf(m, a, i, j, b):
    r"""L_X (x^b dx_i/\dx_j) for X = x^a d_m."""
    out = {}
    # X(x^b) dx_i/\dx_j
    if b[m]:
        c = list(b)
        c[m] -= 1
        _add(out, ('f', i, j, tuple(x + y for x, y in zip(a, c))), Fraction(b[m]))
    # x^b * ( d(X^i)/\dx_j + dx_i/\d(X^j) ),  X^i = a-monomial if i == m else 0
    for (p, q, sgn) in ((i, j, 1), (j, i, -1)):
        if p != m:
            continue
        # d(x^a) /\ dx_q   (times sgn, since dx_i/\dx_j order)
        for r in RANGE:
            if not a[r]:
                continue
            c = list(a)
            c[r] -= 1
            mon = tuple(x + y for x, y in zip(b, c))
            if r == q:
                continue
            ii, jj, s = (r, q, 1) if r < q else (q, r, -1)
            _add(out, ('f', ii, jj, mon), Fraction(sgn * s * a[r]))
    return out


_EPS = {}


def _eps(p):
    """The sign of the permutation p of (0..4), 0 if repeated."""
    p = tuple(p)
    if p in _EPS:
        return _EPS[p]
    if len(set(p)) != N:
        _EPS[p] = 0
        return 0
    s, q = 1, list(p)
    for a in range(N):
        if q[a] != a:
            b = q.index(a)
            q[a], q[b] = q[b], q[a]
            s = -s
    _EPS[p] = s
    return s


def _ff(i, j, a, k, l, b):
    r"""[x^a dx_i/\dx_j, x^b dx_k/\dx_l] = Z with iota_Z vol = the wedge."""
    out = {}
    idx = [i, j, k, l]
    if len(set(idx)) != 4:
        return out
    m = [r for r in RANGE if r not in idx][0]
    # dx_i^dx_j^dx_k^dx_l = eps(i,j,k,l,m) * iota_{d_m} vol
    s = _eps((i, j, k, l, m))
    _add(out, ('v', m, tuple(x + y for x, y in zip(a, b))), Fraction(s))
    return out


def keydeg(k):
    """The degree of a basis monomial."""
    if k[0] == 'E':
        return 0
    if k[0] == 'v':
        return 2 * sum(k[2]) - 2
    return 2 * sum(k[3]) - 1


def bracket(D1, D2):
    out = {}
    for k1, c1 in D1.items():
        for k2, c2 in D2.items():
            c = c1 * c2
            if k1[0] == 'E' and k2[0] == 'E':
                continue
            elif k1[0] == 'E':
                _add(out, k2, c * Fraction(keydeg(k2)))
                continue
            elif k2[0] == 'E':
                _add(out, k1, -c * Fraction(keydeg(k1)))
                continue
            if k1[0] == 'v' and k2[0] == 'v':
                part = _vv(k1[1], k1[2], k2[1], k2[2])
            elif k1[0] == 'v' and k2[0] == 'f':
                part = _vf(k1[1], k1[2], k2[1], k2[2], k2[3])
            elif k1[0] == 'f' and k2[0] == 'v':
                part = {k: -v for k, v in
                        _vf(k2[1], k2[2], k1[1], k1[2], k1[3]).items()}
            else:
                part = _ff(k1[1], k1[2], k1[3], k2[1], k2[2], k2[3])
            for k, v in part.items():
                _add(out, k, c * v)
    return _clean(out)


def par(D):
    ps = {1 if k[0] == 'f' else 0 for k in D}
    assert len(ps) == 1, 'inhomogeneous parity'
    return ps.pop()


def wt(D):
    """The sl(5)-weight, as the 4-tuple (mu_0-mu_4, .., mu_3-mu_4)."""
    ws = set()
    for k in D:
        if k[0] == 'E':
            ws.add((0, 0, 0, 0))
            continue
        mu = [0] * N
        if k[0] == 'v':
            for r in RANGE:
                mu[r] += k[2][r]
            mu[k[1]] -= 1
        else:
            for r in RANGE:
                mu[r] += k[3][r]
            mu[k[1]] += 1
            mu[k[2]] += 1
        ws.add(tuple(mu[r] - mu[4] for r in range(4)))
    assert len(ws) == 1, 'inhomogeneous weight'
    return ws.pop()


# ---------------------------------------------------------------- components
def _basis_from_kernel(gens, rows_of_map, ncols):
    """gens[t] -> its image column; return the kernel vectors as dicts."""
    ker = nullspace(rows_of_map)
    out = []
    for v in ker:
        D = {}
        for t, c in enumerate(v):
            if c:
                _add(D, gens[t], Fraction(c))
        out.append(_clean(D))
    return out


def even_component(n):
    """Divergence-free vector fields with coefficients of polynomial degree n."""
    gens = [('v', m, a) for m in RANGE for a in monomials(n)]
    tgt = {a: t for t, a in enumerate(monomials(n - 1))} if n >= 1 else {}
    rows = [[Fraction(0)] * len(gens) for _ in range(len(tgt))]
    for t, (_, m, a) in enumerate(gens):
        if a[m]:
            b = list(a)
            b[m] -= 1
            rows[tgt[tuple(b)]][t] = Fraction(a[m])
    if not rows:
        return [{g: Fraction(1)} for g in gens]
    return _basis_from_kernel(gens, rows, len(gens))


def odd_component(n):
    """Closed 2-forms with coefficients of polynomial degree n."""
    gens = [('f', i, j, a) for (i, j) in combinations(RANGE, 2) for a in monomials(n)]
    tri = [(i, j, k) for (i, j, k) in combinations(RANGE, 3)]
    tgt = {(t, a): r for r, (t, a) in
           enumerate([(t, a) for t in tri for a in monomials(n - 1)])}
    rows = [[Fraction(0)] * len(gens) for _ in range(len(tgt))]
    for c, (_, i, j, a) in enumerate(gens):
        for r in RANGE:
            if not a[r] or r in (i, j):
                continue
            b = list(a)
            b[r] -= 1
            trip = tuple(sorted((r, i, j)))
            s = _eps3(r, i, j)
            rows[tgt[(trip, tuple(b))]][c] += Fraction(s * a[r])
    if not rows:
        return [{g: Fraction(1)} for g in gens]
    return _basis_from_kernel(gens, rows, len(gens))


def _eps3(r, i, j):
    r"""Sign taking dx_r/\dx_i/\dx_j to the sorted order."""
    p = [r, i, j]
    s = 1
    for a in range(3):
        for b in range(a + 1, 3):
            if p[a] > p[b]:
                p[a], p[b] = p[b], p[a]
                s = -s
    return s


_COMP = {}


def comp(d):
    """A basis of g_d, split into weight vectors (each basis element is one)."""
    if d in _COMP:
        return _COMP[d]
    if d < -2:
        _COMP[d] = []
        return []
    if d % 2 == 0:
        raw = even_component(d // 2 + 1)
    else:
        raw = odd_component((d + 1) // 2)
    out = _split_weights(raw)
    _COMP[d] = out
    return out


def _split_weights(raw):
    """Refine a basis so that every vector is a weight vector."""
    groups = {}
    for D in raw:
        for k in D:
            groups.setdefault(_keywt(k), []).append(D)
            break
    # a basis vector may mix weights: re-split monomial-wise
    allkeys = sorted({k for D in raw for k in D}, key=repr)
    bykey = {}
    for k in allkeys:
        bykey.setdefault(_keywt(k), []).append(k)
    out = []
    for w, keys in sorted(bykey.items()):
        pos = {k: t for t, k in enumerate(keys)}
        rows = []
        for D in raw:
            row = [Fraction(0)] * len(keys)
            hit = False
            for k, c in D.items():
                if k in pos:
                    row[pos[k]] = c
                    hit = True
            if hit:
                rows.append(row)
        if not rows:
            continue
        from exactlin import rref as _rref
        R, _piv = _rref(rows)
        for row in R:
            if any(row):
                out.append(_clean({keys[t]: row[t] for t in range(len(keys)) if row[t]}))
    return out


def _keywt(k):
    if k[0] == 'E':
        return (0, 0, 0, 0)
    mu = [0] * N
    if k[0] == 'v':
        for r in RANGE:
            mu[r] += k[2][r]
        mu[k[1]] -= 1
    else:
        for r in RANGE:
            mu[r] += k[3][r]
        mu[k[1]] += 1
        mu[k[2]] += 1
    return tuple(mu[r] - mu[4] for r in range(4))


def hatcomp(d):
    """kle(9|6;K) + C.E, the grading operator adjoined."""
    out = list(comp(d))
    if d == 0:
        out.append({('E',): Fraction(1)})
    return out
