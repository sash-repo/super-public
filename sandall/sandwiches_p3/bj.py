#!/usr/bin/env python3
r"""
bj.py --- the Bouarroudj superalgebra Bj(3; \One|2) over F_3 and its two
W-regradings Bj(3; \un|3) and ~Bj(3; \un|3), built from the definition given in
in Krutov, Leites, Shchepochkina, arXiv:2309.16370 [KLS].

k(3; \One|2) is realised by generating functions in t, p, q (even, divided
powers of height 1, so t^(3) = 0) and xi, eta (odd):

   {f, g} = (2 - E)(f) d_t g - d_t f (2 - E)(g) - {f, g}_P.b.,
   {f, g}_P.b. = d_p f d_q g - d_q f d_p g - (-1)^p(f)(d_xi f d_eta g
                                                       + d_eta f d_xi g),
   E = p d_p + q d_q + xi d_xi + eta d_eta,

normalised so that {p, q} = {xi, eta} = -1, as the text has it.  Bj is the
Cartan prolong of k(3|2)_{<=0} + V + W_eta inside k(3|2).

Everything is exact over F_3.
"""

import itertools
import numpy as np

P = 3
EVEN = ("t", "p", "q")
ODD = ("xi", "eta")


# --------------------------------------------------------------- the algebra O
MONS = [(a, b, c, d, e)
        for a in range(P) for b in range(P) for c in range(P)
        for d in range(2) for e in range(2)]
IDX = {m: i for i, m in enumerate(MONS)}
N = len(MONS)


def par(m):
    return (m[3] + m[4]) % 2


PAR = [par(m) for m in MONS]


def binom(a, b):
    n, k = a + b, a
    r = 1
    for t in range(k):
        r = r * (n - t) // (t + 1)
    return r % P


def mono_mul(m1, m2):
    """m1 * m2 in O, as (coefficient, monomial) or (0, None)."""
    a = tuple(x + y for x, y in zip(m1[:3], m2[:3]))
    if any(x > P - 1 for x in a):
        return 0, None
    if m1[3] + m2[3] > 1 or m1[4] + m2[4] > 1:
        return 0, None
    c = 1
    for x, y in zip(m1[:3], m2[:3]):
        c = c * binom(x, y) % P
    # sign: move xi^{d2} past eta^{e1}
    if m1[4] and m2[3]:
        c = -c % P
    return c % P, (a[0], a[1], a[2], m1[3] + m2[3], m1[4] + m2[4])


def mul(f, g):
    out = np.zeros(N, dtype=np.int64)
    for i, fi in enumerate(f):
        if fi % P:
            for j, gj in enumerate(g):
                if gj % P:
                    c, m = mono_mul(MONS[i], MONS[j])
                    if c:
                        out[IDX[m]] = (out[IDX[m]] + fi * gj * c) % P
    return out % P


def d(k, f):
    """d/d z_k, k = 0..4 for t, p, q, xi, eta.  Odd derivatives act from the
    left, with the Koszul sign of moving past the preceding odd generator."""
    out = np.zeros(N, dtype=np.int64)
    for i, fi in enumerate(f):
        if not fi % P:
            continue
        m = MONS[i]
        if k < 3:
            if m[k] > 0:
                mm = list(m)
                mm[k] -= 1
                out[IDX[tuple(mm)]] = (out[IDX[tuple(mm)]] + fi) % P
        else:
            j = k - 3
            if m[3 + j] == 1:
                mm = list(m)
                mm[3 + j] = 0
                s = 1 if j == 0 else (-1 if m[3] else 1)   # move past xi
                out[IDX[tuple(mm)]] = (out[IDX[tuple(mm)]] + s * fi) % P
    return out % P


def euler(f):
    """E(f) = (p d_p + q d_q + xi d_xi + eta d_eta)(f): multiplies the monomial
    by its degree in p, q, xi, eta."""
    out = np.zeros(N, dtype=np.int64)
    for i, fi in enumerate(f):
        if fi % P:
            m = MONS[i]
            out[i] = (fi * (m[1] + m[2] + m[3] + m[4])) % P
    return out % P


def kbracket(f, g, pf=None):
    """the contact bracket; pf = parity of f (if f is homogeneous)."""
    if pf is None:
        pf = parity(f)
    two_E_f = (2 * np.array(f) - euler(f)) % P
    two_E_g = (2 * np.array(g) - euler(g)) % P
    out = (mul(two_E_f, d(0, g)) - mul(d(0, f), two_E_g)) % P
    pb = (mul(d(1, f), d(2, g)) - mul(d(2, f), d(1, g))) % P
    odd = (mul(d(3, f), d(4, g)) + mul(d(4, f), d(3, g))) % P
    pb = (pb - (-1) ** pf * odd) % P
    return (out - pb) % P


def parity(f):
    ps = {PAR[i] for i, c in enumerate(f) if c % P}
    assert len(ps) <= 1, "not parity-homogeneous"
    return ps.pop() if ps else 0


# ---------------------------------------------------------------- linear algebra
def rref(rows):
    M = np.array(rows, dtype=np.int64) % P if len(rows) else np.zeros((0, N), np.int64)
    if M.size == 0:
        return M.reshape(0, N if M.ndim < 2 else M.shape[1]), []
    piv, r = [], 0
    for c in range(M.shape[1]):
        nz = [i for i in range(r, M.shape[0]) if M[i, c] % P]
        if not nz:
            continue
        i = nz[0]
        if i != r:
            M[[r, i]] = M[[i, r]]
        M[r] = M[r] * pow(int(M[r, c]), P - 2, P) % P
        for i in range(M.shape[0]):
            if i != r and M[i, c] % P:
                M[i] = (M[i] - M[i, c] * M[r]) % P
        piv.append(c)
        r += 1
    return M[:r] % P, piv


def reduce_by(R, piv, v):
    v = np.array(v, dtype=np.int64) % P
    for i, c in enumerate(piv):
        if v[c]:
            v = (v - v[c] * R[i]) % P
    return v


def inside(R, piv, v):
    return not reduce_by(R, piv, v).any()


# ------------------------------------------------------------------ degrees
def deg_std(m):
    """the standard k-grading: deg t = 2, deg p = deg q = deg xi = deg eta = 1,
    gr(f) = deg(f) - 2."""
    return 2 * m[0] + m[1] + m[2] + m[3] + m[4] - 2


def deg_A(m):
    """Bj(3; \\un|3): deg xi = 0, deg t = deg eta = 2, deg p = deg q = 1."""
    return 2 * m[0] + m[1] + m[2] + 0 * m[3] + 2 * m[4] - 2


def deg_B(m):
    """~Bj(3; \\un|3): deg eta = 0, deg t = deg xi = 2, deg p = deg q = 1."""
    return 2 * m[0] + m[1] + m[2] + 2 * m[3] + 0 * m[4] - 2


def e(name):
    """the generating function of a monomial written as a string, e.g. 'p2q'."""
    raise NotImplementedError


def mvec(a=0, b=0, c=0, dd=0, ee=0, coef=1):
    v = np.zeros(N, dtype=np.int64)
    v[IDX[(a, b, c, dd, ee)]] = coef % P
    return v


def lincomb(*terms):
    v = np.zeros(N, dtype=np.int64)
    for coef, m in terms:
        v = (v + coef * mvec(*m)) % P
    return v


# ------------------------------------------------------------------- the pieces
def std_component(i):
    """basis of k(3; \\One|2)_i in the standard grading."""
    return [mvec(*m) for m in MONS if deg_std(m) == i]


V_BASIS = [mvec(1, 1, 0, 0, 0), mvec(1, 0, 1, 0, 0),
           mvec(1, 0, 0, 1, 0), mvec(1, 0, 0, 0, 1)]         # tp, tq, t xi, t eta

W_XI = [lincomb((1, (0, 2, 1, 0, 0)), (-1, (0, 1, 0, 1, 1))),   # p^(2)q - p xi eta
        lincomb((1, (0, 1, 2, 0, 0)), (1, (0, 0, 1, 1, 1))),    # pq^(2) + q xi eta
        mvec(0, 2, 0, 1, 0), mvec(0, 1, 1, 1, 0), mvec(0, 0, 2, 1, 0)]

W_ETA = [lincomb((1, (0, 1, 2, 0, 0)), (-1, (0, 0, 1, 1, 1))),  # pq^(2) - q xi eta
         lincomb((1, (0, 2, 1, 0, 0)), (1, (0, 1, 0, 1, 1))),   # p^(2)q + p xi eta
         mvec(0, 2, 0, 0, 1), mvec(0, 1, 1, 0, 1), mvec(0, 0, 2, 0, 1)]


def prolong(first):
    """the Cartan prolong of k_{<=0} + first inside k(3; \\One|2)."""
    g = {}
    for i in (-2, -1, 0):
        g[i] = std_component(i)
    g[1] = [np.array(v) % P for v in first]
    gm1 = g[-1]
    top = max(deg_std(m) for m in MONS)
    for i in range(2, top + 1):
        prev, ppiv = rref(g[i - 1])
        cand = std_component(i)
        if not cand:
            g[i] = []
            continue
        rows = []
        for y in gm1:
            imgs = [reduce_by(prev, ppiv, kbracket(y, c, pf=parity(y))) for c in cand]
            for coord in range(N):
                r = [int(im[coord]) for im in imgs]
                if any(r):
                    rows.append(r)
        if rows:
            R, piv = rref(rows)
            free = [c for c in range(len(cand)) if c not in piv]
            sol = []
            for f in free:
                s = np.zeros(len(cand), dtype=np.int64)
                s[f] = 1
                for t, c in enumerate(piv):
                    s[c] = (-R[t, f]) % P
                sol.append(s)
        else:
            sol = [np.eye(len(cand), dtype=np.int64)[k] for k in range(len(cand))]
        comp = []
        for s in sol:
            v = np.zeros(N, dtype=np.int64)
            for co, c in zip(s, cand):
                if co:
                    v = (v + co * c) % P
            if v.any():
                comp.append(v)
        g[i] = comp
    return g
