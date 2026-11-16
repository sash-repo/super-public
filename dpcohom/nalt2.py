#!/usr/bin/env python3
"""Non-alternating symplectic forms over divided power algebras, p = 2.

An analog of Skryabin's classification of the normal shapes of symplectic
forms, for p = 2 and NON-ALTERNATING forms.

Everything here is exact over F_2 (the prime field; for the classification
statements the field enters only through square roots, and F_2 is perfect).

Set-up
------
O(m;u) = divided power algebra, basis x^(alpha), 0 <= alpha_i < 2^{u_i},
    x^(alpha) x^(beta) = binom(alpha+beta, alpha) x^(alpha+beta),
and at p = 2, binom(a+b,a) = 1 mod 2 iff a & b == 0 bitwise (Kummer), so the
product is x^(alpha+beta) iff alpha_i & beta_i == 0 for every i (and the sum is
in range), else 0.  In particular x_i^2 = 0: SQUARES IN O ARE CONSTANTS.

Differential forms at p = 2 must be taken with divided powers in the dx_i as
well (the divided power de Rham complex): dx_i * dx_i = 2 dx_i^{(2)} = 0, so a
non-alternating 2-form needs the separate generator dx_i^{(2)}.  Thus

    Omega^2 = { sum_{i<j} a_{ij} dx_i dx_j  +  sum_i b_i dx_i^{(2)} },

the alternating forms being those with b = 0.  The Gram matrix of the
associated symmetric bilinear form on the vector fields is G_{ij} = a_{ij}
(i != j), G_{ii} = b_i; the form is non-degenerate iff det G is invertible in
O, i.e. iff its constant term is non-zero.

Usage:  python nalt2.py
"""
import sys, os, itertools
from itertools import combinations

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace, rank, backend            # noqa: E402


# --------------------------------------------------------------- the algebra

class Alg:
    """O(m; u) over F_2.  Elements are int bitmasks over the monomial basis."""

    def __init__(self, m, u=None):
        self.m = m
        self.u = tuple(u) if u is not None else (1,) * m
        self.N = tuple(2 ** k for k in self.u)
        self.mons = [tuple(a) for a in itertools.product(*[range(n) for n in self.N])]
        self.idx = {a: i for i, a in enumerate(self.mons)}
        self.dim = len(self.mons)
        self.one = 1 << self.idx[(0,) * m]
        self._mul = {}
        self._par = [self._partial_table(i) for i in range(m)]

    # -- monomial level ----------------------------------------------------
    def _mon_mul(self, a, b):
        c = []
        for i in range(self.m):
            if a[i] & b[i]:                      # a carry => binomial even
                return None
            s = a[i] + b[i]
            if s >= self.N[i]:                   # outside the shearing range
                return None
            c.append(s)
        return tuple(c)

    def _partial_table(self, i):
        t = []
        for a in self.mons:
            if a[i] == 0:
                t.append(0)
            else:
                b = list(a); b[i] -= 1
                t.append(1 << self.idx[tuple(b)])
        return t

    # -- element level -----------------------------------------------------
    def mul(self, f, g):
        if f == 0 or g == 0:
            return 0
        key = (f, g)
        r = self._mul.get(key)
        if r is not None:
            return r
        out = 0
        ff = f
        while ff:
            lo = ff & -ff
            ia = lo.bit_length() - 1
            ff ^= lo
            gg = g
            a = self.mons[ia]
            while gg:
                lo2 = gg & -gg
                ib = lo2.bit_length() - 1
                gg ^= lo2
                c = self._mon_mul(a, self.mons[ib])
                if c is not None:
                    out ^= 1 << self.idx[c]
        self._mul[key] = out
        return out

    def partial(self, i, f):
        out = 0
        t = self._par[i]
        while f:
            lo = f & -f
            k = lo.bit_length() - 1
            f ^= lo
            out ^= t[k]
        return out

    def const_term(self, f):
        return (f >> self.idx[(0,) * self.m]) & 1

    def gen(self, i):
        a = [0] * self.m; a[i] = 1
        return 1 << self.idx[tuple(a)]

    def mono(self, a):
        return 1 << self.idx[tuple(a)]

    def show(self, f):
        if f == 0:
            return "0"
        parts = []
        for k, a in enumerate(self.mons):
            if (f >> k) & 1:
                if all(e == 0 for e in a):
                    parts.append("1")
                else:
                    s = []
                    for i, e in enumerate(a):
                        if e == 1:
                            s.append("x%d" % (i + 1))
                        elif e > 1:
                            s.append("x%d^(%d)" % (i + 1, e))
                    parts.append("".join(s))
        return "+".join(parts)


# ------------------------------------------------------------------- forms
# A 2-form is a dict  key -> element of O, with keys
#     ('a', i, j)  for i < j   ->  coefficient of dx_i dx_j
#     ('b', i)                 ->  coefficient of dx_i^{(2)}
# A 3-form has keys
#     ('A', i, j, k) i<j<k     ->  dx_i dx_j dx_k
#     ('B', i, k)    k != i    ->  dx_i^{(2)} dx_k
#     ('C', i)                 ->  dx_i^{(3)}

def g2_basis(m):
    return [('a', i, j) for i, j in combinations(range(m), 2)] + \
           [('b', i) for i in range(m)]


def g3_basis(m):
    out = [('A', i, j, k) for i, j, k in combinations(range(m), 3)]
    out += [('B', i, k) for i in range(m) for k in range(m) if k != i]
    out += [('C', i) for i in range(m)]
    return out


def addk(d, key, val):
    if val:
        d[key] = d.get(key, 0) ^ val
        if d[key] == 0:
            del d[key]


def addform(d, e):
    for k, v in e.items():
        addk(d, k, v)


def smul(alg, c, form):
    return {k: alg.mul(c, v) for k, v in form.items() if alg.mul(c, v)}


def d1(alg, one_form):
    """d of a 1-form given as a list [g_0..g_{m-1}] (coefficients of dx_i)."""
    out = {}
    for i, g in enumerate(one_form):
        if not g:
            continue
        for k in range(alg.m):
            if k == i:
                continue                      # dx_k dx_k = 0
            p = alg.partial(k, g)
            if p:
                key = ('a', min(k, i), max(k, i))
                addk(out, key, p)
    return out


def d2(alg, om):
    """d of a 2-form -> 3-form."""
    out = {}
    for key, c in om.items():
        if key[0] == 'a':
            _, i, j = key
            for k in range(alg.m):
                if k == i or k == j:
                    continue                  # dx_k dx_i dx_j with a repeat = 0
                p = alg.partial(k, c)
                if p:
                    a, b, cc = sorted((k, i, j))
                    addk(out, ('A', a, b, cc), p)
        else:
            _, i = key
            for k in range(alg.m):
                p = alg.partial(k, c)
                if not p:
                    continue
                if k == i:
                    addk(out, ('C', i), p)     # dx_i dx_i^{(2)} = 3 dx_i^{(3)}
                else:
                    addk(out, ('B', i, k), p)
    return out


def mul1(alg, p, q):
    """product of two 1-forms (lists) -> 2-form (only 'a' keys survive)."""
    out = {}
    for k in range(alg.m):
        for l in range(k + 1, alg.m):
            v = alg.mul(p[k], q[l]) ^ alg.mul(p[l], q[k])
            addk(out, ('a', k, l), v)
    return out


def dvec(alg, f):
    return [alg.partial(k, f) for k in range(alg.m)]


def unit1(alg, i):
    v = [0] * alg.m
    v[i] = alg.one
    return v


def act(alg, D, f):
    """D(f) for a vector field D = [f_0..f_{m-1}]."""
    out = 0
    for k in range(alg.m):
        if D[k]:
            out ^= alg.mul(D[k], alg.partial(k, f))
    return out


def lie(alg, D, om):
    """Lie derivative L_D(omega), omega a 2-form.

    L_D is the derivation with L_D(f) = D(f), L_D(dx_i) = d(f_i) and, on the
    divided square, L_D(dx_i^{(2)}) = d(f_i) dx_i.
    """
    out = {}
    for key, c in om.items():
        if key[0] == 'a':
            _, i, j = key
            addk(out, key, act(alg, D, c))
            addform(out, smul(alg, c, mul1(alg, dvec(alg, D[i]), unit1(alg, j))))
            addform(out, smul(alg, c, mul1(alg, unit1(alg, i), dvec(alg, D[j]))))
        else:
            _, i = key
            addk(out, key, act(alg, D, c))
            addform(out, smul(alg, c, mul1(alg, dvec(alg, D[i]), unit1(alg, i))))
    return out


def bracket(alg, D, E):
    return [act(alg, D, E[i]) ^ act(alg, E, D[i]) for i in range(alg.m)]


# --------------------------------------------------- non-degeneracy, parity

def gram(alg, om):
    m = alg.m
    G = [[0] * m for _ in range(m)]
    for key, c in om.items():
        if key[0] == 'a':
            _, i, j = key
            G[i][j] = c; G[j][i] = c
        else:
            _, i = key
            G[i][i] = c
    return G


def det(alg, G):
    m = len(G)
    out = 0
    for perm in itertools.permutations(range(m)):
        t = alg.one
        for i, j in enumerate(perm):
            t = alg.mul(t, G[i][j])
            if t == 0:
                break
        out ^= t                                # char 2: no signs
    return out


def nondeg(alg, om):
    return alg.const_term(det(alg, gram(alg, om))) == 1


def alternating(om):
    return not any(k[0] == 'b' and v for k, v in om.items())


# ------------------------------------------------------- linear algebra glue

def flatten2(alg, om, basis):
    v = []
    for k in basis:
        c = om.get(k, 0)
        v.extend((c >> t) & 1 for t in range(alg.dim))
    return v


def closed_space(alg):
    """basis of Z^2 = ker(d: Omega^2 -> Omega^3), as 2-forms."""
    b2, b3 = g2_basis(alg.m), g3_basis(alg.m)
    cols = []
    gens = []
    for k in b2:
        for t in range(alg.dim):
            om = {k: 1 << t}
            gens.append(om)
            cols.append(flatten2(alg, d2(alg, om), b3))
    # rows of the matrix = the transpose of the column list
    M = [[cols[c][r] for c in range(len(cols))] for r in range(len(cols[0]))] \
        if cols and cols[0] else [[0] * len(cols)]
    ns = nullspace(M, p=2)
    out = []
    for v in ns:
        om = {}
        for c, coef in enumerate(v):
            if int(coef) % 2:
                addform(om, gens[c])
        out.append(om)
    return out


def preserving_algebra(alg, om):
    """basis of h(omega) = { D in vect(m;u) | L_D omega = 0 }."""
    b2 = g2_basis(alg.m)
    gens, cols = [], []
    for i in range(alg.m):
        for t in range(alg.dim):
            D = [0] * alg.m
            D[i] = 1 << t
            gens.append(D)
            cols.append(flatten2(alg, lie(alg, D, om), b2))
    M = [[cols[c][r] for c in range(len(cols))] for r in range(len(cols[0]))]
    ns = nullspace(M, p=2)
    out = []
    for v in ns:
        D = [0] * alg.m
        for c, coef in enumerate(v):
            if int(coef) % 2:
                for i in range(alg.m):
                    D[i] ^= gens[c][i]
        out.append(D)
    return out


def span_dim(alg, vs):
    """dimension over F_2 of the span of a list of vector fields."""
    rows = []
    for D in vs:
        r = []
        for i in range(alg.m):
            r.extend((D[i] >> t) & 1 for t in range(alg.dim))
        rows.append(r)
    if not rows:
        return 0
    return rank(rows, p=2)


def derived(alg, basis):
    out = []
    for i in range(len(basis)):
        for j in range(i, len(basis)):
            out.append(bracket(alg, basis[i], basis[j]))
    return out


def derived_series(alg, basis, steps=3):
    """dims of h, h^(1), h^(2), ... (each computed as the span of brackets)."""
    dims = [span_dim(alg, basis)]
    cur = basis
    for _ in range(steps):
        br = derived(alg, cur)
        # reduce to a basis of the span before iterating
        cur = reduce_basis(alg, br)
        dims.append(len(cur))
        if len(cur) == dims[-2]:
            break
    return dims


def is_simple(alg, basis):
    """is the Lie algebra spanned by `basis` (over F_2) simple?

    Checks that it is non-abelian and that the ideal generated by every
    non-zero element is the whole algebra.  Feasible only for small dim.
    """
    n = len(basis)
    if n <= 1:
        return False
    if not reduce_basis(alg, derived(alg, basis)):
        return False                                  # abelian
    for bits in range(1, 1 << n):
        v = [0] * alg.m
        for i in range(n):
            if (bits >> i) & 1:
                for k in range(alg.m):
                    v[k] ^= basis[i][k]
        ideal = [v]
        grew = True
        while grew:
            grew = False
            new = reduce_basis(alg, ideal +
                               [bracket(alg, b, w) for b in basis for w in ideal])
            if len(new) > len(ideal):
                ideal, grew = new, True
        if len(ideal) < n:
            return False
    return True


def reduce_basis(alg, vs):
    """row-reduce a list of vector fields over F_2, return an independent set."""
    piv = {}
    out = []
    for D in vs:
        w = 0
        for i in range(alg.m):
            w |= D[i] << (i * alg.dim)
        while w:
            hb = w.bit_length() - 1
            if hb in piv:
                w ^= piv[hb]
            else:
                piv[hb] = w
                break
        if w:
            out.append(w)
    res = []
    for w in sorted(piv.values()):
        D = [(w >> (i * alg.dim)) & ((1 << alg.dim) - 1) for i in range(alg.m)]
        res.append(D)
    return res


# ------------------------------------------------------------- automorphisms
# An automorphism of O(m; 1) over F_2 is x_i -> f_i, f_i in the maximal ideal,
# with invertible linear part.  (For u = 1 no divided powers of the f_i are
# needed, so substitution is an ordinary algebra map.)

def subst(alg, phi, f):
    """phi = [f_1..f_m]; substitute into f in O(m; 1)."""
    assert all(k == 1 for k in alg.u), "substitution implemented for u = 1"
    out = 0
    for k, a in enumerate(alg.mons):
        if not ((f >> k) & 1):
            continue
        t = alg.one
        for i, e in enumerate(a):
            if e:
                t = alg.mul(t, phi[i])
        out ^= t
    return out


def pullback(alg, phi, om):
    """phi^*(omega)."""
    df = [dvec(alg, phi[i]) for i in range(alg.m)]
    out = {}
    for key, c in om.items():
        cc = subst(alg, phi, c)
        if not cc:
            continue
        if key[0] == 'a':
            _, i, j = key
            addform(out, smul(alg, cc, mul1(alg, df[i], df[j])))
        else:
            _, i = key
            # (df_i)^{(2)} = sum_k (d_k f_i)^2 dx_k^{(2)} + sum_{k<l} ... dx_k dx_l
            sq = {}
            for k in range(alg.m):
                s = alg.mul(df[i][k], df[i][k])
                addk(sq, ('b', k), s)
            for k in range(alg.m):
                for l in range(k + 1, alg.m):
                    addk(sq, ('a', k, l), alg.mul(df[i][k], df[i][l]))
            addform(out, smul(alg, cc, sq))
    return out


def automorphisms(alg, linear_only=False):
    """all automorphisms of O(m;1) over F_2 (m small!)."""
    m = alg.m
    maximal = [1 << k for k, a in enumerate(alg.mons) if any(a)]
    if linear_only:
        cand = [alg.gen(i) for i in range(m)]
        pool = [0]
        space = []
        for c in cand:
            space.append(c)
        # all linear combinations of the x_i
        lin = []
        for bits in range(1 << m):
            v = 0
            for i in range(m):
                if (bits >> i) & 1:
                    v ^= alg.gen(i)
            lin.append(v)
        choices = [lin] * m
    else:
        allm = []
        for bits in range(1 << len(maximal)):
            v = 0
            for i, b in enumerate(maximal):
                if (bits >> i) & 1:
                    v ^= b
            allm.append(v)
        choices = [allm] * m
    out = []
    for phi in itertools.product(*choices):
        L = [[(phi[i] >> alg.idx[tuple(1 if t == k else 0 for t in range(m))]) & 1
              for k in range(m)] for i in range(m)]
        if rank(L, p=2) == m:
            out.append(list(phi))
    return out
