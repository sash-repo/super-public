#!/usr/bin/env python3
"""Exact engine for b(n), sb(n)=Ker Delta, sle(n) and sle'(n).

Everything is built from scratch over Q (integer coefficients suffice):

    b(n)  = Pi( C[q_1..q_n, xi_1..xi_n] ) with the Buttin (anti)bracket,
    Delta = sum_i d/dq_i d/dxi_i          (the odd Laplacian / BV operator),
    (f,g) = Delta(fg) - (Delta f)g - (-1)^{|f|} f (Delta g),
    sb(n) = Ker Delta,  sle(n) = sb(n)/C.1,  sle'(n) = [sle(n),sle(n)].

The bracket is *derived* from Delta, so the BV identities are true by
construction; they are then re-checked numerically (Jacobi, Leibniz,
Delta a derivation of the bracket), which is the real test of the code.

Grading: t = total polynomial degree; the vectorial grading of le(n)/sle(n)
is g_k = (degree k+2 part), so g_{-1} = linear, g_0 = quadratic.

Usage:  python3 sle.py [nmax] [tmax]
"""
import sys
from itertools import combinations
from fractions import Fraction


# ---------------------------------------------------------------- monomials
# A monomial is (a, S): a = tuple of q-exponents (len n), S = sorted tuple of
# xi-indices.  An element is a dict {monomial: coefficient}.

def parity(mon):
    return len(mon[1]) % 2


def mul(m1, m2):
    """Product of two monomials -> (monomial, sign) or None if it vanishes."""
    a1, s1 = m1
    a2, s2 = m2
    if set(s1) & set(s2):
        return None
    a = tuple(x + y for x, y in zip(a1, a2))
    # sign of sorting s1 + s2
    merged = list(s1) + list(s2)
    sign = 1
    # bubble sort counting transpositions
    arr = merged[:]
    for i in range(len(arr)):
        for j in range(len(arr) - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                sign = -sign
    return (a, tuple(arr)), sign


def add_into(d, mon, c):
    if c == 0:
        return
    v = d.get(mon, 0) + c
    if v == 0:
        d.pop(mon, None)
    else:
        d[mon] = v


def poly_mul(f, g):
    out = {}
    for m1, c1 in f.items():
        for m2, c2 in g.items():
            r = mul(m1, m2)
            if r is None:
                continue
            m, s = r
            add_into(out, m, s * c1 * c2)
    return out


def d_dq(f, i):
    out = {}
    for (a, s), c in f.items():
        if a[i] == 0:
            continue
        b = list(a)
        k = b[i]
        b[i] -= 1
        add_into(out, (tuple(b), s), c * k)
    return out


def d_dxi(f, i):
    """Left derivative with respect to xi_i."""
    out = {}
    for (a, s), c in f.items():
        if i not in s:
            continue
        pos = s.index(i)
        sign = (-1) ** pos
        t = tuple(x for x in s if x != i)
        add_into(out, (a, t), sign * c)
    return out


def delta(f, n):
    out = {}
    for i in range(n):
        for m, c in d_dxi(d_dq(f, i), i).items():
            add_into(out, m, c)
    return out


def scal(f, k):
    return {m: k * c for m, c in f.items() if k * c != 0}


def padd(*fs):
    out = {}
    for f in fs:
        for m, c in f.items():
            add_into(out, m, c)
    return out


def bracket(f, g, n):
    """The Buttin (anti)bracket

        {f,g} = sum_i ( df/dq_i dg/dxi_i + (-1)^{p(f)} df/dxi_i dg/dq_i ),

    p = Grassmann degree mod 2.  It is odd: p({f,g}) = p(f)+p(g)+1, so the
    Lie superalgebra b(n) is Pi(C[q,xi]) and its parity is ~p = p+1.
    """
    pf = homog_parity(f)
    out = {}
    for i in range(n):
        for m, c in poly_mul(d_dq(f, i), d_dxi(g, i)).items():
            add_into(out, m, c)
        for m, c in poly_mul(d_dxi(f, i), d_dq(g, i)).items():
            add_into(out, m, ((-1) ** pf) * c)
    return out


def bracket_bv(f, g, n):
    """The same bracket read off Delta (BV master formula), for cross-check."""
    pf = homog_parity(f)
    t1 = delta(poly_mul(f, g), n)
    t2 = poly_mul(delta(f, n), g)
    t3 = poly_mul(f, delta(g, n))
    return scal(padd(t1, scal(t2, -1), scal(t3, -(-1) ** pf)), (-1) ** pf)


def homog_parity(f):
    ps = {parity(m) for m in f}
    assert len(ps) <= 1, "non-homogeneous element"
    return ps.pop() if ps else 0


# ---------------------------------------------------------------- bases
def monomials(n, t):
    """All monomials of total degree t."""
    out = []
    for k in range(0, min(t, n) + 1):
        for S in combinations(range(n), k):
            for a in compositions(n, t - k):
                out.append((a, S))
    return sorted(out)


def compositions(n, t):
    if n == 0:
        if t == 0:
            yield ()
        return
    for i in range(t + 1):
        for rest in compositions(n - 1, t - i):
            yield (i,) + rest


# ---------------------------------------------------------------- linear algebra over Q
def rref(rows, ncols):
    """Row-reduce a list of dicts {col: value}; returns list of pivot rows."""
    rows = [dict((k, Fraction(v)) for k, v in r.items() if v) for r in rows]
    piv = {}
    out = []
    for r in rows:
        r = dict(r)
        while r:
            c = min(r)
            if c in piv:
                f = r[c] / piv[c][c]
                for k, v in piv[c].items():
                    nv = r.get(k, 0) - f * v
                    if nv == 0:
                        r.pop(k, None)
                    else:
                        r[k] = nv
            else:
                piv[c] = r
                out.append(r)
                break
    return out


def rank(rows):
    return len(rref(rows, 0))


def vec(f, index):
    return {index[m]: c for m, c in f.items()}


def kernel_basis(vectors, dim):
    """vectors: list of dicts (rows of a matrix, one per basis element of the
    source).  Returns a basis of the kernel as list of coefficient dicts."""
    # Build augmented rows [row | e_i] and reduce; kernel = rows whose left part died.
    rows = []
    for i, v in enumerate(vectors):
        r = dict(v)
        r[('e', i)] = Fraction(1)
        rows.append(r)
    # order: numeric columns first, then tags
    def key(c):
        return (1, c[1]) if isinstance(c, tuple) else (0, c)
    piv = {}
    out = []
    for r in rows:
        r = {k: Fraction(v) for k, v in r.items() if v}
        while True:
            left = [c for c in r if not isinstance(c, tuple)]
            if not left:
                out.append({c[1]: v for c, v in r.items()})
                break
            c = min(left)
            if c in piv:
                f = r[c] / piv[c][c]
                for k, v in piv[c].items():
                    nv = r.get(k, 0) - f * v
                    if nv == 0:
                        r.pop(k, None)
                    else:
                        r[k] = nv
            else:
                piv[c] = r
                break
    return out


# ---------------------------------------------------------------- checks
def check_identities(n, tmax=4):
    """Jacobi, gradedness and Delta-derivation property on random-ish elements."""
    mons = [m for t in range(1, tmax + 1) for m in monomials(n, t)]
    tests = 0
    for i, m1 in enumerate(mons):
        for m2 in mons[: 12]:
            for m3 in mons[: 6]:
                f, g, h = {m1: 1}, {m2: 1}, {m3: 1}
                pf, pg = parity(m1), parity(m2)
                a, b, c = pf + 1, pg + 1, parity(m3) + 1   # the shifted parities
                # the BV master formula reproduces the Buttin bracket
                assert bracket(f, g, n) == bracket_bv(f, g, n), ("BV", m1, m2)
                # Delta is an odd derivation of the bracket:
                #   Delta{f,g} = {Delta f, g} + (-1)^{~p(f)} {f, Delta g}
                lhs = delta(bracket(f, g, n), n)
                df, dg = delta(f, n), delta(g, n)
                rhs = padd(bracket(df, g, n) if df else {},
                           scal(bracket(f, dg, n), (-1) ** a) if dg else {})
                assert lhs == rhs, ("Delta-derivation", m1, m2)
                # graded antisymmetry: {f,g} = -(-1)^{~p(f)~p(g)} {g,f}
                b1 = bracket(f, g, n)
                b2 = bracket(g, f, n)
                assert b1 == scal(b2, -(-1) ** (a * b)), ("antisym", m1, m2)
                # graded Jacobi for the shifted parities
                j = padd(
                    scal(bracket(f, bracket(g, h, n), n), 1),
                    scal(bracket(g, bracket(f, h, n), n), -(-1) ** (a * b)),
                    scal(bracket(bracket(f, g, n), h, n), -1),
                )
                assert not j, ("Jacobi", m1, m2, m3)
                tests += 1
    return tests


def delta_homology(n, tmax):
    """dim Ker Delta, Im Delta, H(Delta) in each total degree t <= tmax."""
    res = {}
    for t in range(0, tmax + 1):
        src = monomials(n, t)
        tgt = monomials(n, t - 2) if t >= 2 else []
        idx = {m: i for i, m in enumerate(tgt)}
        rows = [vec(delta({m: 1}, n), idx) for m in src]
        r = rank(rows)
        res[t] = dict(dim=len(src), im=r, ker=len(src) - r)
    # H^t = Ker_t / Im(from t+2)
    for t in range(0, tmax + 1):
        im_into_t = res[t + 2]['im'] if t + 2 in res else None
        res[t]['H'] = None if im_into_t is None else res[t]['ker'] - im_into_t
    return res


def derived(n, tmax):
    """dim of [sb,sb] in each total degree, against dim sb."""
    # basis of Ker Delta in each degree
    kers = {}
    for t in range(0, tmax + 1):
        src = monomials(n, t)
        tgt = monomials(n, t - 2) if t >= 2 else []
        idx = {m: i for i, m in enumerate(tgt)}
        rows = [vec(delta({m: 1}, n), idx) for m in src]
        kb = kernel_basis(rows, len(src))
        kers[t] = [{src[i]: c for i, c in v.items()} for v in kb]
    out = {}
    for t in range(0, tmax + 1):
        # [sb_s, sb_u] lands in degree s+u-2 = t
        prods = []
        for s in range(0, t + 3):
            u = t + 2 - s
            if u < 0 or s > tmax or u > tmax:
                continue
            for f in kers[s]:
                for g in kers[u]:
                    if not f or not g:
                        continue
                    b = bracket(f, g, n)
                    if b:
                        prods.append(b)
        tmons = monomials(n, t)
        idx = {m: i for i, m in enumerate(tmons)}
        out[t] = dict(sb=len(kers[t]), der=rank([vec(p, idx) for p in prods]))
    return out


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tmax = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    for n in range(2, nmax + 1):
        print(f"=== n = {n} " + "=" * 40)
        k = check_identities(n, tmax=3)
        print(f"BV identities (Jacobi, antisymmetry, Delta a derivation): {k} triples OK")
        h = delta_homology(n, tmax)
        print(" t : dim  Ker  Im   H(Delta)")
        for t in sorted(h):
            r = h[t]
            print(f"{t:2d} : {r['dim']:4d} {r['ker']:4d} {r['im']:4d}   "
                  f"{'?' if r['H'] is None else r['H']}")
        d = derived(n, tmax - 2)
        print(" t : dim sb_t  dim [sb,sb]_t  dim Im(Delta)_t   (g_k, k = t-2)")
        for t in sorted(d):
            imt = h[t + 2]['im'] if t + 2 in h else None
            print(f"{t:2d} : {d[t]['sb']:7d} {d[t]['der']:12d} {str(imt):>14s}"
                  f"        k={t-2}")
        # conclusions
        hnz = [t for t in h if h[t]['H']]
        print(f"  H(Delta) is nonzero exactly in degree(s) {hnz} "
              f"(dim {[h[t]['H'] for t in hnz]}); Xi has degree {n}")
        codim = [(t - 2, d[t]['sb'] - d[t]['der']) for t in sorted(d)
                 if d[t]['sb'] != d[t]['der']]
        print(f"  sb/[sb,sb]: nonzero exactly in g_k for (k, codim) = {codim}")
        same = all(d[t]['der'] == h[t + 2]['im'] for t in sorted(d) if t + 2 in h)
        print(f"  [sb,sb] = Im(Delta) in every computed degree: {same}")
        print(f"  dim g_0 = {d[2]['sb']} , dim spe({n}) = {2*n*n-1}")


if __name__ == '__main__':
    main()
