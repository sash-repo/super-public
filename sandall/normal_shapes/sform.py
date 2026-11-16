#!/usr/bin/env python3
"""The divided power de Rham complex over O(m|n;u), any p, INCLUDING p = 2.

The superization of the union of Skryabin's and the non-alternating cases.
The two halves are:

    nalt2.py : non-alternating (p = 2), no odd indeterminates;
    peri.py  : ODD 2-forms over O(n|n;u), p > 2.

This module is the union of the two ambients, and it is the union in a strong
sense: at p = 2 the four kinds of generator obey ONE rule.

The generators, in this order
----------------------------
    x_1..x_m    even, divided powers, cap p^{u_i}          form degree 0
    xi_1..xi_n  odd,  square zero                          form degree 0
    dx_1..dx_m  odd  (d is odd)                            form degree 1
    dxi_1..dxi_n even, divided powers, cap p^dh            form degree 1

For p > 2 an odd generator is exterior: its square is 0 and it has no divided
powers, because 2 y^{(2)} = y^2 = 0 and 2 is invertible.

For p = 2 that argument dies: 2 y^{(2)} = 0 holds identically and y^{(2)} need
not vanish.  The divided power de Rham complex of [BGLL] keeps it, and then

    * xi_j has xi_j^2 = 0 and no divided powers -- this is the DEFINITION of an
      odd indeterminate, and it is exactly a divided power generator of cap 2;
    * dx_i has (dx_i)^2 = 2 (dx_i)^{(2)} = 0 and DOES have (dx_i)^{(2)};

so at p = 2 every generator is a divided power generator, the parities only
label the monomials, and the Koszul signs are all +1.  That is the collapse
Theorem the super case is built on; `collapse.py` verifies it.

Conventions (inherited from peri.py, where they cost time)
---------------------------------------------------------
* a monomial is a tuple of exponents, one per generator, in the order above;
  for a divided power generator the exponent e means g^{(e)}, so that
  g^{(a)} g^{(b)} = binom(a+b, a) g^{(a+b)};
* signs come only from the exterior generators, all of parity 1;
* a derivation obeys D(g^{(e)}) = g^{(e-1)} D(g) on divided powers, and its
  image must stay to the LEFT of the odd letters that follow;
* L_D = [iota_D, d], so L_D(de) = (-1)^{p(D)} d(D(e));
* nullspace(rows) is the RIGHT nullspace: {D | L_D omega = 0} needs a transpose.

Usage:  python3 sform.py      (plain python3 works too, slower)
"""
import sys, os, itertools
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from exactlin import rank as _rank, nullspace as _nullspace   # noqa: E402

DIV, EXT = 0, 1          # kind of a generator


class Om:
    """Omega^*(m|n; u) over F_p, p >= 2."""

    def __init__(self, m, n, p, u=None, dh=2, dxh=2):
        self.m, self.n, self.p = m, n, p
        self.u = tuple(u) if u is not None else (1,) * m
        assert len(self.u) == m
        N = 2 * (m + n)
        self.N = N
        # slices
        self.ix, self.ixi = 0, m
        self.idx, self.idxi = m + n, m + n + m
        kind, cap, par, deg = [], [], [], []
        for i in range(m):                       # x_i
            kind.append(DIV); cap.append(p ** self.u[i]); par.append(0); deg.append(0)
        for j in range(n):                       # xi_j
            if p == 2:
                kind.append(DIV); cap.append(2)
            else:
                kind.append(EXT); cap.append(2)
            par.append(1); deg.append(0)
        for i in range(m):                       # dx_i
            if p == 2:
                kind.append(DIV); cap.append(p ** dxh)
            else:
                kind.append(EXT); cap.append(2)
            par.append(1); deg.append(1)
        for j in range(n):                       # dxi_j
            kind.append(DIV); cap.append(p ** dh); par.append(0); deg.append(1)
        self.kind, self.cap = tuple(kind), tuple(cap)
        self.par, self.deg = tuple(par), tuple(deg)
        self._bcache = {}

    # ------------------------------------------------------------ monomials
    def zero(self):
        return {}

    def one(self):
        return {(0,) * self.N: 1}

    def gen(self, i, k=1):
        e = [0] * self.N
        e[i] = k
        return {tuple(e): 1}

    def x(self, i, k=1):
        return self.gen(self.ix + i, k)

    def xi(self, j):
        return self.gen(self.ixi + j)

    def dx(self, i, k=1):
        return self.gen(self.idx + i, k)

    def dxi(self, j, k=1):
        return self.gen(self.idxi + j, k)

    def parity(self, mon):
        return sum(e * self.par[i] for i, e in enumerate(mon)) % 2

    def degree(self, mon):
        return sum(e * self.deg[i] for i, e in enumerate(mon))

    # ----------------------------------------------------------- arithmetic
    def mul_mon(self, a, b):
        p = self.p
        e = []
        c = 1
        for i in range(self.N):
            s = a[i] + b[i]
            if self.kind[i] == EXT:
                if s > 1:
                    return None
            else:
                if s >= self.cap[i]:
                    return None
                c = (c * comb(s, a[i])) % p
                if c == 0:
                    return None
            e.append(s)
        # Koszul sign: move each odd letter of b left past the odd letters of a
        sign = 0
        for j in range(self.N):
            if self.kind[j] == EXT and b[j]:
                sign += sum(a[k] for k in range(j + 1, self.N)
                            if self.kind[k] == EXT)
        if sign % 2:
            c = (-c) % p
        return (c, tuple(e))

    def mul(self, X, Y):
        p, out = self.p, {}
        for a, ca in X.items():
            for b, cb in Y.items():
                r = self.mul_mon(a, b)
                if r is None:
                    continue
                c, mm = r
                v = (out.get(mm, 0) + c * ca * cb) % p
                if v:
                    out[mm] = v
                elif mm in out:
                    del out[mm]
        return out

    def add(self, *Xs):
        p, out = self.p, {}
        for X in Xs:
            for mm, c in X.items():
                v = (out.get(mm, 0) + c) % p
                if v:
                    out[mm] = v
                elif mm in out:
                    del out[mm]
        return out

    def smul(self, c, X):
        p = self.p
        c %= p
        if c == 0:
            return {}
        return {mm: (c * v) % p for mm, v in X.items()}

    def sub(self, X, Y):
        return self.add(X, self.smul(-1, Y))

    # ---------------------------------------------------------- derivations
    def derivation(self, imgs, par):
        """The derivation of parity `par` sending generator i to imgs[i]."""
        def apply_mon(mon):
            out = {}
            for i in range(self.N):
                a = mon[i]
                if not a or not imgs[i]:
                    continue
                pre = [0] * self.N
                suf = [0] * self.N
                for k in range(i):
                    pre[k] = mon[k]
                for k in range(i + 1, self.N):
                    suf[k] = mon[k]
                red = list(pre)
                red[i] = a - 1 if self.kind[i] == DIV else 0
                P = {tuple(red): 1}
                S = {tuple(suf): 1}
                term = self.mul(self.mul(P, imgs[i]), S)
                # sign: D passes the odd letters of the prefix (including the
                # a-1 remaining copies of g_i itself when g_i is odd)
                ppar = sum(red[k] * self.par[k] for k in range(self.N)) % 2
                if par * ppar % 2:
                    term = self.smul(-1, term)
                out = self.add(out, term)
            return out

        cache = {}

        def apply(X):
            out = {}
            for mm, c in X.items():
                if mm not in cache:
                    cache[mm] = apply_mon(mm)
                out = self.add(out, self.smul(c, cache[mm]))
            return out
        return apply

    def d(self):
        if not hasattr(self, "_d"):
            imgs = ([self.dx(i) for i in range(self.m)]
                    + [self.dxi(j) for j in range(self.n)]
                    + [{}] * (self.m + self.n))
            self._d = self.derivation(imgs, 1)
        return self._d

    # -- vector fields: D = (fs, gs, par), D = sum f_i d/dx_i + sum g_j d/dxi_j
    def lie(self, D):
        fs, gs, par = D
        dd = self.d()
        s = (-1) ** par
        imgs = (list(fs) + list(gs)
                + [self.smul(s, dd(f)) for f in fs]
                + [self.smul(s, dd(g)) for g in gs])
        return self.derivation(imgs, par)

    def iota(self, D):
        fs, gs, par = D
        imgs = ([{}] * (self.m + self.n)) + list(fs) + list(gs)
        return self.derivation(imgs, (par + 1) % 2)

    # ---------------------------------------------------------------- bases
    def basis(self, deg, par=None):
        key = (deg, par)
        if key in self._bcache:
            return self._bcache[key]
        ranges = []
        for i in range(self.N):
            hi = 2 if self.kind[i] == EXT else self.cap[i]
            if self.deg[i] == 1:
                hi = min(hi, deg + 1)
            ranges.append(range(hi))
        out = []
        for e in itertools.product(*ranges):
            if self.degree(e) != deg:
                continue
            if par is not None and self.parity(e) != par:
                continue
            out.append(e)
        self._bcache[key] = out
        return out

    def fun_basis(self, par=None):
        return self.basis(0, par)

    def sdim_funs(self):
        return (len(self.fun_basis(0)), len(self.fun_basis(1)))


# ------------------------------------------------------------ linear algebra
def vec(alg, X, idx):
    v = [0] * len(idx)
    for mm, c in X.items():
        v[idx[mm]] = c % alg.p
    return v


def matrix_of(alg, f, src, tgt_idx):
    return [vec(alg, f({mm: 1}), tgt_idx) for mm in src]


def rank_p(rows, p):
    return _rank(rows, p) if rows else 0


def nullspace_p(rows, p):
    return _nullspace(rows, p)


# ------------------------------------------------------------------ de Rham
def cohomology(alg, deg, par=None):
    """(dim H, dim Omega^deg, rank d_deg, rank d_{deg-1}) in the given parity."""
    d = alg.d()
    B0 = alg.basis(deg - 1) if deg > 0 else []
    B1 = alg.basis(deg)
    B2 = alg.basis(deg + 1)
    if par is not None:
        B0 = [mm for mm in B0 if alg.parity(mm) == (par + 1) % 2]
        B1 = [mm for mm in B1 if alg.parity(mm) == par]
        B2 = [mm for mm in B2 if alg.parity(mm) == (par + 1) % 2]
    i1 = {mm: i for i, mm in enumerate(B1)}
    i2 = {mm: i for i, mm in enumerate(B2)}
    r1 = rank_p(matrix_of(alg, d, B1, i2), alg.p)
    r0 = rank_p(matrix_of(alg, d, B0, i1), alg.p) if B0 else 0
    return len(B1) - r1 - r0, len(B1), r1, r0


def is_closed(alg, om):
    return not alg.d()(om)


def coboundaries(alg, deg, par):
    """Matrix whose rows span d(Omega^{deg-1}) inside Omega^deg (parity par)."""
    d = alg.d()
    B0 = [mm for mm in alg.basis(deg - 1) if alg.parity(mm) == (par + 1) % 2]
    B1 = [mm for mm in alg.basis(deg) if alg.parity(mm) == par]
    i1 = {mm: i for i, mm in enumerate(B1)}
    return matrix_of(alg, d, B0, i1), B1, i1


def is_exact(alg, om, deg=2):
    pars = set(alg.parity(mm) for mm in om)
    assert len(pars) == 1, "give a homogeneous form"
    par = pars.pop()
    M, B1, i1 = coboundaries(alg, deg, par)
    r = rank_p(M, alg.p)
    return r == rank_p(M + [vec(alg, om, i1)], alg.p)


# ------------------------------------------------------------ vector fields
def vfield_basis(alg, par=None):
    out = []
    Z = tuple([{}] * alg.m), tuple([{}] * alg.n)
    for i in range(alg.m):
        for mm in alg.fun_basis():
            fs = [{}] * alg.m
            fs[i] = {mm: 1}
            pr = alg.parity(mm)                       # p(f d/dx) = p(f)
            if par is None or pr == par:
                out.append((tuple(fs), Z[1], pr))
    for j in range(alg.n):
        for mm in alg.fun_basis():
            gs = [{}] * alg.n
            gs[j] = {mm: 1}
            pr = (alg.parity(mm) + 1) % 2             # p(g d/dxi) = p(g)+1
            if par is None or pr == par:
                out.append((Z[0], tuple(gs), pr))
    return out


def vf_vec(alg, D, idx):
    fs, gs, _ = D
    L = len(idx)
    v = [0] * ((alg.m + alg.n) * L)
    for i, f in enumerate(fs):
        for mm, c in f.items():
            v[i * L + idx[mm]] = c % alg.p
    for j, g in enumerate(gs):
        for mm, c in g.items():
            v[(alg.m + j) * L + idx[mm]] = c % alg.p
    return v


def vec_vf(alg, v, mons, par):
    L = len(mons)
    fs = []
    for i in range(alg.m):
        fs.append({mons[k]: v[i * L + k] % alg.p
                   for k in range(L) if v[i * L + k] % alg.p})
    gs = []
    for j in range(alg.n):
        j0 = (alg.m + j) * L
        gs.append({mons[k]: v[j0 + k] % alg.p
                   for k in range(L) if v[j0 + k] % alg.p})
    return (tuple(fs), tuple(gs), par)


def preserving(alg, om, par=None):
    """{D in vect(m|n;u) | L_D omega = 0}, as (dim_ev, dim_od, basis)."""
    dims, bas = [], []
    for pr in ((0, 1) if par is None else (par,)):
        B = vfield_basis(alg, pr)
        deg2 = alg.basis(2)
        i2 = {mm: i for i, mm in enumerate(deg2)}
        rows = []
        for D in B:
            rows.append(vec(alg, alg.lie(D)(om), i2))
        # {c | sum c_k L_{D_k} omega = 0} = left nullspace of rows = nullspace of
        # the transpose
        T = [[rows[k][i] for k in range(len(rows))] for i in range(len(deg2))]
        ns = nullspace_p(T, alg.p) if T else []
        mons = alg.fun_basis()
        idx = {mm: i for i, mm in enumerate(mons)}
        fields = []
        for v in ns:
            acc = [0] * ((alg.m + alg.n) * len(mons))
            for k, c in enumerate(v):
                c = int(c) % alg.p
                if not c:
                    continue
                w = vf_vec(alg, B[k], idx)
                for t in range(len(acc)):
                    acc[t] = (acc[t] + c * w[t]) % alg.p
            fields.append(vec_vf(alg, acc, mons, pr))
        dims.append(len(ns))
        bas.append(fields)
    if par is None:
        return dims[0], dims[1], bas[0] + bas[1]
    return dims[0], bas[0]


# --------------------------------------------------------- non-degeneracy
def gram(alg, om):
    """Gram matrix of omega in the flat basis d/dx_1..d/dxi_n, entries in O."""
    N = alg.m + alg.n
    G = [[None] * N for _ in range(N)]
    flats = []
    for i in range(alg.m):
        fs = [{}] * alg.m
        fs[i] = alg.one()
        flats.append((tuple(fs), tuple([{}] * alg.n), 0))
    for j in range(alg.n):
        gs = [{}] * alg.n
        gs[j] = alg.one()
        flats.append((tuple([{}] * alg.m), tuple(gs), 1))
    for a in range(N):
        ia = alg.iota(flats[a])
        for b in range(N):
            ib = alg.iota(flats[b])
            G[a][b] = ib(ia(om))
    return G


def const_term(alg, f):
    return f.get((0,) * alg.N, 0) % alg.p


def nondegenerate(alg, om):
    """omega non-degenerate <=> the constant part of its Gram matrix is
    invertible over K (the rest of O is nilpotent)."""
    G = gram(alg, om)
    N = alg.m + alg.n
    C = [[const_term(alg, G[a][b]) for b in range(N)] for a in range(N)]
    return rank_p(C, alg.p) == N, C


if __name__ == "__main__":
    for (m, n, p) in [(2, 1, 3), (1, 1, 2), (2, 2, 2)]:
        A = Om(m, n, p)
        print(f"O({m}|{n}), p={p}: sdim O = {A.sdim_funs()}, "
              f"dim Omega^1 = {len(A.basis(1))}, dim Omega^2 = {len(A.basis(2))}")
