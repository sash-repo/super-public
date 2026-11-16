#!/usr/bin/env python3
"""Periplectic (odd symplectic) forms over divided power superalgebras, p > 2.

An analog of Skryabin's classification of the normal
shapes of symplectic forms, for ODD non-degenerate closed 2-forms.

Set-up
------
O(n|n; u) = O(n;u) tensor Lambda(xi_1..xi_n), the even indeterminates q_i
carrying divided powers of height p^{u_i}.

The divided power de Rham complex.  The differential d is ODD, hence

    p(dq_i) = 1   (dq_i anticommute, (dq_i)^2 = 0 for p > 2),
    p(dxi_i) = 0  (dxi_i commute and carry DIVIDED POWERS (dxi)^{(k)}),

so that

    Omega^* = O(n|n;u) (x) Lambda(dq_1..dq_n) (x) Gamma(dxi_1..dxi_n)

is the free supercommutative divided power superalgebra on the even generators
q_i, dxi_i and the odd generators xi_i, dq_i.  Everything below is one uniform
object: a supercommutative superalgebra with divided powers on its even
generators, of which O, Omega^1, Omega^2, ... are the homogeneous components
with respect to the FORM degree (the number of d's).

Conventions
-----------
* monomial = (ev, od): ev = exponents of the even generators (q's then dxi's),
  od = 0/1 exponents of the odd generators (xi's then dq's);
* multiplication: binomials on the even part (divided powers), Koszul sign on
  the odd part;
* d = the odd derivation q -> dq, xi -> dxi, dq -> 0, dxi -> 0;
* L_D = the derivation of parity p(D) with q_i -> D(q_i), xi_i -> D(xi_i),
  dq_i -> d(D(q_i)), dxi_i -> d(D(xi_i));
* iota_D = the derivation of parity p(D)+1 with q_i, xi_i -> 0,
  dq_i -> D(q_i), dxi_i -> D(xi_i).

The divided powers of the dxi are truncated at height p^2, which is an algebra
quotient (a divided power ideal), and which is invisible in the form degrees
<= 4 the classification uses.

Usage:  python3 peri.py         (plain python3 works too)
"""
import sys, os, itertools
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from exactlin import rank as _rank, nullspace as _nullspace   # noqa: E402


# --------------------------------------------------------------- the algebra

class Omega:
    """Omega^*(n|n; u) over F_p, p > 2."""

    def __init__(self, n, p, u=None, dxi_height=2):
        assert p > 2
        self.n, self.p = n, p
        self.u = tuple(u) if u is not None else (1,) * n
        # even generators: q_1..q_n (caps p^{u_i}), dxi_1..dxi_n (cap p^dxi_height)
        self.evcap = tuple(p ** k for k in self.u) + (p ** dxi_height,) * n
        self.nev = 2 * n
        # odd generators: xi_1..xi_n, dq_1..dq_n
        self.nod = 2 * n
        # form degree of each generator
        self.evdeg = (0,) * n + (1,) * n          # q_i, dxi_i
        self.oddeg = (0,) * n + (1,) * n          # xi_i, dq_i
        self.evpar = (0,) * (2 * n)
        self.odpar = (1,) * (2 * n)

    # ---- monomial helpers -------------------------------------------------
    def zero(self):
        return {}

    def one(self):
        return {((0,) * self.nev, (0,) * self.nod): 1}

    def gen_ev(self, i, k=1):
        ev = [0] * self.nev
        ev[i] = k
        return {(tuple(ev), (0,) * self.nod): 1}

    def gen_od(self, i):
        od = [0] * self.nod
        od[i] = 1
        return {((0,) * self.nev, tuple(od)): 1}

    def q(self, i, k=1):
        return self.gen_ev(i, k)

    def dxi(self, i, k=1):
        return self.gen_ev(self.n + i, k)

    def xi(self, i):
        return self.gen_od(i)

    def dq(self, i):
        return self.gen_od(self.n + i)

    def parity(self, mon):
        return sum(mon[1]) % 2

    def degree(self, mon):
        ev, od = mon
        return (sum(ev[self.n + i] for i in range(self.n))
                + sum(od[self.n + i] for i in range(self.n)))

    # ---- arithmetic -------------------------------------------------------
    def mul_mon(self, a, b):
        """product of two monomials -> (coeff, monomial) or None."""
        p = self.p
        aev, aod = a
        bev, bod = b
        ev = []
        c = 1
        for i in range(self.nev):
            s = aev[i] + bev[i]
            if s >= self.evcap[i]:
                return None
            c = (c * comb(s, aev[i])) % p
            if c == 0:
                return None
            ev.append(s)
        od = []
        sign = 0
        # move each odd letter of b left past the odd letters of a that follow it
        for j in range(self.nod):
            if bod[j]:
                if aod[j]:
                    return None
                sign += sum(aod[k] for k in range(j + 1, self.nod))
            od.append(aod[j] + bod[j])
        if sign % 2:
            c = (-c) % p
        return (c, (tuple(ev), tuple(od)))

    def mul(self, x, y):
        p, out = self.p, {}
        for a, ca in x.items():
            for b, cb in y.items():
                r = self.mul_mon(a, b)
                if r is None:
                    continue
                c, m = r
                v = (out.get(m, 0) + c * ca * cb) % p
                if v:
                    out[m] = v
                elif m in out:
                    del out[m]
        return out

    def add(self, *xs):
        p, out = self.p, {}
        for x in xs:
            for m, c in x.items():
                v = (out.get(m, 0) + c) % p
                if v:
                    out[m] = v
                elif m in out:
                    del out[m]
        return out

    def smul(self, c, x):
        p = self.p
        c %= p
        if c == 0:
            return {}
        return {m: (c * v) % p for m, v in x.items()}

    def sub(self, x, y):
        return self.add(x, self.smul(-1, y))

    def divpow(self, x, k):
        """x^{(k)} for an even element x, k < p (so x^{(k)} = x^k / k!)."""
        if k == 0:
            return self.one()
        assert k < self.p, "divided power of a general element needs k < p"
        y = x
        for _ in range(k - 1):
            y = self.mul(y, x)
        inv = pow(1, 1, self.p)
        f = 1
        for i in range(2, k + 1):
            f = (f * i) % self.p
        inv = pow(f, self.p - 2, self.p)
        return self.smul(inv, y)

    # ---- derivations ------------------------------------------------------
    def derivation(self, imgs_ev, imgs_od, par):
        """Return a function applying the derivation of parity par given by the
        images of the generators (imgs_ev[i], imgs_od[i] as elements)."""
        def apply_mon(mon):
            ev, od = mon
            out = {}
            pref_par = 0
            # even letters first (all of parity 0): prefix parity stays 0
            for i in range(self.nev):
                a = ev[i]
                if a and imgs_ev[i]:
                    # monomial = E * O with E even; D(E)*O, and inside E the
                    # image D(g_i) must stay to the LEFT of the odd letters O
                    pev = list(ev)
                    pev[i] = a - 1
                    ered = {(tuple(pev), (0,) * self.nod): 1}
                    odpart = {((0,) * self.nev, od): 1}
                    term = self.mul(self.mul(ered, imgs_ev[i]), odpart)
                    out = self.add(out, term)
            # odd letters, in order, accumulating the prefix parity
            for j in range(self.nod):
                if od[j]:
                    if imgs_od[j]:
                        pod = list(od)
                        pod[j] = 0
                        # prefix = even part * odd letters before j
                        pre_od = [0] * self.nod
                        for k in range(j):
                            pre_od[k] = od[k]
                        suf_od = [0] * self.nod
                        for k in range(j + 1, self.nod):
                            suf_od[k] = od[k]
                        pre = {(ev, tuple(pre_od)): 1}
                        suf = {((0,) * self.nev, tuple(suf_od)): 1}
                        term = self.mul(self.mul(pre, imgs_od[j]), suf)
                        if par * pref_par % 2:
                            term = self.smul(-1, term)
                        out = self.add(out, term)
                    pref_par = (pref_par + 1) % 2
            return out

        cache = {}

        def apply(x):
            out = {}
            for m, c in x.items():
                if m not in cache:
                    cache[m] = apply_mon(m)
                out = self.add(out, self.smul(c, cache[m]))
            return out
        return apply

    def d(self):
        if not hasattr(self, "_d"):
            imgs_ev = [self.dq(i) for i in range(self.n)] + [{}] * self.n
            imgs_od = [self.dxi(i) for i in range(self.n)] + [{}] * self.n
            self._d = self.derivation(imgs_ev, imgs_od, 1)
        return self._d

    def lie(self, D):
        """D = (fs, gs, par): D = sum f_i d/dq_i + sum g_i d/dxi_i."""
        fs, gs, par = D
        dd = self.d()
        # L_D := [iota_D, d]; since p(iota_D) = p(D)+1 this makes L_D super-
        # commute with d, i.e. L_D(de) = (-1)^{p(D)} d(D(e)).
        s = (-1) ** par
        imgs_ev = list(fs) + [self.smul(s, dd(g)) for g in gs]
        imgs_od = list(gs) + [self.smul(s, dd(f)) for f in fs]
        return self.derivation(imgs_ev, imgs_od, par)

    def iota(self, D):
        fs, gs, par = D
        imgs_ev = [{}] * self.n + list(gs)
        imgs_od = [{}] * self.n + list(fs)
        return self.derivation(imgs_ev, imgs_od, (par + 1) % 2)

    # ---- bases ------------------------------------------------------------
    def basis(self, deg, par=None):
        """monomials of the given form degree (and parity, if given)."""
        if (deg, par) in getattr(self, "_bcache", {}):
            return self._bcache[(deg, par)]
        n = self.n
        out = []
        qranges = [range(self.evcap[i]) for i in range(n)]
        dxcap = self.evcap[n]
        # form part: a subset S of the dq's and exponents e on the dxi's with
        # |S| + |e| = deg
        forms = []
        for S in itertools.chain.from_iterable(
                itertools.combinations(range(n), k) for k in range(min(n, deg) + 1)):
            rest = deg - len(S)
            for e in itertools.product(*[range(min(dxcap, rest + 1))] * n):
                if sum(e) == rest:
                    forms.append((S, e))
        for qe in itertools.product(*qranges):
            for xb in itertools.product(*[range(2)] * n):
                for (S, e) in forms:
                    ev = tuple(qe) + tuple(e)
                    od = tuple(xb) + tuple(1 if i in S else 0 for i in range(n))
                    m = (ev, od)
                    if par is not None and self.parity(m) != par:
                        continue
                    out.append(m)
        if not hasattr(self, "_bcache"):
            self._bcache = {}
        self._bcache[(deg, par)] = out
        return out

    def fun_basis(self, par=None):
        return self.basis(0, par)


# --------------------------------------------------------------- linear algebra

def vec(alg, x, idx):
    v = [0] * len(idx)
    for m, c in x.items():
        v[idx[m]] = c % alg.p
    return v


def matrix_of(alg, f, src, tgt_idx):
    """matrix (rows = images) of a linear map on the given source monomials."""
    rows = []
    for m in src:
        rows.append(vec(alg, f({m: 1}), tgt_idx))
    return rows


def rank_p(rows, p):
    if not rows:
        return 0
    return _rank(rows, p)


def nullspace_p(rows, p):
    return _nullspace(rows, p)


# --------------------------------------------------------------- de Rham

def cohomology(alg, deg, par=None):
    """dim H^deg (of the given parity) of the divided power de Rham complex."""
    d = alg.d()
    B0 = alg.basis(deg - 1) if deg > 0 else []
    B1 = alg.basis(deg)
    B2 = alg.basis(deg + 1)
    if par is not None:
        B0 = [m for m in B0 if alg.parity(m) == (par + 1) % 2]   # d is odd
        B1 = [m for m in B1 if alg.parity(m) == par]
        B2 = [m for m in B2 if alg.parity(m) == (par + 1) % 2]
    i1 = {m: i for i, m in enumerate(B1)}
    i2 = {m: i for i, m in enumerate(B2)}
    M1 = matrix_of(alg, d, B1, i2)              # d : deg -> deg+1
    M0 = matrix_of(alg, d, B0, i1) if B0 else []
    r1 = rank_p(M1, alg.p)
    r0 = rank_p(M0, alg.p) if M0 else 0
    return len(B1) - r1 - r0, len(B1), r1, r0


def is_closed(alg, om):
    return not alg.d()(om)


def is_exact(alg, om, deg=2):
    """is om a coboundary?"""
    par = None
    ms = set(alg.parity(m) for m in om)
    if len(ms) == 1:
        par = ms.pop()
    d = alg.d()
    B0 = alg.basis(deg - 1)
    if par is not None:
        B0 = [m for m in B0 if alg.parity(m) == (par + 1) % 2]
    B1 = alg.basis(deg)
    if par is not None:
        B1 = [m for m in B1 if alg.parity(m) == par]
    i1 = {m: i for i, m in enumerate(B1)}
    M = matrix_of(alg, d, B0, i1)
    r = rank_p(M, alg.p)
    r2 = rank_p(M + [vec(alg, om, i1)], alg.p)
    return r == r2


# --------------------------------------------------------------- vector fields

def vfield_basis(alg, par=None):
    """basis of vect(n|n;u) as (fs, gs, parity) triples."""
    out = []
    for i in range(alg.n):
        for m in alg.fun_basis():
            f = {m: 1}
            fs = [{}] * alg.n
            fs[i] = f
            pr = alg.parity(m)                    # p(f d/dq) = p(f)
            if par is None or pr == par:
                out.append((tuple(fs), tuple([{}] * alg.n), pr))
    for i in range(alg.n):
        for m in alg.fun_basis():
            g = {m: 1}
            gs = [{}] * alg.n
            gs[i] = g
            pr = (alg.parity(m) + 1) % 2          # p(g d/dxi) = p(g) + 1
            if par is None or pr == par:
                out.append((tuple([{}] * alg.n), tuple(gs), pr))
    return out


def vf_to_vec(alg, D, idx):
    """coordinates of a vector field in the flat basis O^{2n}."""
    fs, gs, _ = D
    v = [0] * (2 * alg.n * len(idx))
    L = len(idx)
    for i, f in enumerate(fs):
        for m, c in f.items():
            v[i * L + idx[m]] = c % alg.p
    for i, g in enumerate(gs):
        for m, c in g.items():
            v[(alg.n + i) * L + idx[m]] = c % alg.p
    return v


def vec_to_vf(alg, v, mons):
    L = len(mons)
    fs, gs = [], []
    for i in range(alg.n):
        f = {mons[j]: v[i * L + j] % alg.p for j in range(L) if v[i * L + j] % alg.p}
        fs.append(f)
    for i in range(alg.n):
        j0 = (alg.n + i) * L
        g = {mons[j]: v[j0 + j] % alg.p for j in range(L) if v[j0 + j] % alg.p}
        gs.append(g)
    # parity, if homogeneous
    ps = set()
    for f in fs:
        ps |= {alg.parity(m) for m in f}
    for g in gs:
        ps |= {(alg.parity(m) + 1) % 2 for m in g}
    par = ps.pop() if len(ps) == 1 else 0
    return (tuple(fs), tuple(gs), par)


def nondegenerate(alg, om):
    """is D -> iota_D om a bijection vect -> Omega^1 ?"""
    mons = alg.fun_basis()
    B1 = alg.basis(1)
    i1 = {m: i for i, m in enumerate(B1)}
    rows = []
    for D in vfield_basis(alg):
        rows.append(vec(alg, alg.iota(D)(om), i1))
    return rank_p(rows, alg.p) == len(rows) == len(B1)


def hamiltonian(alg, om, f):
    """the field H_f with iota_{H_f} om = df, if it exists."""
    d = alg.d()
    B1 = alg.basis(1)
    i1 = {m: i for i, m in enumerate(B1)}
    basis = vfield_basis(alg)
    rows = [vec(alg, alg.iota(D)(om), i1) for D in basis]
    # solve x * M = df
    from exactlin import solve
    tgt = vec(alg, d(f), i1)
    # transpose: columns are the basis images
    M = [[rows[j][i] for j in range(len(rows))] for i in range(len(B1))]
    x = solve(M, tgt, alg.p)
    if x is None:
        return None
    fs = [{}] * alg.n
    gs = [{}] * alg.n
    out_fs = [dict() for _ in range(alg.n)]
    out_gs = [dict() for _ in range(alg.n)]
    for c, D in zip(x, basis):
        c %= alg.p
        if not c:
            continue
        for i in range(alg.n):
            for m, v in D[0][i].items():
                out_fs[i][m] = (out_fs[i].get(m, 0) + c * v) % alg.p
            for m, v in D[1][i].items():
                out_gs[i][m] = (out_gs[i].get(m, 0) + c * v) % alg.p
    fs = tuple({m: v for m, v in x_.items() if v} for x_ in out_fs)
    gs = tuple({m: v for m, v in x_.items() if v} for x_ in out_gs)
    ps = set()
    for x_ in fs:
        ps |= {alg.parity(m) for m in x_}
    for x_ in gs:
        ps |= {(alg.parity(m) + 1) % 2 for m in x_}
    par = ps.pop() if len(ps) == 1 else 0
    return (fs, gs, par)


def preserving(alg, om):
    """basis of h(om) = {D | L_D om = 0}, as vectors in the flat O^{2n}."""
    mons = alg.fun_basis()
    idx = {m: i for i, m in enumerate(mons)}
    B2 = alg.basis(2)
    i2 = {m: i for i, m in enumerate(B2)}
    basis = vfield_basis(alg)
    rows = [vec(alg, alg.lie(D)(om), i2) for D in basis]
    cols = [[rows[j][i] for j in range(len(rows))] for i in range(len(B2))]
    ns = nullspace_p(cols, alg.p)     # left nullspace: sum c_i rows_i = 0
    return ns, basis, rows


def vf_bracket(alg, D, E):
    """[D,E] as a vector field (super-commutator of derivations on O)."""
    LD, LE = alg.lie(D), alg.lie(E)
    par = (D[2] + E[2]) % 2
    sgn = (-1) ** (D[2] * E[2])
    fs, gs = [], []
    for i in range(alg.n):
        qi = alg.q(i)
        fs.append(alg.sub(LD(LE(qi)), alg.smul(sgn, LE(LD(qi)))))
    for i in range(alg.n):
        xii = alg.xi(i)
        gs.append(alg.sub(LD(LE(xii)), alg.smul(sgn, LE(LD(xii)))))
    return (tuple(fs), tuple(gs), par)


# --------------------------------------------------------------- forms

def omega0(alg):
    """the standard periplectic form sum dq_i dxi_i."""
    out = {}
    for i in range(alg.n):
        out = alg.add(out, alg.mul(alg.dq(i), alg.dxi(i)))
    return out


def alpha0(alg):
    """a primitive of omega0: sum q_i dxi_i  (d of it is omega0 up to sign)."""
    out = {}
    for i in range(alg.n):
        out = alg.add(out, alg.mul(alg.q(i), alg.dxi(i)))
    return out


def omegaXi(alg, coeffs=None):
    """omega0 + sum a_i xi_i (dxi_i)^{(2)}  --- the exceptional p = 3 family."""
    if coeffs is None:
        coeffs = [1] + [0] * (alg.n - 1)
    out = omega0(alg)
    for i, a in enumerate(coeffs):
        if a % alg.p:
            out = alg.add(out, alg.smul(a, alg.mul(alg.xi(i), alg.dxi(i, 2))))
    return out


def exp_v(alg, j, eps=1):
    """exp(eps q_j^{(p^{u_j})}) = sum_{0<=a<p} eps^a q_j^{(a p^{u_j})}, an element
    of the algebra with the height of q_j raised by one (Skryabin's completion,
    truncated where it must be: (q^{(p^N)})^{(a)} = q^{(a p^N)} for a < p)."""
    p = alg.p
    N = p ** (alg.u[j] - 1)          # here alg is ALREADY the enlarged algebra
    out = alg.zero()
    for a in range(p):
        out = alg.add(out, alg.smul(pow(eps, a, p), alg.q(j, a * N)))
    return out


def omega2exp(alg, j=0, eps=1):
    """the odd analog of Skryabin's 2nd type,  omega_{2,j} = d(exp(v) alpha_0),
    v = q_j^{(p^{u_j})}.  ALG MUST BE THE ENLARGED ALGEBRA, i.e. carry the height
    u_j + 1 in the j-th even indeterminate; the form is then closed by
    construction and its restriction to O(n|n;u) is what is classified."""
    E = exp_v(alg, j, eps)
    return alg.d()(alg.mul(E, alpha0(alg)))
