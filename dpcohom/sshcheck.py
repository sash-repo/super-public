#!/usr/bin/env python3
"""sshcheck.py -- Schwarz--Shapiro's Grassmann divided powers against the
divided power cohomology of [BGLL].

Everything is exact: the complexes are built over Q (Fractions), their
integrality is checked, and the ranks are taken mod p with ../lib/exactlin.py.

A)  THE MECHANISM.  Even coordinates y_1..y_k evaluated at Grassmann points
    y_j = sum of N pairs xi_a xi_b (fresh generators).  Over F_p the ordinary
    monomials y^K of degree n give a map S^n -> Lambda whose rank is the number
    of K with all k_j < p (the p-th powers die), whereas the divided power
    monomials y^K/K! are integral and their map Gamma^n -> Lambda is injective
    mod p as soon as N >= n.  Hence the lattice of the Q-polynomials that are
    integral on all Grassmann points is exactly the divided power lattice
    ([SSh1], Lemma 4.7 / Theorem 3.8, recomputed).

B)  THE COCHAINS.  For a Lie superalgebra g over Z (squaring s(f) = [f,f]/2
    integral) the Chevalley--Eilenberg complex is the algebra of functions on
    Pi g with the homological vector field Q.  Two lattices in it:
       poly : the polynomial functions  (= the dual of the divided power chains
              Lambda(g_ev) (x) Gamma(g_od) of [BGLL], i.e., the complex computing
              Ext_{U(g)}(K;K) by the main theorem of the text),
       grass: the functions on the Grassmann neighborhood of 0 in Pi g, i.e.,
              the divided power lattice in the EVEN coordinates of Pi g (those
              dual to the odd elements of g).
    Checked: Q^2 = 0, Q integral on both lattices, and the cohomology mod p of
    each lattice, compared with podd.py's anti-symmetric AH and divided power
    dpH (p > 2) and with dpco.py's Ext (p = 2).

C)  THE CHAINS.  The Z-dual of the poly cochain lattice is the divided power
    chain lattice = functions on the Grassmann neighborhood of 0 in Pi(g^*):
    its homology mod p (transposed matrices) is dual to the poly cohomology.

Usage:  python sshcheck.py      (writes sshcheck.log)
"""

import sys
from fractions import Fraction
from itertools import combinations
from math import comb, factorial, prod

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, backend                                 # noqa: E402

LOG = []


def out(s=""):
    print(s)
    LOG.append(s)
    sys.stdout.flush()


# ===================================================================== A
def grass_mul(a, b):
    """product of two elements of a Grassmann algebra over Z, each a dict
    {sorted tuple of generator indices: int}"""
    res = {}
    for ka, ca in a.items():
        sa = set(ka)
        for kb, cb in b.items():
            if sa.intersection(kb):
                continue
            # sign of merging ka (left) and kb (right) into increasing order
            inv = sum(1 for x in ka for y in kb if x > y)
            key = tuple(sorted(ka + kb))
            v = (-1) ** inv * ca * cb
            res[key] = res.get(key, 0) + v
    return {k: v for k, v in res.items() if v}


def grass_points(k, N):
    """y_j = sum_{t<N} xi_{2(jN+t)} xi_{2(jN+t)+1}, fresh pairs"""
    pts = []
    for j in range(k):
        y = {}
        for t in range(N):
            a = 2 * (j * N + t)
            y[(a, a + 1)] = 1
        pts.append(y)
    return pts


def compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for a in range(n + 1):
        for rest in compositions(n - a, k - 1):
            yield (a,) + rest


def eval_matrix(k, N, n, divided):
    pts = grass_points(k, N)
    powers = []
    for j in range(k):
        pw = [{(): 1}]
        for _ in range(n):
            pw.append(grass_mul(pw[-1], pts[j]))
        powers.append(pw)
    rows = []
    cols = {}
    for K in compositions(n, k):
        v = {(): 1}
        for j in range(k):
            v = grass_mul(v, powers[j][K[j]])
        if divided:
            c = prod(factorial(a) for a in K)
            for key in v:
                assert v[key] % c == 0, "divided power not integral"
            v = {key: val // c for key, val in v.items()}
        rows.append(v)
        for key in v:
            cols.setdefault(key, len(cols))
    mat = []
    for v in rows:
        r = [0] * max(1, len(cols))
        for key, val in v.items():
            r[cols[key]] = val
        mat.append(r)
    return mat


def part_A():
    out("=" * 78)
    out("A)  the mechanism: Grassmann points see Gamma, not S")
    out("=" * 78)
    for p in (2, 3, 5):
        for k in (1, 2):
            for n in range(0, 2 * p + 2):
                if k == 2 and n > p + 2:
                    break
                N = n
                ms = eval_matrix(k, N, n, False)
                md = eval_matrix(k, N, n, True)
                rs = rank(ms, p)
                rd = rank(md, p)
                dimS = comb(n + k - 1, k - 1)
                trunc = sum(1 for K in compositions(n, k) if max(K) < p)
                out("   p=%d k=%d n=%2d N=%2d  dim=%3d  rank_p S^n-eval=%3d (expected"
                    " %3d, exps<p)  rank_p Gamma^n-eval=%3d  %s"
                    % (p, k, n, N, dimS, rs, trunc, rd,
                       "OK" if (rs == trunc and rd == dimS) else "MISMATCH"))
    out("   too few odd generators: k=1, p=5, N pairs, Gamma^n-eval rank")
    for N in (1, 2, 3):
        rr = [rank(eval_matrix(1, N, n, True), 5) for n in range(6)]
        out("      N=%d  ranks in degrees 0..5: %s  (y^(a) = 0 on these points"
            " for a > N)" % (N, rr))
    out()


# ===================================================================== B
class SLAQ:
    """Lie superalgebra over Z: basis 0..m-1 even, m..m+n-1 odd;
    br[(i,j)] = {k: c} for i <= j; the rest by super anti-symmetry."""

    def __init__(self, m, n, brackets, name):
        self.m, self.n, self.dim, self.name = m, n, m + n, name
        self.c = {}
        for (i, j), v in brackets.items():
            self.c[(i, j)] = dict(v)
            if i != j:
                s = -((-1) ** (self.par(i) * self.par(j)))
                self.c[(j, i)] = {k: s * x for k, x in v.items()}

    def par(self, i):
        return 1 if i >= self.m else 0

    def br(self, i, j):
        return self.c.get((i, j), {})

    def jacobi_defect(self):
        bad = 0
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    # [i,[j,k]] - [[i,j],k] - (-1)^{p_i p_j} [j,[i,k]]
                    acc = {}

                    def add(d, s):
                        for t, x in d.items():
                            acc[t] = acc.get(t, 0) + s * x
                    for a, x in self.br(j, k).items():
                        add({t: x * y for t, y in self.br(i, a).items()}, 1)
                    for a, x in self.br(i, j).items():
                        add({t: x * y for t, y in self.br(a, k).items()}, -1)
                    s = (-1) ** (self.par(i) * self.par(j))
                    for a, x in self.br(i, k).items():
                        add({t: x * y for t, y in self.br(j, a).items()}, -s)
                    if any(acc.values()):
                        bad += 1
        return bad


class SPoly:
    """supercommutative polynomials in z_0..z_{d-1}, z_i of parity q_i;
    monomial = exponent tuple (odd exponents <= 1); element = {mono: Fraction}"""

    def __init__(self, q):
        self.q = q
        self.d = len(q)

    def mono_mul(self, A, B):
        for i in range(self.d):
            if self.q[i] and A[i] and B[i]:
                return None, 0
        inv = 0
        for i in range(self.d):
            if self.q[i] and B[i]:
                inv += sum(1 for j in range(i + 1, self.d) if self.q[j] and A[j])
        return tuple(a + b for a, b in zip(A, B)), (-1) ** inv

    def mul(self, u, v):
        res = {}
        for A, a in u.items():
            for B, b in v.items():
                C, s = self.mono_mul(A, B)
                if C is None:
                    continue
                res[C] = res.get(C, 0) + s * a * b
        return {k: x for k, x in res.items() if x}

    def dleft(self, k, A):
        """left derivative d/dz_k of the monomial A: (coefficient, monomial)"""
        if A[k] == 0:
            return 0, None
        B = list(A)
        B[k] -= 1
        if self.q[k]:
            s = (-1) ** sum(1 for j in range(k) if self.q[j] and A[j])
            return s, tuple(B)
        return A[k], tuple(B)

    def degree_basis(self, n):
        res = []

        def rec(i, left, cur):
            if i == self.d:
                if left == 0:
                    res.append(tuple(cur))
                return
            top = 1 if self.q[i] else left
            for a in range(min(top, left) + 1):
                rec(i + 1, left - a, cur + [a])
        rec(0, n, [])
        return res


def var(P, i):
    A = [0] * P.d
    A[i] = 1
    return {tuple(A): Fraction(1)}


def make_Q(g, signs):
    """Q(z_k) = 1/2 sum_{i,j} eps_{ij} c^k_{ij} z_i z_j"""
    a, b, c = signs
    q = [1 - g.par(i) for i in range(g.dim)]
    P = SPoly(q)
    Qk = []
    for k in range(g.dim):
        tot = {}
        for i in range(g.dim):
            for j in range(g.dim):
                x = g.br(i, j).get(k, 0)
                if not x:
                    continue
                pi, pj = g.par(i), g.par(j)
                e = (-1) ** (a * pi + b * pj + c * pi * pj)
                t = P.mul(var(P, i), var(P, j))
                for A, y in t.items():
                    tot[A] = tot.get(A, 0) + Fraction(e * x, 2) * y
        Qk.append({A: y for A, y in tot.items() if y})
    return P, Qk


def Q_apply(P, Qk, u):
    res = {}
    for A, a in u.items():
        for k in range(P.d):
            s, B = P.dleft(k, A)
            if not s:
                continue
            t = P.mul(Qk[k], {B: Fraction(s) * a})
            for C, y in t.items():
                res[C] = res.get(C, 0) + y
    return {k: x for k, x in res.items() if x}


def Q_squared_zero(P, Qk):
    for k in range(P.d):
        if Q_apply(P, Qk, Q_apply(P, Qk, var(P, k))):
            return False
    return True


def find_signs(samples):
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                ok = True
                for g in samples:
                    P, Qk = make_Q(g, (a, b, c))
                    if not Q_squared_zero(P, Qk) or not any(Qk):
                        ok = False
                        break
                if ok:
                    return (a, b, c)
    return None


def scale(P, A, lattice):
    """basis element of the lattice = monomial A / scale"""
    if lattice == "poly":
        return 1
    return prod(factorial(A[i]) for i in range(P.d) if not P.q[i])


def Q_matrix(P, Qk, n, lattice):
    src = P.degree_basis(n)
    tgt = P.degree_basis(n + 1)
    idx = {B: t for t, B in enumerate(tgt)}
    rows = []
    nonint = 0
    for A in src:
        cA = scale(P, A, lattice)
        img = Q_apply(P, Qk, {A: Fraction(1, cA)})
        r = [Fraction(0)] * max(1, len(tgt))
        for B, y in img.items():
            v = y * scale(P, B, lattice)
            if v.denominator != 1:
                nonint += 1
            r[idx[B]] = v
        rows.append(r)
    return rows, len(src), len(tgt), nonint


def cohom(P, Qk, maxdeg, p, lattice):
    mats = [Q_matrix(P, Qk, n, lattice) for n in range(maxdeg + 1)]
    rk = []
    for rows, ns, nt, _ in mats:
        rk.append(rank([[int(x) % p for x in r] for r in rows], p) if ns and nt else 0)
    H = []
    for n in range(maxdeg + 1):
        H.append(mats[n][1] - rk[n] - (rk[n - 1] if n else 0))
    nonint = sum(m[3] for m in mats)
    return H, nonint


def samples():
    s = []
    s.append(SLAQ(1, 1, {(1, 1): {0: 2}}, "(1|1): [f,f]=2e, e central  (U=K[f])"))
    s.append(SLAQ(1, 1, {(0, 1): {1: 1}}, "(1|1): [e,f]=f, [f,f]=0"))
    s.append(SLAQ(1, 2, {(1, 2): {0: 1}}, "(1|2): [f1,f2]=e, e central"))
    # gl(1|1): e0=E11, e1=E22, f0=E12 (idx 2), f1=E21 (idx 3)
    s.append(SLAQ(2, 2, {(0, 2): {2: 1}, (0, 3): {3: -1}, (1, 2): {2: -1},
                         (1, 3): {3: 1}, (2, 3): {0: 1, 1: 1}}, "gl(1|1)"))
    # osp(1|2): h=0, e=1, f=2, x=3, y=4
    s.append(SLAQ(3, 2, {(0, 1): {1: 2}, (0, 2): {2: -2}, (1, 2): {0: 1},
                         (0, 3): {3: 1}, (0, 4): {4: -1}, (1, 4): {3: -1},
                         (2, 3): {4: -1}, (3, 3): {1: 2}, (4, 4): {2: -2},
                         (3, 4): {0: 1}}, "osp(1|2)"))
    return s


def podd_numbers(g, maxdeg, p):
    from podd import SLAz, ModZ, ACZ
    br = {}
    for (i, j), v in g.c.items():
        if i <= j:
            br[(i, j)] = tuple(v.get(k, 0) for k in range(g.dim))
    gz = SLAz(g.m, g.n, br)
    ac = ACZ(gz, ModZ.trivial(gz))
    return (ac.cohom(maxdeg, p, divided=False), ac.cohom(maxdeg, p, divided=True))


def dpco_numbers(g, maxdeg):
    """Ext over U(g) at p = 2 from dpco.py (squaring s(f) = [f,f]/2 mod 2)"""
    from dpco import SLA2, Resolution, cohomology_dims
    br = {}
    for (i, j), v in g.c.items():
        if i < j:
            mask = 0
            for k, x in v.items():
                if x % 2:
                    mask |= 1 << k
            if mask:
                br[(i, j)] = mask
    sq = []
    for f in range(g.m, g.dim):
        mask = 0
        for k, x in g.br(f, f).items():
            assert x % 2 == 0
            if (x // 2) % 2:
                mask |= 1 << k
        sq.append(mask)
    g2 = SLA2(g.m, g.n, br, sq)
    assert not g2.check_axioms(), "mod-2 reduction not a Lie superalgebra"
    res = Resolution(g2)
    return cohomology_dims(res, g2.trivial_module(), maxdeg)


def part_B():
    out("=" * 78)
    out("B)  the cochains: polynomial functions on Pi g vs functions on its")
    out("    Grassmann neighborhood (divided powers in the even coordinates)")
    out("=" * 78)
    S = samples()
    for g in S:
        jd = g.jacobi_defect()
        out("   %-42s Jacobi defects: %d" % (g.name, jd))
    sg = find_signs(S)
    out("   sign convention found by Q^2 = 0 on all samples: %s" % (sg,))
    assert sg is not None
    maxdeg = 7
    for g in S:
        out("-" * 78)
        out(g.name)
        P, Qk = make_Q(g, sg)
        out("   Q^2 = 0: %s" % Q_squared_zero(P, Qk))
        for p in (2, 3, 5, 7):
            Hp, nip = cohom(P, Qk, maxdeg, p, "poly")
            Hg, nig = cohom(P, Qk, maxdeg, p, "grass")
            line = ("   p=%d  poly H^n=%s  grass H^n=%s  [non-integral: %d, %d]"
                    % (p, Hp[:maxdeg], Hg[:maxdeg], nip, nig))
            if p == 2:
                try:
                    ext = dpco_numbers(g, maxdeg - 1)
                    line += "   dpco Ext=%s %s" % (
                        ext, "poly=Ext" if ext == Hp[:maxdeg] else "poly!=Ext")
                except AssertionError as e:
                    line += "   (dpco: %s)" % e
            else:
                ah, dp = podd_numbers(g, maxdeg - 1, p)
                L = len(dp)
                line += ("   podd AH=%s dp=%s  %s %s" % (
                    ah, dp, "poly=dp" if dp == Hp[:L] else "poly!=dp",
                    "grass=AH" if ah == Hg[:L] else "grass!=AH"))
            out(line)
            fd = next((n for n in range(maxdeg) if Hp[n] != Hg[n]), None)
            out("          first degree where the two lattices differ: %s"
                % ("none up to %d" % (maxdeg - 1) if fd is None else fd))
    out()


def main():
    out("sshcheck.py -- Schwarz-Shapiro Grassmann divided powers vs [BGLL]")
    out("18 October 2026;  exact over Z/Q, ranks mod p;  backend: %s" % backend())
    out()
    part_A()
    part_B()
    out("C)  the chains: the Z-dual of the poly cochain lattice is the divided")
    out("    power chain lattice (functions on the Grassmann neighborhood of 0 in")
    out("    Pi(g^*)); over F_p its homology is the transpose of the poly")
    out("    cochain complex, so dim dpH_n = dim (poly H^n) above, degree by degree.")
    with open(os.path.join(_HERE, "sshcheck.log"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
