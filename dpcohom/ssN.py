#!/usr/bin/env python3
"""ssN.py -- test the spectral sequence of the square-weight
filtration on the divided power complex C = Hom(Lambda(V) (x) Gamma(g'), M),
p = 2, exact over F_2.

weight j of a monomial  prod gamma_{a_k}(x_k)  =  sum floor(a_k/2),
"square-free degree" i = n - 2j.  N = F^1 (weight >= 1).

Checks, degrees n <= MAXDEG:
 (1) the components of the codifferential, by the weight jump Delta;
 (2) E_1 dims  vs  dim H^i(g;M) * binom(dim g' + j - 1, j);
 (3) E_2, ..., E_infty, dim H^n(N), dim H^n_res, sum E_infty = H^n;
 (4) (a) H^2(N) = dim g' * dim M^g,
     (b) H^3(N) = dim ker( H^1(g;M) (x) g'^* -> M^g (x) S^2, c(x)phi |-> theta(c) phi ),
     (c) d_1 on E_1^{1,0} is theta, and d_1(c (x) phi) = theta(c) phi for
         phi of square-weight 1 and 2 (S-linearity), d_0(c (x) phi) = 0.

Usage:  python ssN.py
"""

import sys
from math import comb
from collections import Counter

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
sys.setrecursionlimit(100000)

import dpco
from dpco import bits, rank2_int
from restcx import RLA, Closure

MAXDEG = 5
RMAX = 8


# ------------------------------------------------------------ linear algebra

def kernel(images, nsrc):
    piv, ker = {}, []
    for j in range(nsrc):
        cur, cmb = images[j], 1 << j
        while cur:
            h = cur.bit_length() - 1
            if h in piv:
                cur ^= piv[h][0]
                cmb ^= piv[h][1]
            else:
                piv[h] = (cur, cmb)
                break
        if cur == 0:
            ker.append(cmb)
    return ker


def apply_map(images, v):
    o = 0
    for j in bits(v):
        o ^= images[j]
    return o


def dim(vecs):
    return rank2_int(list(vecs))


# -------------------------------------------------------------- the complex

def codiff(ce, M, dimalg, n):
    """rows[a*Md+i] = image of the basis cochain (src monomial a, v_i)"""
    src, dst = ce.basis(n), ce.basis(n + 1)
    sidx = {c: a for a, c in enumerate(src)}
    Md = M.dim
    rows = [0] * (len(src) * Md)
    for b, cc in enumerate(dst):
        for k in range(dimalg):
            p = ce.partial(k, cc)
            if p is not None and p in sidx:
                a = sidx[p]
                for i in range(Md):
                    for t in bits(M.apply(k, 1 << i)):
                        rows[a * Md + i] ^= 1 << (b * Md + t)
        for c2 in ce.delta(cc):
            if c2 in sidx:
                a = sidx[c2]
                for i in range(Md):
                    rows[a * Md + i] ^= 1 << (b * Md + i)
    return rows


class Filtered:
    def __init__(self, ce, M, dimalg, maxdeg):
        self.Md = M.dim
        self.mono = [ce.basis(n) for n in range(maxdeg + 2)]
        self.d = [codiff(ce, M, dimalg, n) for n in range(maxdeg + 1)]
        self.wt = []
        for n in range(maxdeg + 2):
            w = []
            for (S, al) in self.mono[n]:
                ww = sum(a // 2 for a in al)
                w += [ww] * self.Md
            self.wt.append(w)
        self.N = [len(w) for w in self.wt]
        self.maxdeg = maxdeg

    def mask(self, n, p, floor=0):
        p = max(p, floor)
        m = 0
        for b, w in enumerate(self.wt[n]):
            if w >= p:
                m |= 1 << b
        return m

    def F(self, n, p, floor=0):
        if n < 0:
            return []
        p = max(p, floor)
        return [1 << b for b, w in enumerate(self.wt[n]) if w >= p]

    def Z(self, n, p, r, floor=0):
        """Z_r^{p,n} = F^p C^n  cap  d^{-1}(F^{p+r}); r <= 0 gives F^p"""
        if n < 0:
            return []
        Fp = self.F(n, p, floor)
        if r <= 0:
            return Fp
        notm = ~self.mask(n + 1, p + r, floor)
        imgs = [apply_map(self.d[n], v) & notm for v in Fp]
        out = []
        for c in kernel(imgs, len(Fp)):
            w = 0
            for a in bits(c):
                w ^= Fp[a]
            out.append(w)
        return out

    def dZ(self, n, p, r, floor=0):
        """d(Z_r^{p,n-1})"""
        if n - 1 < 0:
            return []
        return [apply_map(self.d[n - 1], v) for v in self.Z(n - 1, p, r, floor)]

    def E(self, n, p, r, floor=0):
        """dim E_r^{p,n} = dim Z_r^p - dim(Z_{r-1}^{p+1} + d Z_{r-1}^{p-r+1})"""
        if p < floor:
            return 0
        den = self.Z(n, p + 1, r - 1, floor) + self.dZ(n, p - r + 1, r - 1, floor)
        return dim(self.Z(n, p, r, floor)) - dim(den)

    def H(self, n, floor=0):
        Fn = self.F(n, 0, floor)
        imgs = [apply_map(self.d[n], v) for v in Fn]
        z = len(kernel(imgs, len(Fn)))
        b = dim(self.dZ(n, floor, 0, floor)) if n > 0 else 0
        return z - b

    def comp_jumps(self):
        """Counter of (Delta j, Delta i) over the nonzero matrix entries"""
        cnt = Counter()
        for n in range(self.maxdeg + 1):
            for a, row in enumerate(self.d[n]):
                for b in bits(row):
                    dj = self.wt[n + 1][b] - self.wt[n][a]
                    cnt[dj] += 1
        return cnt


def ord_cohom(br, dimalg, M, maxdeg):
    """H^i of the plain Lie algebra (all letters exterior)"""
    g = RLA(dimalg, br, dimalg, [], "g")
    ce = g.res.ce
    d = [codiff(ce, M, dimalg, n) for n in range(maxdeg + 1)]
    out = []
    for n in range(maxdeg + 1):
        z = len(kernel(d[n], len(d[n])))
        b = dim(d[n - 1]) if n > 0 else 0
        out.append(z - b)
    return out, ce, d


# -------------------------------------------------------------------- run

def run(log, name, L, dimalg, br, nfree, M, gseq=None):
    """L: RLA (nfree exterior letters, the rest g' with squaring);
    gseq: for a closure, dim g (the N of the comparison sequence, as a check)"""
    out = log.append
    dg = dimalg - nfree                      # dim g'
    ce = L.res.ce
    Fc = Filtered(ce, M, dimalg, MAXDEG)
    D = MAXDEG
    jumps = Fc.comp_jumps()
    out("   M = %s (dim %d), dim g = %d, dim g' = %d" % (M.name, M.dim, dimalg, dg))
    out("   (1) weight jumps of d (Delta j: #entries): %s   %s"
        % (dict(sorted(jumps.items())),
           "only 0,1" if set(jumps) <= {0, 1} else
           ("FILTERED but extra components" if min(jumps) >= 0 else "NOT FILTERED")))
    Hord, oce, od = ord_cohom(br, dimalg, M, D)
    out("       H^i(g;M) plain Lie algebra, i=0..%d: %s" % (D, Hord))
    # (2) E_1
    bad1 = []
    rows = []
    for n in range(D + 1):
        for p in range(n // 2 + 1):
            i = n - 2 * p
            e = {r: Fc.E(n, p, r) for r in (1, 2, 3, 4, RMAX)}
            pred = Hord[i] * comb(dg + p - 1, p) if i <= D else None
            if e[1] != pred:
                bad1.append((i, p, e[1], pred))
            rows.append((n, i, p, e, pred))
    out("   (2) E_1^{i,j} = dim H^i(g;M) binom(dg+j-1,j): %s"
        % ("OK, all %d spots" % len(rows) if not bad1 else "FAILS at %s" % bad1))
    out("   (3) full complex C_res: n  (i,j): E1 E2 E3 E4 Einf | pred")
    degen = 1
    for (n, i, p, e, pred) in rows:
        for r in (1, 2, 3, 4):
            if e[r] != e[RMAX]:
                degen = max(degen, r + 1)
    for n in range(D + 1):
        s = "       n=%d: " % n
        tot = 0
        for (nn, i, p, e, pred) in rows:
            if nn == n:
                s += "(%d,%d):%d %d %d %d %d|%s  " % (i, p, e[1], e[2], e[3], e[4],
                                                   e[RMAX], pred)
                tot += e[RMAX]
        Hres = Fc.H(n)
        s += " sumEinf=%d H_res=%d %s" % (tot, Hres, "OK" if tot == Hres else "MISMATCH")
        out(s)
    out("       full SS: E_r = E_inf from r = %d" % degen)
    # N = F^1
    out("       N = F^1:")
    degN = 1
    HN = []
    for n in range(D + 1):
        s = "       n=%d: " % n
        tot = 0
        for p in range(1, n // 2 + 1):
            i = n - 2 * p
            e = {r: Fc.E(n, p, r, floor=1) for r in (1, 2, 3, 4, RMAX)}
            for r in (1, 2, 3, 4):
                if e[r] != e[RMAX]:
                    degN = max(degN, r + 1)
            s += "(%d,%d):%d %d %d %d %d  " % (i, p, e[1], e[2], e[3], e[4], e[RMAX])
            tot += e[RMAX]
        h = Fc.H(n, floor=1)
        HN.append(h)
        s += " sumEinf=%d H(N)=%d %s" % (tot, h, "OK" if tot == h else "MISMATCH")
        out(s)
    out("       N SS: E_r = E_inf from r = %d" % degN)
    if gseq is not None:
        # the N of the comparison sequence of Theorem Tseq24 (letters of g only)
        out("       (closure) note: the N of Thm Tseq24 kills square-free g-monomials "
            "only; its H^n listed for comparison: %s" % Hseq_N(Fc, gseq))

    # (4) M^g, theta, d_1
    Mg = [v for v in range(1, 1 << M.dim)
          if all(M.apply(k, v) == 0 for k in range(dimalg))]
    dMg = dim(Mg)
    out("   (4a) H^2(N) = %d, dim g' * dim M^g = %d: %s"
        % (HN[2], dg * dMg, "OK" if HN[2] == dg * dMg else "FAILS"))
    res = theta_tests(Fc, ce, oce, od, L, M, dimalg, nfree, br)
    out("   (4c) d_1 = theta on Z^1 (%d cocycles): %s;  d_0(c phi)=0 and "
        "d_1(c phi)=theta(c)phi, |phi|=1: %s, |phi|=2: %s"
        % (res["nZ"], res["theta"], res["S1"], res["S2"]))
    kb = res["kerdim"]
    out("   (4b) H^3(N) = %d;  dim ker(H^1 (x) g'^* -> M^g (x) S^2, theta-product) = %d: %s"
        % (HN[3], kb, "OK" if HN[3] == kb else "FAILS"))
    return dict(E1ok=not bad1, degen=degen, degN=degN, HN=HN,
                Hres=[Fc.H(n) for n in range(D + 1)], Hord=Hord,
                jumps=dict(jumps), a=HN[2] == dg * dMg, b=HN[3] == kb,
                c=res["theta"] == "OK" and res["S1"] == "OK" and res["S2"] == "OK")


def Hseq_N(Fc, dim_g):
    """cohomology of the subcomplex of cochains vanishing on the monomials that
    are square-free in the letters < dim_g (all letters are g' here)"""
    out = []
    keep = []
    for n in range(Fc.maxdeg + 2):
        m = 0
        Md = Fc.Md
        for a, (S, al) in enumerate(Fc.mono[n]):
            sf = all(x <= 1 for x in al[:dim_g]) and all(x == 0 for x in al[dim_g:])
            if not sf:
                for t in range(Md):
                    m |= 1 << (a * Md + t)
        keep.append(m)
    for n in range(Fc.maxdeg + 1):
        B = [1 << b for b in bits(keep[n])]
        z = len(kernel([apply_map(Fc.d[n], v) for v in B], len(B)))
        if n > 0:
            Bp = [1 << b for b in bits(keep[n - 1])]
            bd = dim(apply_map(Fc.d[n - 1], v) for v in Bp)
        else:
            bd = 0
        out.append(z - bd)
    return out


def theta_tests(Fc, ce, oce, od, L, M, dimalg, nfree, br):
    Md = M.dim
    dg = dimalg - nfree
    m1 = Fc.mono[1]
    zero = tuple([0] * dg)

    def key(k, q):
        """monomial: letter k square-free, squares q (tuple over g')"""
        if k < nfree:
            return (frozenset([k]), tuple(2 * x for x in q))
        al = [2 * x for x in q]
        al[k - nfree] += 1
        return (frozenset(), tuple(al))

    def sqkey(q):
        return (frozenset(), tuple(2 * x for x in q))

    idx = [{c: a for a, c in enumerate(Fc.mono[n])} for n in range(len(Fc.mono))]
    # ordinary 1-cocycles in the coordinates of oce.basis(1)
    ob1 = oce.basis(1)
    Z1 = kernel(od[1], len(od[1]))
    B1 = list(od[0])

    def cval(z, k):
        a = ob1.index((frozenset([k]), ()))
        return (z >> (a * Md)) & ((1 << Md) - 1)

    def cochain(z, q, n):
        w = 0
        for k in range(dimalg):
            a = idx[n][key(k, q)]
            w |= cval(z, k) << (a * Md)
        return w

    def blk(v, n, c):
        a = idx[n][c]
        return (v >> (a * Md)) & ((1 << Md) - 1)

    def theta(z, l):
        """theta(c)(x_l) for l in g' (index l over g')"""
        x = nfree + l
        want = 0
        for k in bits(L.pmap[l]):
            want ^= cval(z, k)
        want ^= M.apply(x, cval(z, x))
        return want

    thok, s1, s2 = "OK", "OK", "OK"
    for z in Z1:
        # d_1 on E_1^{1,0}
        dz = apply_map(Fc.d[1], cochain(z, zero, 1))
        for b in bits(dz):
            if Fc.wt[2][b] == 0:
                thok = "FAILED (weight 0 part)"
        for l in range(dg):
            q = [0] * dg
            q[l] = 1
            if blk(dz, 2, sqkey(q)) != theta(z, l):
                thok = "FAILED"
        # S-linearity, |phi| = 1, 2
        for j, n in ((1, 3), (2, 5)):
            if n > Fc.maxdeg:
                continue
            qs = [q for q in dpco.gamma_basis(dg, j)]
            for q in qs:
                v = apply_map(Fc.d[n], cochain(z, q, n))
                st = "OK"
                for b in bits(v):
                    if Fc.wt[n + 1][b] == j:
                        st = "FAILED (d_0 != 0)"
                for m in range(dg):
                    q2 = list(q)
                    q2[m] += 1
                    if blk(v, n + 1, sqkey(q2)) != theta(z, m):
                        st = "FAILED (d_1 != theta phi)"
                if st != "OK":
                    if j == 1:
                        s1 = st
                    else:
                        s2 = st
    # kernel of H^1 (x) g'^* -> M (x) S^2 by c phi_l |-> theta(c) phi_l
    q2list = dpco.gamma_basis(dg, 2)
    q2i = {q: a for a, q in enumerate(q2list)}
    imgs = []
    for z in Z1:
        for l in range(dg):
            w = 0
            for m in range(dg):
                q = [0] * dg
                q[l] += 1
                q[m] += 1
                a = q2i[tuple(q)]
                w ^= theta(z, m) << (a * Md)
            imgs.append(w)
    rk = dim(imgs)
    kerdim = len(Z1) * dg - rk - dim(B1) * dg
    return dict(nZ=len(Z1), theta=thok, S1=s1, S2=s2, kerdim=kerdim)


# ------------------------------------------------------------------ samples

def mkbr(d, pairs):
    br = [[0] * d for _ in range(d)]
    for (i, j), v in pairs.items():
        br[i][j] = v
        br[j][i] = v
    return br


def bit(*ks):
    v = 0
    for k in ks:
        v |= 1 << k
    return v


def module(dimalg, Md, act, name):
    return dpco.Module(Md, [[act(k, 1 << i) for k in range(dimalg)]
                            for i in range(Md)], [0] * Md, name)


def adj(br, k, v):
    o = 0
    for i in bits(v):
        o ^= br[k][i]
    return o


def main():
    LOG = []

    def out(s=""):
        print(s)
        sys.stdout.flush()
        LOG.append(s)

    out("ssN.py -- the square-weight spectral sequence, exact over F_2,"
        " n <= %d" % MAXDEG)
    summary = []
    samples = []
    # name, dim, nfree, br, pmap (over g'), closure data
    samples.append(("S1 2-str abelian dim2, [2]=0", 2, 0, mkbr(2, {}), [0, 0], True))
    samples.append(("S2 2-str [x,y]=y, x^[2]=x", 2, 0, mkbr(2, {(0, 1): bit(1)}),
                    [bit(0), 0], True))
    samples.append(("S3 2-str Heisenberg, [2]=0", 3, 0, mkbr(3, {(0, 1): bit(2)}),
                    [0, 0, 0], True))
    samples.append(("S8 2-str sl(2): [e,f]=h, h^[2]=h", 3, 0,
                    mkbr(3, {(0, 1): bit(2)}), [0, 0, bit(2)], True))
    samples.append(("S4 2|4 gl(1|1), e1^[2]=e1", 4, 0,
                    mkbr(4, {(1, 2): bit(2), (1, 3): bit(3), (2, 3): bit(0)}),
                    [0, bit(1), 0, 0], True))
    # Lie superalgebras with squaring on the odd part only (V = g_ev exterior)
    samples.append(("T1 superalg V=<z>, g'=<a,b>, [a,b]=z, s=0", 3, 1,
                    mkbr(3, {(1, 2): bit(0)}), [0, 0], True))
    samples.append(("T2 superalg V=<h,z>, g'=<a,b>, [h,a]=a,[h,b]=b,[a,b]=z, s=0", 4, 2,
                    mkbr(4, {(0, 2): bit(2), (0, 3): bit(3), (2, 3): bit(1)}),
                    [0, 0], True))
    samples.append(("T3 superalg V=<z>, g'=<a>, s(a)=z", 2, 1, mkbr(2, {}),
                    [bit(0)], True))
    for (nm, d, nf, br, pm, doadj) in samples:
        L = RLA(d, br, nf, pm, nm)
        bad = L.check()
        out("=" * 78)
        out(nm + ("   [axioms OK]" if not bad else "   [AXIOMS FAIL %s]" % bad[:2]))
        mods = [module(d, 1, lambda k, v: 0, "K")]
        if doadj:
            Ma = module(d, d, lambda k, v, br=br: adj(br, k, v), "ad")
            if not L.check_restricted(Ma):
                mods.append(Ma)
            else:
                out("   (adjoint not restricted, skipped)")
        for M in mods:
            r = run(LOG, nm, L, d, br, nf, M)
            summary.append((nm, M.name, r))
    # (2,4)-structures via the closure
    cl24 = [("S5 (2,4) abelian, g+=<x>, g-=<y>", 1, 1, mkbr(2, {}), [0], [0]),
            ("S6 (2,4) o^(1)(3), H^[2]=H", 1, 2,
             mkbr(3, {(0, 1): bit(1), (0, 2): bit(2), (1, 2): bit(0)}), [bit(0)], [0, 0]),
            ("S7 (2,4) abelian, g+=<x1,x2>, g-=<y>", 2, 1, mkbr(3, {}), [0, 0], [0])]
    for (nm, dp, dm, br, p2, p4) in cl24:
        cl = Closure(dp, dm, br, p2, p4)
        L = cl.rla()
        dL = cl.dimL
        bad = L.check() + cl.problems
        out("=" * 78)
        out(nm + "  -> closure L, dim %d, all letters squared%s"
            % (dL, "   [OK]" if not bad else "   [FAIL %s]" % bad[:2]))
        dg = dp + dm
        mods = [cl.module_L(1, lambda k, v: 0)]
        mods[0].name = "K"
        Ma = cl.module_L(dg, lambda k, v, br=br: adj(br, k, v))
        Ma.name = "ad_g (lifted)"
        if not L.check_restricted(Ma):
            mods.append(Ma)
        for M in mods:
            r = run(LOG, nm, L, dL, cl.br, 0, M, gseq=dg)
            summary.append((nm, M.name, r))

    out("=" * 78)
    out("SUMMARY  (jumps = Delta j of nonzero entries; degen = first r with E_r=E_inf)")
    out("%-48s %-6s %-10s E1 a b c degC degN  H(N) n=0..%d   H_res" % ("sample", "M", "jumps", MAXDEG))
    for nm, mn, r in summary:
        out("%-48s %-6s %-10s %s  %s %s %s  %d    %d    %s  %s"
            % (nm[:48], mn[:6], ",".join(str(k) for k in sorted(r["jumps"])),
               "Y" if r["E1ok"] else "N", "Y" if r["a"] else "N",
               "Y" if r["b"] else "N", "Y" if r["c"] else "N",
               r["degen"], r["degN"], r["HN"], r["Hres"]))
    with open(os.path.join(_HERE, "ssN.log"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
