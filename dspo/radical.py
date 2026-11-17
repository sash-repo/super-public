#!/usr/bin/env python3
"""
Statement 3.2.6 (m = 5): g_x / r(g_x) for po(0|m): the solvable radical r of g_x = Ker ad_x/Im ad_x is
computed by iterating the nilradical, N = {y | tr(ad_y b)=0 for all b in A},
A = the unital associative envelope of ad(g) in End(g) (Dickson's criterion;
valid mod p as soon as p exceeds every multiplicity, here p ~ 2^31), and
then the semisimple quotient s = g_x/r is described: sdim, dim of its
envelope A (= (dim s)^2 iff the adjoint module is absolutely irreducible),
dim of the centroid, dim s' = dim [s,s].
Modulo the prime p = 2^31 - 1. An independent cross-check of radical2.py, which
uses a sampled envelope; this one builds the whole envelope, which is slow beyond m = 5.
Usage: python3 radical.py 5
"""
import sys
from fractions import Fraction
import flint
from po import Po, Quot, homologic
from porun import type_reps

P = 2147483647

def nm(rows, ncols, p):
    return flint.nmod_mat(len(rows), ncols, [int(c) % p for r in rows for c in r], p)

def rref_rows(rows, ncols, p):
    if not rows:
        return []
    R, rk = nm(rows, ncols, p).rref()
    return [[int(R[i, j]) for j in range(ncols)] for i in range(rk)]

def nullspace(rows, ncols, p):
    if not rows:
        return [[1 if i == j else 0 for j in range(ncols)] for i in range(ncols)]
    X, nul = nm(rows, ncols, p).nullspace()
    return [[int(X[i, j]) for i in range(ncols)] for j in range(nul)]

class Alg:
    def __init__(self, tab, par, p):
        self.d = len(par); self.tab = tab; self.par = par; self.p = p
        d = self.d
        self.ads = [flint.nmod_mat(d, d, [tab[i][j][t] % p for t in range(d) for j in range(d)], p)
                    for i in range(d)]
    def envelope(self):
        d, p = self.d, self.p
        flat = lambda M: [int(M[a, b]) for a in range(d) for b in range(d)]
        I = flint.nmod_mat(d, d, [1 if a == b else 0 for a in range(d) for b in range(d)], p)
        basis = rref_rows([flat(I)] + [flat(g) for g in self.ads], d*d, p)
        frontier = [flint.nmod_mat(d, d, v, p) for v in basis]
        while True:
            cand = [flat(g * M) for M in frontier for g in self.ads]
            new = rref_rows(basis + cand, d*d, p)
            if len(new) == len(basis):
                break
            # new frontier = the new basis elements (a spanning set of everything)
            basis = new
            frontier = [flint.nmod_mat(d, d, v, p) for v in basis]
        return [flint.nmod_mat(d, d, v, p) for v in basis]
    def nilradical(self, env):
        d, p = self.d, self.p
        tr = lambda M: sum(int(M[i, i]) for i in range(d)) % p
        rows = [[tr(self.ads[i] * b) for i in range(d)] for b in env]
        return nullspace(rows, d, p)
    def quotient(self, sub):
        d, p = self.d, self.p
        base = [list(r) for r in sub]
        comp = []
        k = len(base)
        for i in range(d):
            e = [1 if j == i else 0 for j in range(d)]
            if len(rref_rows(base + [e], d, p)) > len(base):
                base.append(e); comp.append(i)
        B = flint.nmod_mat(d, d, [base[c][r] % p for r in range(d) for c in range(d)], p)
        Binv = B.inv()
        nd = len(comp)
        tab = [[None]*nd for _ in range(nd)]
        for a, i in enumerate(comp):
            for b, j in enumerate(comp):
                w = flint.nmod_mat(d, 1, [c % p for c in self.tab[i][j]], p)
                co = Binv * w
                tab[a][b] = [int(co[k + t, 0]) for t in range(nd)]
        return Alg(tab, [self.par[i] for i in comp], p)
    def sdim(self):
        return (self.par.count(0), self.par.count(1))
    def derived_dim(self):
        d = self.d
        return len(rref_rows([self.tab[i][j] for i in range(d) for j in range(d)], d, self.p))
    def centroid_dim(self):
        d, p = self.d, self.p
        rows = []
        for A in self.ads:
            for a in range(d):
                for b in range(d):
                    r = [0] * (d*d)
                    for c in range(d):
                        r[a*d + c] += int(A[c, b])
                        r[c*d + b] -= int(A[a, c])
                    rows.append(r)
        return d*d - len(rref_rows(rows, d*d, p))

def radical_quotient(tab, par, p):
    cur = Alg(tab, par, p)
    steps = []
    while cur.d:
        env = cur.envelope()
        N = cur.nilradical(env)
        if not N:
            return cur, steps, env
        steps.append(len(N))
        cur = cur.quotient(N)
    return cur, steps, []

def modp(c, p):
    c = Fraction(c)
    return c.numerator * pow(c.denominator, -1, p) % p

if __name__ == "__main__":
    for m in map(int, sys.argv[1:] or ["5", "6", "7"]):
        alg = Po(m)
        reps, mult = type_reps(alg)
        for ty, msk in sorted(reps.items()):
            x = {msk: 1}
            if not homologic(alg, x): continue
            q = Quot(alg, x)
            if q.dim == 0: continue
            T = q.table()
            outs = []
            for p in (2147483647, 1000000007):
                tab = [[[modp(c, p) for c in v] for v in row] for row in T]
                s, steps, env = radical_quotient(tab, q.par, p)
                o = f"nil-steps={steps} sdim g_x/r={s.sdim()}"
                if s.d:
                    o += f" dim env={len(env)} (d^2={s.d**2}) centroid={s.centroid_dim()} dim s'={s.derived_dim()}"
                outs.append(o)
            agree = "" if outs[0] == outs[1] else "  PRIMES DISAGREE: " + outs[1]
            print(f"m={m} {alg.name(msk):>16} rank={q.rank:>3} sdim g_x={q.sdim()} {outs[0]}{agree}", flush=True)
