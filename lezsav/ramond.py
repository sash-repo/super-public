#!/usr/bin/env python3
"""
ramond.py -- the classical anomaly in the RAMOND sector, N = 1 and N = 2,
read off ALL sectors of k^M(1|N).

N = 1: the Ramond frame is the NS frame on the double cover t = s^2, theta =
s zeta; k^M(1|1) = the M-invariant generating functions, i.e. a0(t) + theta
t^{1/2} b(t); the anomaly is the same local expression -1/2 D^5 a (nscoc.py),
paired by the same Berezin residue.  So G_r = K_{-2 theta x^{r+1/2}}, r in Z.

N = 2: n2ram.py's basis; here the two REAL towers are tested on their own
diagonal pairs, {G^1_r, G^1_-r} (r in Z+1/2) and {G^2_k, G^2_-k} (k in Z).
"""
import sympy as sp
import nscoc as ns
from n2coc import F, bracket as br2, A, res
import n2ram

H = sp.Rational(1, 2)
bad = 0

print("N = 1, k^M(1|1)")
sig = -2
def G(r):                          # r in Z: odd part theta*x^{r+1/2}, M-invariant
    return ns.SF(0, sig * ns.x ** (H + r), 1)
for m in range(-2, 3):
    for k in range(-2, 3):
        b = ns.bracket(ns.L(m), G(k)) - G(m + k).scal(sp.Rational(m, 2) - k)
        bad += not b.simp().is_zero()
    for l in range(-2, 3):
        b = ns.bracket(G(m), G(l)) - ns.L(m + l).scal(2)
        bad += not b.simp().is_zero()
print("   relations [L,G], {G,G}=2L with r in Z: failures", bad)
for m in range(2, 5):
    v = ns.cocycle(ns.L(m), ns.L(-m)); print(f"   c(L_{m},L_-{m}) = {v}  -> c = {12*v/(m**3-m)}")
for k in range(0, 4):
    v = ns.cocycle(G(k), G(-k)); print(f"   c(G_{k},G_-{k}) = {v}  -> c/3*(r^2-1/4) gives c = {3*v/(k*k-H/2)}")
basis = [ns.L(m) for m in (-1, 0, 1, 2)] + [G(k) for k in (-1, 0, 1, 2)]
for a in basis:
    for b in basis:
        for c in basis:
            t = ((-1) ** (a.p * c.p) * ns.cocycle(ns.bracket(a, b), c)
                 + (-1) ** (b.p * a.p) * ns.cocycle(ns.bracket(b, c), a)
                 + (-1) ** (c.p * b.p) * ns.cocycle(ns.bracket(c, a), b))
            bad += sp.simplify(t) != 0
print("   cocycle identity on k^M(1|1), 512 triples: failures so far", bad)

print("N = 2, k^M(1|2): the diagonal supercharge pairs")
v1 = sp.I; v2 = sp.I               # n2ram: {G^i,G^i}=2L fixes c1 = c2 = I
G1 = lambda r: F.mono(sp.Rational(2 * r + 1, 2), (1,)).scal(v1)
G2 = lambda r: F.mono(sp.Rational(2 * r + 1, 2), (2,)).scal(v2)
for k in range(0, 4):
    r = sp.Rational(2 * k + 1, 2)
    v = n2ram.coc(G1(r), G1(-r)); print(f"   c(G^1_{r},G^1_-{r}) = {v}   -> c = {sp.nsimplify(3*v/(r*r-H/2))}")
for k in range(0, 4):
    v = n2ram.coc(G2(sp.Integer(k)), G2(sp.Integer(-k)))
    print(f"   c(G^2_{k},G^2_-{k}) = {v}   -> c = {sp.nsimplify(3*v/(k*k-H/2))}")
print("TOTAL failures", bad)
