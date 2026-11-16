"""Dimension of the space of bilinear operators T(lam) x T(mu) -> T(nu) of a given
order that commute with h_{-1} + h_0 -- that is, of the schemes that actually land
in the fiber V_nu.  Exactly the number of terms a formula written in schemes needs.
"""
import argparse, os, random, sys
from fractions import Fraction
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace
from symp import Symp
from fiber import Fiber
from gen import Problem
from rungen import rand_field, parse_part

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=2)
ap.add_argument("--lam", default="2,1"); ap.add_argument("--mu", default="2")
ap.add_argument("--nu", default="2,1"); ap.add_argument("--order", type=int, default=2)
ap.add_argument("--samples", type=int, default=3)
a = ap.parse_args()
S = Symp(a.m)
Fl, Fm, Fn = Fiber(S, parse_part(a.lam)), Fiber(S, parse_part(a.mu)), Fiber(S, parse_part(a.nu))
P = Problem(S, Fl, Fm, Fn, a.order)
rng = random.Random(31415)
rows = []
for _ in range(a.samples):
    A = rand_field(Fl, rng, 3, 3); B = rand_field(Fm, rng, 3, 3)
    rows += P.membership_rows(A, B)
ns = nullspace(rows) if rows else [[Fraction(i == j) for i in range(len(P.terms))]
                                   for j in range(len(P.terms))]
rng2 = random.Random(2718)
probes = [(rand_field(Fl, rng2, 3, 3), rand_field(Fm, rng2, 3, 3)) for _ in range(2)]
r, _ = P.operator_rank(ns, probes)
print(f"m={a.m} T{Fl.lam}xT{Fm.lam}->T{Fn.lam} order {a.order}: {len(P.terms)} schemes; "
      f"{r} independent operators landing in the fiber")
