r"""Example 1.10.2: superdimensions of Omega^1, Omega^3, P^3 and X^+(Omega^1).

Run (from this folder):  ../venv/bin/python halves_sdim.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys, warnings
warnings.filterwarnings("ignore")
from fractions import Fraction
from exactlin import rank
from halves import structures, matrix_Xplus, matrix_Xminus, mat_mul


def parity(F, mon):
    return sum(F.par[i] for i in mon) % 2


def split(F, basis):
    ev = [m for m in basis if parity(F, m) == 0]
    od = [m for m in basis if parity(F, m) == 1]
    return ev, od


def report(kind, n, m=0):
    F, om, xm, label = structures(kind, n, m)
    pw = sum(F.par[i] for i in list(om)[0]) % 2      # parity of omega
    O1, O3 = F.monomials(1), F.monomials(3)
    e1, o1 = split(F, O1)
    e3, o3 = split(F, O3)
    print("=" * 68)
    print("%s   (p(omega) = %d)" % (label, pw))
    print("   sdim Omega^1 = (%d|%d),  sdim Omega^3 = (%d|%d)"
          % (len(e1), len(o1), len(e3), len(o3)))
    # X^+ : Omega^1_eps -> Omega^3_{eps+p(omega)};  X^- : Omega^3_eps -> Omega^1_{eps+p(omega)}
    tgt = {0: (e3 if pw == 0 else o3), 1: (o3 if pw == 0 else e3)}
    src1 = {0: e1, 1: o1}
    im = {}
    for eps in (0, 1):
        A = matrix_Xplus(F, om, src1[eps], tgt[eps])
        im[(eps + pw) % 2] = rank(A) if A and A[0] else 0
    tg1 = {0: (e1 if pw == 0 else o1), 1: (o1 if pw == 0 else e1)}
    src3 = {0: e3, 1: o3}
    ker = {}
    for eps in (0, 1):
        A = matrix_Xminus(F, xm, src3[eps], tg1[eps])
        ker[eps] = len(src3[eps]) - (rank(A) if A and A[0] else 0)
    print("   sdim X^+(Omega^1) = (%d|%d),  sdim P^3 = (%d|%d)"
          % (im[0], im[1], ker[0], ker[1]))
    print("   sum of the two = (%d|%d)" % (im[0] + ker[0], im[1] + ker[1]))


for args in (("even", 2, 1), ("even", 3, 1), ("even", 3, 2),
             ("odd", 2), ("odd", 3)):
    report(*args)
