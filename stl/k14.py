"""k^L(1|4) (and the Moebius k^M(1|4)) in the theta-basis:
{f,g} = (2-E)(f) d_x g - d_x f (2-E)(g) - (-1)^{p(f)} sum_i d_i f d_i g.
The functionals (1), (2), (3) of Table (77) on the monomial windows (1) needs the operator
Delta_IS of Proposition 2.8.10, defined on k^L'(1|4) only).  Used by h2k14.py.
Run:  python3 k14.py 1   (Jacobi on the window, and the cocycle test of (1), (2), (3) on k^L'(1|4)
and on k^L(1|4): triples tested, failures, triples where the functional is undefined).
"""
from fractions import Fraction as Q
from itertools import product, combinations
import sys, os
import os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"), os.path.dirname(os.path.abspath(__file__))]
from exactlin import rank, nullspace, solve
from stringy import mono, add, smul, mul, d_t, d_odd, parity

ODD = (1, 2, 3, 4)
TOP = (1, 2, 3, 4)

def E(f):
    return {k: len(k[1]) * v for k, v in f.items() if len(k[1])}
def nab(f):
    return {k: (2 - len(k[1])) * v for k, v in f.items() if 2 - len(k[1])}
def br(f, g):
    s = {}
    for i in ODD:
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    return add(mul(nab(f), d_t(g)), smul(-1, mul(d_t(f), nab(g))),
               smul(-((-1) ** parity(f)), s))

def Res(f):
    return f.get((-1, TOP), Q(0))

def subsets():
    out = []
    for r in range(5):
        out += list(combinations(ODD, r))
    return out

def monos(N, moebius=False):
    out = []
    for S in subsets():
        half = Q(1, 2) if (moebius and 4 in S) else 0
        for a in range(-N, N + 1):
            out.append(mono(a + half, S))
    return out

def deg(f):
    ((a, S), _), = f.items()
    return 2 * a + len(S)

# --------------------------------------------------- the three functionals
def Dth(i, g):          # D_{theta_i} = theta_i d_x + d_{theta_i}
    return add(mul(mono(0, (i,)), d_t(g)), d_odd(i, g))
def D1inv(g):
    out = {}
    for (a, S), c in g.items():
        if a == -1:
            return None                         # undefined
        out[(a + 1, S)] = c / (a + 1)
    return out
def c1(f, g):
    h = g
    for i in (4, 3, 2, 1):                      # D_th1 D_th2 D_th3 D_th4 (g)
        h = Dth(i, h)
    h = D1inv(h)
    if h is None: return None
    return (-1) ** parity(f) * Res(mul(f, h))
def c2(f, g):
    h = {(a - 1, S): v for (a, S), v in nab(g).items()}
    return Res(mul(f, h))
def c3(f, g):
    return Res(mul(f, d_t(g)))

# ------------------------------------------------------ checks
def jacobi(B):
    bad = 0
    for f, g, h in product(B, repeat=3):
        lhs = br(f, br(g, h))
        rhs = add(br(br(f, g), h), smul((-1) ** (parity(f) * parity(g)), br(g, br(f, h))))
        if add(lhs, smul(-1, rhs)): bad += 1
    return bad

def lin(c, X, Y):
    """c on arbitrary (non-monomial) X,Y by bilinearity over monomials."""
    s = Q(0)
    for k1, v1 in X.items():
        for k2, v2 in Y.items():
            val = c({k1: Q(1)}, {k2: Q(1)})
            if val is None: return None
            s += v1 * v2 * val
    return s

def cocycle_fail(c, B, need_wt0=True):
    bad = 0; n = 0; undefined = 0
    for f, g, h in product(B, repeat=3):
        if deg(f) + deg(g) + deg(h) != 6: continue
        sgn = (-1) ** (parity(f) * parity(g))
        vals = [lin(c, f, br(g, h)), lin(c, br(f, g), h), lin(c, g, br(f, h))]
        if None in vals: undefined += 1; continue
        n += 1
        if vals[0] - vals[1] - sgn * vals[2] != 0: bad += 1
    return n, bad, undefined

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    BL = monos(N)
    z = mono(-1, TOP)
    Bprime = [f for f in BL if f != z]
    print("Jacobi failures k^L(1|4), window", N, ":", jacobi(monos(min(N, 1))))
    for name, c in (("(1)", c1), ("(2)", c2), ("(3)", c3)):
        print(name, "on k^L'(1|4): (triples, failures, undefined) =", cocycle_fail(c, Bprime))
        print(name, "on k^L (1|4): (triples, failures, undefined) =", cocycle_fail(c, BL))
