"""k^L(1|4) (and the Moebius k^M(1|4)) in the theta-basis:
{f,g} = (2-E)(f) d_x g - d_x f (2-E)(g) - (-1)^{p(f)} sum_i d_i f d_i g.
The functionals (1), (2), (3) of Table (77) on the monomial windows ((1) needs the operator
Delta_IS of Proposition 2.8.10, defined on k^L'(1|4) only, simplified before it is evaluated).  Used by h2k14.py.
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
def theta_ops(S):
    """D_th1 D_th2 D_th3 D_th4 applied to theta^S, kept as an operator: {(j, T): c} stands for
    c * theta^T d_x^j (d_x commutes with every theta-operator)."""
    cur = {(0, tuple(S)): Q(1)}
    for i in (4, 3, 2, 1):
        new = {}
        for (j, T), c in cur.items():
            if i not in T:                      # theta_i d_x
                Tn = tuple(sorted(T + (i,)))
                new[(j + 1, Tn)] = new.get((j + 1, Tn), 0) + (-1) ** sum(t < i for t in T) * c
            else:                               # d_theta_i
                Tn = tuple(t for t in T if t != i)
                new[(j, Tn)] = new.get((j, Tn), 0) + (-1) ** T.index(i) * c
        cur = {k: v for k, v in new.items() if v}
    return cur
def falling(a, k):
    r = Q(1)
    for t in range(k): r *= (a - t)
    return r
def DIS(g):
    """Delta_IS = D_1^{-1} D_th1 D_th2 D_th3 D_th4, SIMPLIFIED BEFORE it is evaluated (Sect. 2.8):
    sum_{j>=1} P_j d_x^{j-1} + P_0 D_1^{-1}.  Undefined (None) only on x^{-1}theta^S with a
    non-zero P_0-term, i.e. on the monomial x^{-1}theta_1theta_2theta_3theta_4 (Proposition 2.8.10).
    Evaluating D_th1...D_th4 first and dividing by d_x afterwards loses the constants killed
    by d_x (a 0 * 1/0), and then (1) is not a cocycle."""
    out = {}
    for (a, S), c in g.items():
        for (j, T), s in theta_ops(S).items():
            if j >= 1:
                co, e = falling(a, j - 1), a - (j - 1)
            else:
                if a == -1: return None
                co, e = Q(1) / (a + 1), a + 1
            if co: out[(e, T)] = out.get((e, T), 0) + c * s * co
    return {k: v for k, v in out.items() if v}
def c1(f, g):
    h = DIS(g)
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
