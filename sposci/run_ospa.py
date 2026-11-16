import sys, random
from fractions import Fraction as F
import sympy as sp
import flint
sys.path.insert(0, '.')
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))
from exactlin import rank
from ospa import *

def sig_of(a): return (-1 - a, 1, a)

def mats(sig, kmax):
    S = setup(sig)
    out = {}
    for cv in (0, 1):
        chi = character(S[0], S[2], S[3], S[7], F(cv))
        V = Verma(S, chi)
        for k in range(1, kmax + 1):
            B = V.basis(k); Bm = V.basis(k - 1)
            pos = {m: i for i, m in enumerate(Bm)}
            rows = []
            for x in V.g1:
                blk = [[F(0)] * len(B) for _ in Bm]
                for j, m in enumerate(B):
                    for m2, a in V.act(x, m).items():
                        blk[pos[m2]][j] += F(a)
                rows.extend(blk)
            out.setdefault(k, []).append(rows)
    return out, V

def crit(A0, A1, n):
    c = sp.Symbol('c')
    g = None
    for seed in (7, 11, 13):
        p = detpoly(A0, A1, n, seed, c)
        g = p if g is None else sp.gcd(g, p)
    if g == 0: return 'ALL'
    g = sp.Poly(g, c)
    good = []
    for r in sp.roots(g):
        if r.is_rational:
            rr = F(int(r.p), int(r.q)); rows = len(A0)
            Mc = [[A0[i][j] + rr * (A1[i][j] - A0[i][j]) for j in range(n)] for i in range(rows)]
            if rank(Mc) < n: good.append(rr)
        else:
            good.append(('irr', r))
    return sorted(good, key=str), str(sp.factor(g.as_expr()))

def detpoly(A0, A1, n, seed, c):
    rows = len(A0)
    random.seed(seed)
    R = [[random.randint(-5, 5) for _ in range(rows)] for _ in range(n)]
    def M(c):
        Mc = [[A0[i][j] + c * (A1[i][j] - A0[i][j]) for j in range(n)] for i in range(rows)]
        return [[sum(R[r][i] * Mc[i][j] for i in range(rows)) for j in range(n)] for r in range(n)]
    pts = list(range(n + 1))
    vals = [flint.fmpq_mat([[flint.fmpq(x.numerator, x.denominator) for x in row] for row in M(F(p))]).det() for p in pts]
    return sp.interpolate([(p, sp.Rational(int(v.p), int(v.q))) for p, v in zip(pts, vals)], c)

kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
avals = [F(x) for x in sys.argv[2:]] or [F(1), F(2), F(1,2), F(-3), F(3,5)]
for a in avals:
    sig = sig_of(a)
    S = build(sig)
    print('a', a, 'jacobi failures', jacobi(S[0], S[1], S[3], S[4]), flush=True)
    out, V = mats(sig, kmax)
    chi1 = character(V.names, V.idx, V.bracket, V.g0, F(1))
    print('  chi(h1,h2,h3) at c=1:', [chi1.get(V.idx[('h', i)]) for i in range(3)], flush=True)
    for k in range(1, kmax + 1):
        A0, A1 = out[k]
        n = len(A0[0])
        print('  k', k, 'dim', n, 'singular c (M):', crit(A0, A1, n), flush=True)
