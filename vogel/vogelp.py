"""Vogel's parameters of the exceptional modular Lie (super)algebras with Cartan
matrix of the text, in characteristic p = 3 and 5.

Data (vogelp-data.json, as printed in the text): the Cartan matrix, the positive
root vectors x_k given by their roots in the simple-root coordinates, the highest
weight vectors V_1, V_2 of S^2(g) as sums of monomials x_a x_b, and their printed
weights.  Parities: a_ii=2 even, a_ii=0 odd isotropic, a_ii=1 odd non-isotropic.

Casimir on a highest weight vector of weight lam: B(lam, lam+2rho), with
2rho = sum(even positive roots) - sum(odd positive roots) and B the form
induced by a symmetrization d_i a_ij = d_j a_ji (mod p).  Vogel's law
(P. Vogel, The universal Lie algebra, preprint 1999: half the Casimir acts on
X_2, Y_2(x) by 2t, 2t-x):  C|_g = 2t,  C|_{Y_2(x)} = 4t - 2x,  t = alpha+beta+gamma.
"""
import sys
from fractions import Fraction as Fr


def inv(a, p): return pow(a % p, p - 2, p)

def solve_mod(A, b, p):
    n = len(A); M = [[A[i][j] % p for j in range(n)] + [b[i] % p] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c])
        M[c], M[piv] = M[piv], M[c]
        iv = inv(M[c][c], p); M[c] = [x * iv % p for x in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]; M[r] = [(x - f * y) % p for x, y in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]

def symmetrize(A, p):
    n = len(A); d = [None] * n; d[0] = 1
    for _ in range(n):
        for i in range(n):
            for j in range(n):
                if d[i] is not None and d[j] is None and A[i][j] % p and A[j][i] % p:
                    d[j] = d[i] * A[i][j] * inv(A[j][i], p) % p
    for i in range(n):
        for j in range(n):
            assert (d[i] * A[i][j] - d[j] * A[j][i]) % p == 0, "not symmetrizable"
    return d

def run(name, A, p, leaves, V1, V2, printed, par=None):
    n = len(A)
    if par is None:
        par = [0 if A[i][i] % p == 2 % p else 1 for i in range(n)]
    d = symmetrize(A, p)
    def wt(v): return [sum(A[i][j] * v[j] for j in range(n)) % p for i in range(n)]
    def parity(v): return sum(a * b for a, b in zip(v, par)) % 2
    # 2rho in h-coordinates
    two_rho = [0] * n
    for k, v in leaves.items():
        s = -1 if parity(v) else 1
        two_rho = [(a + s * b) % p for a, b in zip(two_rho, wt(v))]
    def B(lam, mu):
        c = solve_mod(A, lam, p)       # lam = sum c_j alpha_j
        return sum(c[j] * d[j] * mu[j] for j in range(n)) % p
    def cas(lam): return B(lam, [(a + b) % p for a, b in zip(lam, two_rho)])
    top = max(leaves, key=lambda k: sum(leaves[k]))
    wtop = wt(leaves[top])
    def mono(m):
        v = [0] * n
        for t in m: v = [a + b for a, b in zip(v, leaves[t])]
        return wt(v)
    w1 = [mono(m) for m in V1]; w2 = [mono(m) for m in V2]
    ok1 = all(w == w1[0] for w in w1); ok2 = all(w == w2[0] for w in w2)
    pr = [[x % p for x in w] for w in printed]
    t2 = cas(wtop); i2 = inv(2, p)
    t = t2 * i2 % p
    c1, c2 = cas(w1[0]), cas(w2[0])
    al = (4 * t - c1) * i2 % p; be = (4 * t - c2) * i2 % p; ga = (t - al - be) % p
    npos = len(leaves); nodd = sum(parity(v) for v in leaves.values())
    sd = (2 * (npos - nodd) + n, 2 * nodd)
    print(f"{name}  p={p}  par={par}  #pos={npos}  sdim={sd}  top=x_{top} wt={wtop}")
    print(f"   V1 wt {w1[0]} homog={ok1} printed {pr[0]} {'OK' if w1[0]==pr[0] else 'DIFF'};"
          f"  V2 wt {w2[0]} homog={ok2} printed {pr[1]} {'OK' if w2[0]==pr[1] else 'DIFF'}")
    print(f"   C(g)={t2} C(V1)={c1} C(V2)={c2}   t={t}  (alpha,beta,gamma)=({al},{be},{ga})")
    return (t, al, be, ga)

if __name__ == '__main__':
    import json
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    cases = json.load(open(os.path.join(here, 'vogelp-data.json')))
    for c in cases:
        roots = {int(k): v for k, v in c['roots'].items()}
        run(c['name'], c['A'], c['p'], roots, c['V1'], c['V2'], c['printed'], c.get('par'))
