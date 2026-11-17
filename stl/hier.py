"""The sl(lambda) Gelfand-Dickey ("KdV") hierarchy, built exactly.

L = D^lam + sum_{k>=2} u_k D^{lam-k}   (u_k on the left; u_1 = 0 is the sl-condition)
Q = L^{1/lam} = D + sum_{j>=1} q_j D^{-j}
d L / d t_m = [ (Q^m)_+ , L ].

Q^lam is defined as W D^lam W^{-1} (Sato dressing, Q = W D W^{-1}); its left coefficients are
polynomials in lam, so they are obtained by interpolating the integer powers Q^n.

Statement 10.4.2: run  python3 hier.py 7  -- it prints q_1, q_2, q_3 and the flows du_k/dt_m
(m = 2, 3; k = 2, 3, 4), of which du_2/dt_2, du_3/dt_2, du_2/dt_3 are the system (274).
"""
import sys
import sympy as sp
from psdo import Sym, comm, jet, Dx, Dxn, lam, binom

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
n_ = sp.Symbol('n')


def build(N):
    Q = Sym(1, 0, {0: 1, **{j + 1: jet('q', j, 0) for j in range(1, N)}}, N)
    pw = [Sym(0, 0, {0: 1}, N)]
    for n in range(1, N + 3):
        pw.append(pw[-1] * Q)
    # coefficient of D^{n-k} in Q^n, interpolated in n
    ccoef = {}
    for k in range(0, N + 1):
        pts = [(n, pw[n].c.get(k, 0)) for n in range(0, N + 3)]
        # Lagrange through the first k+1 points, check on the rest
        poly = sp.expand(sp.interpolate(pts[:k + 2], n_)) if k + 2 <= len(pts) else None
        for n, v in pts:
            assert sp.expand(poly.subs(n_, n) - v) == 0, ('not polynomial of degree<=k+1', k)
        ccoef[k] = sp.expand(poly.subs(n_, lam))
    return Q, ccoef


def invert(ccoef, N):
    """q_j as differential polynomials in u, from u_k = ccoef[k](q)"""
    assert ccoef[1] == 0
    qu = {}
    for k in range(2, N + 1):
        j = k - 1
        expr = ccoef[k]
        # u_k = lam*q_j + rest(q_{<j})
        lead = sp.expand(expr.coeff(jet('q', j, 0)))
        assert sp.simplify(lead - lam) == 0, (k, lead)
        rest = sp.expand(expr - lam * jet('q', j, 0))
        sub = {}
        for s in rest.free_symbols:
            if s.name.startswith('q'):
                jj, i = map(int, s.name[1:].split('_'))
                assert jj < j
                sub[s] = Dxn(qu[jj], i)
        rest = sp.expand(rest.xreplace(sub))
        qu[j] = sp.expand((jet('u', k, 0) - rest) / lam)
    return qu


def qsubs(qu):
    def f(e):
        sub = {}
        for s in sp.sympify(e).free_symbols:
            if s.name.startswith('q'):
                jj, i = map(int, s.name[1:].split('_'))
                sub[s] = Dxn(qu[jj], i)
        return sp.expand(sp.sympify(e).xreplace(sub))
    return f


def flows(N, mmax):
    Q, cc = build(N)
    qu = invert(cc, N)
    f = qsubs(qu)
    Qu = Sym(1, 0, {k: f(v) for k, v in Q.c.items()}, N)
    L = Sym(0, 1, {0: 1, **{k: jet('u', k, 0) for k in range(2, N + 1)}}, N)
    fl = {}
    P = Sym(0, 0, {0: 1}, N)
    for m in range(1, mmax + 1):
        P = P * Qu
        B = P.plus()
        C = comm(B, L)          # order lam + m - 1 base; read coefficient of D^{lam-k}
        # C has base a = m, b = 1: C.c[k'] is coeff of D^{m+lam-k'}; D^{lam-k} <-> k' = m+k
        fl[m] = {k: sp.expand(C.c.get(m + k, 0)) for k in range(1, N - m + 1)}
        assert all(sp.expand(C.c.get(kk, 0)) == 0 for kk in range(0, m + 1)), 'order too high'
    return cc, qu, fl


if __name__ == '__main__':
    cc, qu, fl = flows(N, 4)
    for j in range(1, 4):
        print(f'q{j} =', sp.factor_terms(qu[j]))
    for m in (2, 3):
        for k in range(2, 5):
            print(f'd u{k}/dt{m} =', sp.collect(fl[m][k], lam))
