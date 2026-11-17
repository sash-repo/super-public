"""Adler-Gelfand-Dickey map at complex order (Khesin-Zakharevich / Figueroa-O'Farrill-Mas-Ramos):
J(X) = (L X)_+ L - L (X L)_+,  X = D^{-lam+j-1} o f  is dual to u_j  (Res(dL o X) = f du_j).
P_{ij}(f) := coefficient of D^{lam-i} in J(X_j(f)), on the gl-space L = D^lam + sum_{k>=1} u_k D^{lam-k}.

Statement 10.6.2: run  python3 adler.py  -- it prints P_11, P_12, P_21, P_22 as operators
{order: coefficient}; at u_1 = 0 they are -lam d, lam(lam-1)/2 d^2, -lam(lam-1)/2 d^2,
lam(lam-1)(2lam-1)/6 d^3 + 2 u_2 d + u_2'.  The Dirac reduction is item 7 of checks.py.
"""
import sympy as sp
from psdo import Sym, jet, Dx, lam
N = 6
L = Sym(0, 1, {0: 1, **{k: jet('u', k, 0) for k in range(1, N + 1)}}, N)
f = jet('v', 0, 0)

def P(i, j):
    X = Sym(j - 1, -1, {0: 1}, N) * Sym(0, 0, {0: f}, N)
    J = (L * X).plus() * L - L * (X * L).plus()
    # J has base (j-1, 1): c[k] is coeff of D^{lam + j-1-k}; D^{lam-i} <-> k = j-1+i
    return sp.expand(J.c.get(j - 1 + i, 0))

def asop(e):
    """write a linear expression in v-jets as an operator sum a_i d^i"""
    e = sp.expand(e)
    return {i: sp.factor(e.coeff(jet('v', 0, i))) for i in range(0, 8) if e.coeff(jet('v', 0, i)) != 0}

if __name__ == '__main__':
    for (i, j) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        print((i, j), asop(P(i, j)))
