r"""Theorem 3.3.5: the derived series of the closure of {E_i, E_j, F_i, F_j, d}
(two odd J's and an even pseudo-Hermitian h) on the periplectic C^{2|2} and C^{4|4}:

    g = g^{(0)} > g^{(1)} > g^{(2)} > ... ,   g^{(k+1)} = [g^{(k)}, g^{(k)}],

which shows that the algebra is solvable.

Run (from this folder):  ../venv/bin/python derived.py
"""
import sys

from flatmodel import Model, closure, d_operator          # noqa: E402
from conj2 import sd, reduce_basis                        # noqa: E402

sys.setrecursionlimit(10000)


def periplectic_model(n):
    return Model([0] * n + [1] * n)


def derived_series(M, basis):
    """[(sdim of g^{(k)})], stopping at 0 or at a fixed point."""
    cur = reduce_basis(M, basis)
    out = [sd(M, cur)]
    while out[-1] != (0, 0):
        nxt = reduce_basis(M, [M.br(u, v) for u in cur for v in cur])
        s = sd(M, nxt)
        out.append(s)
        if s == out[-2]:
            break
        cur = nxt
        if len(out) > 12:
            break
    return out


def sost(n):
    """E_i, E_j, F_i, F_j, d of Subsec. 3.3.4 on C^{n|n}, n even (Darboux pairs)."""
    M = periplectic_model(n)
    c = lambda a: M.gen('c', a)                            # noqa: E731
    b = lambda a: M.gen('b', a)                            # noqa: E731
    d = d_operator(M)
    pairs = [(2 * k, 2 * k + 1) for k in range(n // 2)]    # (q_1,q_2), ...
    Ei = M.add(*[M.mul(c(i), c(n + i)) for i in range(n)])
    Ej = M.add(*[t for (u, v) in pairs
                 for t in (M.mul(c(u), c(n + v)),
                           M.smul(-1, M.mul(c(v), c(n + u))))])
    Fi = M.add(*[M.mul(b(n + i), b(i)) for i in range(n)])
    Fj = M.add(*[t for (u, v) in pairs
                 for t in (M.mul(b(n + v), b(u)),
                           M.smul(-1, M.mul(b(n + u), b(v))))])
    return M, [Ei, Ej, Fi, Fj, d]


for n in (2, 4):
    M, gens = sost(n)
    g, s = closure(M, gens, limit=400)
    ser = derived_series(M, g)
    print('C^{%d|%d}: sdim <E_i,E_j,F_i,F_j,d> = %s' % (n, n, (s,)))
    print('   derived series  %s'
          % ' > '.join('(%d|%d)' % t for t in ser))
    print('   solvable        %s ; derived length %d'
          % (ser[-1] == (0, 0), len(ser) - 1))
