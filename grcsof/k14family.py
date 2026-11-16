#!/usr/bin/env python3
"""The graded subalgebras h of k(1|4) with h_- = k(1|4)_- and dim h = 17.

k(1|4) is Z-graded by deg t = 2, deg y_i = 1 (deg K_f = deg f - 2):
   g_{-2} = <K_1>,  g_{-1} = <K_{y_i}>,  g_0 = <K_t, K_{y_iy_j}> = C + o(4),
   g_1 = <K_{ty_i}, K_{y_iy_jy_k}>  (dim 4+4),
   g_2 = <K_{t^2}, K_{ty_iy_j}, K_{y_1y_2y_3y_4}>  (dim 1+6+1), ...

As o(4)-modules, g_1 = V + Lambda^3 V and V is irreducible with
Lambda^3 V isomorphic to V; hence the g_0-stable complements to Lambda^3 V in
g_1 form a one-parameter family: the graphs of  a.sigma,  where
sigma: V --> Lambda^3 V is THE equivariant map.  This is where the
one-parameter family osp_a(4|2) = D(2,1;a) inside k(1|4) comes from.

The script (1) computes sigma, (2) checks that h(a) is a subalgebra for every
a, (3) identifies the isomorphism type by the three sl(2)-triples.
"""
from itertools import combinations
from fractions import Fraction as Fr
from sympy import symbols, Symbol, expand, simplify, Matrix, factor, S, solve, linsolve
from superN import Contact, add, smul, mul, mono, parity, clean

ALG = Contact(4, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
              names=['xi1', 'eta1', 'xi2', 'eta2'])
n = 4


def msk(*idx):
    m = 0
    for i in idx:
        m |= 1 << i
    return m


LIN = [mono(0, 1 << i) for i in range(n)]
CUB = [mono(0, msk(*[j for j in range(n) if j != i])) for i in range(n)]
TY = [mono(1, 1 << i) for i in range(n)]
QUAD = [mono(0, msk(i, j)) for i, j in combinations(range(n), 2)]
TOP = mono(0, msk(0, 1, 2, 3))
bra = ALG.kb


def coords(u, basis):
    ks = [next(iter(b)) for b in basis]
    v = [u.get(k, 0) for k in ks]
    rest = {k: c for k, c in u.items() if k not in ks}
    assert not clean(rest), f"not in the span: {rest}"
    return v


def action_matrix(q, basis):
    """matrix of ad K_q on the given monomial basis: {q, b_i} = sum_j M_ij b_j."""
    return [coords(bra(q, b), basis) for b in basis]


def sigma():
    """the o(4)-equivariant map V --> Lambda^3 V, as a matrix M with
    sigma(t y_i) = sum_j M_ij CUB_j.  Found by intertwining ad K_{y_iy_j}."""
    M = [[Symbol(f'm{i}{j}') for j in range(n)] for i in range(n)]
    eqs = []
    for q in QUAD:
        A = action_matrix(q, TY)         # action on the "t y_i" basis
        Bm = action_matrix(q, CUB)       # action on the cubics
        # requirement:  M . B = A . M   (both as matrices acting on the right)
        MB = Matrix(M) * Matrix(Bm)
        AM = Matrix(A) * Matrix(M)
        for e in (MB - AM):
            eqs.append(expand(e))
    unk = [M[i][j] for i in range(n) for j in range(n)]
    sol = solve([e for e in eqs if e != 0], unk, dict=True)
    return M, sol


def build(a):
    """h(a): the generators of h_1 and the resulting h_2."""
    Ms, sol = sigma()
    s = sol[0]
    free = sorted({str(v) for e in s.values() for v in e.free_symbols} |
                  {str(k) for k in unknowns_not_in(s)}, key=str)
    return Ms, s, free


def unknowns_not_in(s):
    unk = {Symbol(f'm{i}{j}') for i in range(n) for j in range(n)}
    return unk - set(s.keys())


if __name__ == '__main__':
    M, sol = sigma()
    print("intertwiners V -> Lambda^3 V:")
    for s in sol:
        print("  ", s)
        free = unknowns_not_in(s)
        print("   free parameters:", sorted(map(str, free)))
        # normalise: set the free parameter to 1
        assert len(free) == 1, "the intertwiner is not unique up to scalar!"
        p = free.pop()
        subs = {k: v.subs({p: 1}) for k, v in s.items()}
        subs[p] = S(1)
        Mat = Matrix([[subs.get(M[i][j], M[i][j]) for j in range(n)]
                      for i in range(n)])
        print("   sigma =\n", Mat)
        # the family
        a = Symbol('a')
        W = [add(TY[i], *[smul(a * Mat[i, j], CUB[j]) for j in range(n)])
             for i in range(n)]
        print("   h_1(a) generators:")
        for i, w in enumerate(W):
            print("     w%d = %s" % (i, ALG.show(w)))
        # h_2 = [h_1,h_1]
        print("   brackets {w_i,w_j}:")
        B2 = []
        for i in range(n):
            for j in range(i, n):
                b = bra(W[i], W[j])
                b = {k: expand(v) for k, v in b.items()}
                b = clean(b)
                if b:
                    print(f"     {{w{i},w{j}}} = {ALG.show(b)}")
                    B2.append(b)
        # h_3 must vanish: {w_i, {w_j,w_k}}
        print("   {w_i, h_2} (must be 0 for dim h = 17):")
        bad = 0
        for b in B2:
            for i in range(n):
                r = clean({k: expand(v) for k, v in bra(W[i], b).items()})
                if r:
                    bad += 1
                    print(f"     nonzero: {ALG.show(r)}")
        print("   ok" if not bad else f"   {bad} nonzero brackets")
