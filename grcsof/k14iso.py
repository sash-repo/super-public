#!/usr/bin/env python3
"""Isomorphism type of the family h(a) subset k(1|4) found in k14family.py.

h(a) is 17-dimensional with even part sl(2)+sl(2)+sl(2) and odd part
2 (x) 2 (x) 2, i.e. it is a member of the family D(2,1;alpha) = osp_alpha(4|2).
We compute alpha as a function of a.

Method.  Fix the Cartan h = <H_1,H_2,H_3> (one coroot from each sl(2), each
normalised by beta_i(H_i) = 2 for the even root beta_i = 2eps_i).  For an odd
root vector X of weight delta = eps_1+eps_2+eps_3 and the opposite Y, one has
   [X,Y] = h_delta = sum_i c_i H_i.
Invariance of the (unique up to scalar) bilinear form gives
   (eps_i, eps_i) = c_i / const,
and for D(2,1;alpha) the standard normalisation is
   (eps_1,eps_1) : (eps_2,eps_2) : (eps_3,eps_3) = -(1+alpha) : 1 : alpha,
so that  c_1 + c_2 + c_3 = 0  (isotropy of the odd roots) and
alpha = c_3/c_2 up to the S_3-action  alpha -> 1/alpha, -1-alpha, ...
The S_3-invariant is  j(alpha) = (alpha^2+alpha+1)^3 / (alpha^2 (alpha+1)^2);
j = 27/4 <==> alpha in {1,-2,-1/2} <==> the algebra is osp(4|2).
"""
from itertools import combinations
from sympy import Symbol, symbols, expand, simplify, factor, Matrix, S, Rational, solve, together
from superN import Contact, add, smul, mul, mono, parity, clean
from k14family import ALG, TY, CUB, QUAD, TOP, msk

bra = ALG.kb
a = Symbol('a')

SIG = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]


def hgens(av):
    """basis of h(av): (name, superfunction, parity)."""
    W = [add(TY[i], *[smul(av * SIG[i][j], CUB[j]) for j in range(4)])
         for i in range(4)]
    E1 = add(mono(2, 0), smul(-2 * av, TOP))          # t^2 - 2a y1y2y3y4
    B = [('K1', mono(0, 0)), ('Kt', mono(1, 0)), ('Kt2', E1)]
    names4 = ALG.names
    for (i, j) in combinations(range(4), 2):
        B.append((f'K_{names4[i]}{names4[j]}', mono(0, msk(i, j))))
    for i in range(4):
        B.append((f'K_{names4[i]}', mono(0, 1 << i)))
    for i in range(4):
        B.append((f'w_{names4[i]}', W[i]))
    return B


def as_vec(u, B):
    """coordinates of the superfunction u in the basis B (monomial-wise)."""
    keys = set()
    for _, f in B:
        keys |= set(f)
    M = Matrix([[f.get(k, 0) for _, f in B] for k in sorted(keys)])
    rhs = Matrix([[u.get(k, 0)] for k in sorted(keys)])
    sol = M.solve_least_squares(rhs) if M.rows != M.cols else None
    # exact: use gauss
    aug = M.row_join(rhs)
    r = aug.rref()[0]
    v = [0] * len(B)
    piv = aug.rref()[1]
    for i, p in enumerate(piv):
        if p == len(B):
            raise ValueError("not in the span")
        v[p] = r[i, len(B)]
    return v


def report(av):
    B = hgens(av)
    names = [n for n, _ in B]
    idx = {n: i for i, n in enumerate(names)}
    # ---- Cartan: t, xi1 eta1, xi2 eta2
    H = {'h1': mono(1, 0), 'h2': mono(0, msk(0, 1)), 'h3': mono(0, msk(2, 3))}
    for x, y in combinations(H, 2):
        assert not clean({k: expand(v) for k, v in bra(H[x], H[y]).items()}), \
            f"{x},{y} do not commute"
    # ---- weights of the basis vectors
    wts = {}
    for nm, f in B:
        w = []
        ok = True
        for hn in ('h1', 'h2', 'h3'):
            r = clean({k: expand(v) for k, v in bra(H[hn], f).items()})
            if not r:
                w.append(S(0))
                continue
            # r must be proportional to f
            k0 = next(iter(f))
            c = simplify(r.get(k0, 0) / f[k0])
            chk = clean({k: expand(v - c * f.get(k, 0)) for k, v in r.items()})
            if chk:
                ok = False
                break
            w.append(simplify(c))
        wts[nm] = tuple(w) if ok else None
    print(f"a = {av}: weights (ad h1, ad h2, ad h3)")
    for nm in names:
        print(f"   {nm:12s} {wts[nm]}")
    return B, names, idx, H, wts


def alpha_of(av, verbose=False):
    B = hgens(av)
    names = [n for n, _ in B]
    H = [mono(1, 0), mono(0, msk(0, 1)), mono(0, msk(2, 3))]
    # weights
    def weight(f):
        w = []
        for h in H:
            r = clean({k: expand(v) for k, v in bra(h, f).items()})
            k0 = next(iter(f))
            c = S(0) if not r else simplify(r.get(k0, 0) / f[k0])
            w.append(simplify(c))
        return tuple(w)
    W = {nm: weight(f) for nm, f in B}
    # ---- the three sl(2) triples.
    # sl(2)_1: E = K1 (weight (-2,0,0) under ad t), F = Kt2, H_1 propto Kt
    # sl(2)_2, sl(2)_3 inside o(4): even root vectors are the K_{y_iy_j}
    #   with i,j in different pairs; H_2, H_3 are combinations of h2, h3.
    triples = []
    # sl(2)_1
    E, F = dict(B)['K1'], dict(B)['Kt2']
    Hc = clean({k: expand(v) for k, v in bra(E, F).items()})
    triples.append((E, F, Hc))
    # o(4): root vectors xi1 xi2 (weights under (h2,h3) = (1,1)) etc.
    names4 = ALG.names
    for (i, j), (k, l) in (((0, 2), (1, 3)), ((0, 3), (1, 2))):
        E = mono(0, msk(i, j))
        F = mono(0, msk(k, l))
        Hc = clean({m: expand(v) for m, v in bra(E, F).items()})
        triples.append((E, F, Hc))
    # normalise each triple: [H,E] = 2E
    normed = []
    for (E, F, Hc) in triples:
        r = clean({k: expand(v) for k, v in bra(Hc, E).items()})
        k0 = next(iter(E))
        c = simplify(r.get(k0, 0) / E[k0])
        normed.append((E, smul(simplify(2 / c), F), smul(simplify(2 / c), Hc)))
    Hs = [h for (_, _, h) in normed]
    if verbose:
        for i, (E, F, h) in enumerate(normed):
            print(f"  sl(2)_{i+1}: E = {ALG.show(E)},  F = {ALG.show(F)},  "
                  f"H = {ALG.show(h)}")
    # ---- an odd root vector and its opposite
    odd = [(nm, f) for nm, f in B if parity(f) == 1]
    # pick X = w_{xi1} (highest), Y = the odd vector of the opposite weight
    X = dict(B)['w_' + names4[0]]
    wX = weight(X)
    Y = None
    for nm, f in odd:
        if all(simplify(x + y) == 0 for x, y in zip(wX, weight(f))):
            Y = f
            break
    assert Y is not None, "no opposite odd root vector"
    hd = clean({k: expand(v) for k, v in bra(X, Y).items()})
    # expand hd in H_1, H_2, H_3
    keys = sorted(set().union(*[set(h) for h in Hs]) | set(hd))
    M = Matrix([[h.get(k, 0) for h in Hs] for k in keys])
    rhs = Matrix([[hd.get(k, 0)] for k in keys])
    sol = solve(list(M * Matrix(symbols('c1 c2 c3')) - rhs),
                symbols('c1 c2 c3'), dict=True)
    assert sol, "h_delta is not in the Cartan"
    c1, c2, c3 = [simplify(sol[0][s]) for s in symbols('c1 c2 c3')]
    return c1, c2, c3


if __name__ == '__main__':
    print("=== the standard subalgebra a = 0 (i(osp(4|2))) and the family ===")
    for av in (S(0), S(1), S(2), Rational(1, 3), a):
        c = alpha_of(av, verbose=(av == 0))
        s = simplify(sum(c))
        al = simplify(c[2] / c[1]) if c[1] != 0 else None
        j = None
        if al is not None and al not in (0, -1):
            j = factor(simplify((al**2 + al + 1)**3 / (al**2 * (al + 1)**2)))
        print(f"a = {av}: (c1,c2,c3) = {tuple(map(str,c))}, sum = {s}, "
              f"alpha = {al}, j = {j}")
