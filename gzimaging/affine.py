"""The k_1-conditions as an affine matrix pencil in (lam1,lam2).

L_f = A_f + lam (f_t .), with A_f free of the weight, so for fixed weighted order W
(whence lam3 = lam1+lam2+(W+k1+k2-k3)/2) the condition matrix is

    N(lam1,lam2) = N_0 + lam1 N_1 + lam2 N_2,

affine in both.  Building it once turns the sweep in the weights into linear algebra on
a matrix with at most dim V_0 columns, and lets the exceptional locus be found as the
common zero set of the maximal minors instead of being guessed from a grid.
"""

import os, sys
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE]
from exactlin import nullspace, rank, rref  # noqa: E402

from contact import Module, madd, mscal, pmono  # noqa: E402
from gen import defect, target_weight, rand_field, K0, K1, K2, K3  # noqa: E402
from fast import sp2_invariants, _compose  # noqa: E402

SAMPLES = [(Fr(0), Fr(0)), (Fr(1), Fr(0)), (Fr(0), Fr(1)), (Fr(1), Fr(1))]


def pencil(k1, k2, k3, W, gens=None, deg=None, nfield=2, seed=0):
    """Return (terms, V0, N0, N1, N2) with the rows compressed."""
    terms, V0 = sp2_invariants(k1, k2, k3, W)
    if not V0:
        return terms, V0, [], [], []
    if deg is None:
        deg = W + 1
    if gens is None:
        gens = [pmono(0, 3, 0), pmono(1, 1, 0)]
    F_list = [rand_field(k1, deg, seed=seed + 11 + i) for i in range(nfield)]
    G_list = [rand_field(k2, deg, seed=seed + 71 + i) for i in range(nfield)]
    N0, N1, N2 = [], [], []
    for f in gens:
        for F in F_list:
            for G in G_list:
                # each TERM's defect once per weight sample, then the change of basis
                # to V_0 as a matrix product -- not once per basis vector of V_0
                blocks = []
                keys = set()
                for (a, b) in SAMPLES:
                    l3 = target_weight(k1, a, k2, b, k3, W)
                    M1, M2, M3 = Module(k1, a), Module(k2, b), Module(k3, l3)
                    LfF, LfG = M1.act(f, F), M2.act(f, G)
                    cols = [defect(term, f, F, G, M1, M2, M3, LfF, LfG)
                            for term in terms]
                    for acc in cols:
                        keys |= set(acc)
                    blocks.append(cols)
                if not keys:
                    continue
                keys = sorted(keys)
                V0s = [[(i, c) for i, c in enumerate(vec) if c] for vec in V0]
                mats = []
                for cols in blocks:
                    mat_k = []
                    for k in keys:
                        row = [c.get(k) for c in cols]
                        mat_k.append([sum((row[i] * c for i, c in vs
                                           if row[i] is not None), Fr(0))
                                      for vs in V0s])
                    mats.append(mat_k)
                B00, B10, B01, B11 = mats
                A0 = B00
                A1 = [[x - y for x, y in zip(r1, r0)] for r1, r0 in zip(B10, B00)]
                A2 = [[x - y for x, y in zip(r1, r0)] for r1, r0 in zip(B01, B00)]
                # affineness is a claim about the module action -- check it, do not assume
                for r, (a0, a1, a2, b) in enumerate(zip(A0, A1, A2, B11)):
                    for c in range(len(V0)):
                        assert a0[c] + a1[c] + a2[c] == b[c], "N is not affine in the weights"
                N0.extend(A0)
                N1.extend(A1)
                N2.extend(A2)
    # compress: one row operation on the stacked matrix acts on all three blocks
    stacked = [r0 + r1 + r2 for r0, r1, r2 in zip(N0, N1, N2)]
    red = [r for r in rref(stacked)[0] if any(r)]
    n = len(V0)
    N0 = [r[:n] for r in red]
    N1 = [r[n:2 * n] for r in red]
    N2 = [r[2 * n:] for r in red]
    return terms, V0, N0, N1, N2


def at(N0, N1, N2, lam, mu):
    lam, mu = Fr(lam), Fr(mu)
    return [[a + lam * b + mu * c for a, b, c in zip(r0, r1, r2)]
            for r0, r1, r2 in zip(N0, N1, N2)]


def dim_at(V0, N0, N1, N2, lam, mu):
    if not V0:
        return 0
    if not N0:
        return len(V0)
    M = at(N0, N1, N2, lam, mu)
    return len(V0) - rank(M)


def ops_at(terms, V0, N0, N1, N2, lam, mu):
    if not V0:
        return []
    if not N0:
        return _compose([[Fr(1) if i == j else Fr(0) for i in range(len(V0))]
                         for j in range(len(V0))], V0)
    return _compose(nullspace(at(N0, N1, N2, lam, mu)), V0)


def _minors(V0, N0, N1, N2, r, ntrials, seed):
    import random
    import sympy as sp
    lam, mu = sp.symbols("lam mu")
    Msym = sp.Matrix([[sp.Rational(a) + lam * sp.Rational(b) + mu * sp.Rational(c)
                       for a, b, c in zip(r0, r1, r2)]
                      for r0, r1, r2 in zip(N0, N1, N2)])
    rnd = random.Random(seed)
    rows, cols = list(range(Msym.rows)), list(range(Msym.cols))
    out = []
    for _ in range(ntrials):
        if len(rows) < r or len(cols) < r:
            break
        d = sp.expand(Msym[rnd.sample(rows, r), rnd.sample(cols, r)].det())
        if d != 0:
            out.append(d)
    return out


def exceptional_locus(V0, N0, N1, N2, ntrials=12, seed=3):
    """The jump locus, exactly: the variety of the ideal of maximal minors.

    Returns (generic rank, description).  The ideal is generated by ALL maximal minors;
    a random sample of them generates an ideal whose variety CONTAINS the jump locus,
    so an empty variety is a proof and a non-empty one is a candidate list -- each
    candidate is then evaluated exactly by dim_at.
    """
    import sympy as sp
    if not V0 or not N0:
        return None, []
    r = rank(at(N0, N1, N2, Fr(17, 13), Fr(31, 11)))
    if r == 0:
        return r, ["everything"]
    mins = _minors(V0, N0, N1, N2, r, ntrials, seed)
    if not mins:
        return r, ["everything"]
    lam, mu = sp.symbols("lam mu")
    G = sp.groebner(mins, mu, lam, order="lex")
    if list(G.exprs) == [sp.Integer(1)]:
        return r, []
    # elimination: the last generator of a lex GB involves lam alone when the ideal
    # is 0-dimensional; otherwise there is a curve component
    elim = [g for g in G.exprs if mu not in g.free_symbols]
    if not elim:
        return r, ["curve: " + str(sp.factor_list(sp.gcd_list(mins)))]
    out = []
    for lval in sp.roots(sp.Poly(elim[0], lam)).keys() if elim[0].free_symbols else []:
        sub = [sp.expand(m.subs(lam, lval)) for m in mins]
        sub = [s for s in sub if s != 0]
        if not sub:
            out.append((lval, "any"))
            continue
        for mval in sp.roots(sp.Poly(sp.gcd_list(sub), mu)).keys():
            out.append((lval, mval))
    return r, out
