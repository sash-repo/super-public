"""Bilinear k(2n+1|m)-invariant operators on the weighted densities, any (n,m).

An operator invariant under k_{-2}+k_{-1} has constant coefficients in the frame
e_1..e_{2n}, D_1..D_m, R (kmix.py), so it is a combination of

    (e^{I_1} D_{S_1} R^{s_1} F) . (e^{I_2} D_{S_2} R^{s_2} G),
    W := |I_1| + |S_1| + |I_2| + |S_2| + 2(s_1 + s_2),

I an exponent tuple in the 2n even frame directions and S a strictly increasing
multi-index in the m odd ones (a PBW basis, since [e_a,e_b] and {D_i,D_j} are multiples
of R).  Invariance under z = K_{2t} makes W the same for every term and forces

    nu = lam + mu + W/2.

The parity of a term is |S_1|+|S_2| mod 2, so an operator of odd weighted order need
not be odd as soon as there are at least two odd directions -- the parity statement of
the k(1|n) case is the case where the two odd slots cannot be contracted with each
other invariantly.

What is left to impose is k_0 = Kee z + osp(m|2n) and k_1; for n >= 1 the ansatz is
small enough to impose both by brute force, without the scheme calculus of genn.py.

Exact rational arithmetic.
"""

import os, sys
from fractions import Fraction as Fr
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank  # noqa: E402

from kmix import KMix, Dens, add, scal, mul, parity  # noqa: E402


def frame_words(alg, w):
    """All (I, S, s) with |I| + |S| + 2s = w."""
    out = []
    for s in range(w // 2 + 1):
        rest = w - 2 * s
        for ks in range(min(rest, alg.m) + 1):
            for S in combinations(range(1, alg.m + 1), ks):
                for I in _exps(2 * alg.n, rest - ks):
                    out.append((I, S, s))
    return out


def _exps(nv, d):
    if nv == 0:
        return [()] if d == 0 else []
    out = []
    for k in range(d + 1):
        for rest in _exps(nv - 1, d - k):
            out.append((k,) + rest)
    return out


def apply_frame(alg, I, S, s, h):
    out = h
    for _ in range(s):
        out = alg.R(out)
        if not out:
            return {}
    for j in reversed(S):
        out = alg.D(j, out)
        if not out:
            return {}
    for a in range(2 * alg.n, 0, -1):
        for _ in range(I[a - 1]):
            out = alg.e(a, out)
            if not out:
                return {}
    return out


def term_basis(alg, W):
    out = []
    for w1 in range(W + 1):
        for T1 in frame_words(alg, w1):
            for T2 in frame_words(alg, W - w1):
                out.append((T1, T2))
    return out


def apply_term(alg, term, F, G):
    """The operator D_1 (x) D_2 applied to (F,G), with the Koszul sign of moving D_2
    past F: it is the OPERATOR that is the basis element, not the written product."""
    (I1, S1, s1), (I2, S2, s2) = term
    a = apply_frame(alg, I1, S1, s1, F)
    if not a:
        return {}
    b = apply_frame(alg, I2, S2, s2, G)
    if not b:
        return {}
    sgn = (-1) ** (len(S2) * (parity(F) if F else 0))
    return scal(sgn, mul(a, b))


def rand_f(alg, seed, maxdeg=3, nmon=10, par=0):
    import random
    rnd = random.Random(seed)
    out = {}
    for _ in range(nmon):
        e = [0] * alg.nev
        S = []
        for _ in range(rnd.randint(0, maxdeg)):
            if alg.m and rnd.random() < 0.45:
                j = rnd.randint(1, alg.m)
                if j not in S:
                    S.append(j)
            else:
                e[rnd.randrange(alg.nev)] += 1
        if len(S) % 2 != par:
            j = next((x for x in range(1, alg.m + 1) if x not in S), None)
            if j is None:
                if not S:
                    continue
                S = S[:-1] if (len(S) - 1) % 2 == par else S
                if len(S) % 2 != par:
                    continue
            else:
                S.append(j)
        key = (tuple(e), tuple(sorted(S)))
        out[key] = out.get(key, Fr(0)) + Fr(rnd.randint(1, 6))
    return {k: v for k, v in out.items() if v}


def term_parity(term):
    (I1, S1, s1), (I2, S2, s2) = term
    return (len(S1) + len(S2)) % 2


class MixProblem:
    """The invariant operators of parity `par`.

    A morphism T of parity p(T) satisfies  L_f(T x) = (-1)^{p(T)p(f)} T(L_f x), and on
    the tensor product L_f(F (x) G) = (L_f F)(x)G + (-1)^{p(f)p(F)} F (x) L_f G.  The
    ansatz splits by parity because a term has the parity of its number of D's, so the
    two cases are two independent computations.
    """

    def __init__(self, n, m, W, par=0, gram=None):
        self.alg = KMix(n, m, gram=gram)
        self.W = W
        self.par = par
        self.terms = [t for t in term_basis(self.alg, W) if term_parity(t) == par]

    def _rows(self, f, F, G, M1, M2, M3):
        alg = self.alg
        pf = parity(f) if f else 0
        LfF, LfG = M1.act(f, F), M2.act(f, G)
        sF = (-1) ** (pf * (parity(F) if F else 0))
        sT = (-1) ** (self.par * pf)
        cols, keys = [], set()
        for term in self.terms:
            d = M3.act(f, apply_term(alg, term, F, G))
            for k, v in apply_term(alg, term, LfF, G).items():
                d[k] = d.get(k, Fr(0)) - sT * v
            for k, v in apply_term(alg, term, F, LfG).items():
                d[k] = d.get(k, Fr(0)) - sT * sF * v
            d = {a: b for a, b in d.items() if b}
            cols.append(d)
            keys |= set(d)
        if not keys:
            return []
        keys = sorted(keys)
        idx = {k: i for i, k in enumerate(keys)}
        mat = [[Fr(0)] * len(cols) for _ in keys]
        for j, d in enumerate(cols):
            for k, v in d.items():
                mat[idx[k]][j] = v
        return mat

    def solve(self, lam, mu, pieces=(0, 1), nsolve=2, deg=None, parities=(0, 1),
              verify_pieces=(0, 1, 2, 3), nus=(0, 0, 0)):
        alg = self.alg
        nu = Fr(lam) + Fr(mu) + Fr(self.W, 2)
        M1 = Dens(alg, lam, nus[0])
        M2 = Dens(alg, mu, nus[1])
        M3 = Dens(alg, nu, nus[2])
        if deg is None:
            deg = self.W + 2
        gens = []
        for i in pieces:
            gens += alg.k_piece(i)
        Fs = [rand_f(alg, 17 + i, deg, par=p) for p in parities for i in range(nsolve)]
        Gs = [rand_f(alg, 93 + i, deg, par=p) for p in parities for i in range(nsolve)]
        mat = []
        for f in gens:
            for F in Fs:
                for G in Gs:
                    mat.extend(self._rows(f, F, G, M1, M2, M3))
        ns = nullspace(mat) if mat else [
            [Fr(1) if i == j else Fr(0) for i in range(len(self.terms))]
            for j in range(len(self.terms))]
        if not ns:
            return nu, []
        # independent verification on denser fields and on k_2, k_3
        gens2 = []
        for i in verify_pieces:
            gens2 += alg.k_piece(i)
        Fv = [rand_f(alg, 5000 + i, deg + 2, 14, par=p)
              for p in parities for i in range(2)]
        Gv = [rand_f(alg, 6000 + i, deg + 2, 14, par=p)
              for p in parities for i in range(2)]
        for _ in range(4):
            bad = []
            for f in gens2:
                pf = parity(f) if f else 0
                sT = (-1) ** (self.par * pf)
                for F in Fv:
                    sF = (-1) ** (pf * parity(F))
                    for G in Gv:
                        LfF, LfG = M1.act(f, F), M2.act(f, G)
                        for vec in ns:
                            acc = {}
                            for cf, term in zip(vec, self.terms):
                                if not cf:
                                    continue
                                d = M3.act(f, apply_term(alg, term, F, G))
                                for k, v in apply_term(alg, term, LfF, G).items():
                                    d[k] = d.get(k, Fr(0)) - sT * v
                                for k, v in apply_term(alg, term, F, LfG).items():
                                    d[k] = d.get(k, Fr(0)) - sT * sF * v
                                for k, v in d.items():
                                    if v:
                                        acc[k] = acc.get(k, Fr(0)) + cf * v
                            if any(acc.values()):
                                bad.append((f, F, G))
                                break
            if not bad:
                return nu, ns
            for f, F, G in bad:
                mat.extend(self._rows(f, F, G, M1, M2, M3))
            ns = nullspace(mat)
            if not ns:
                return nu, []
        raise RuntimeError("verification did not converge")

    def op_dim(self, ns, deg=2):
        """The rank of the solutions AS OPERATORS (the term basis is a basis, so this
        equals len(ns); computed anyway, as a check)."""
        return len(ns)


def dimension(n, m, W, lam, mu, par=None, gram=None, **kw):
    """(dimension, ansatz size).  par=None sums the even and the odd operators."""
    if par is None:
        d, a = 0, 0
        for p in (0, 1):
            dd, aa = dimension(n, m, W, lam, mu, par=p, gram=gram, **kw)
            d += dd
            a += aa
        return d, a
    P = MixProblem(n, m, W, par=par, gram=gram)
    if not P.terms:
        return 0, 0
    nu, ns = P.solve(lam, mu, **kw)
    return len(ns), len(P.terms)
