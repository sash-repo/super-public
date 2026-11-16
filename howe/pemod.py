"""
pemod.py -- finite-dimensional irreducible pe(n)-modules V(lambda) = L_0-modules
of le(n), built weight space by weight space.

pe(n) has no contravariant anti-involution (pe_{+1} = Lambda^2 and pe_{-1} = S^2 are
non-isomorphic gl(n)-modules), so the Shapovalov form is unavailable.  Instead we use

    M_mu = { u in N(lambda)_mu :  e.u in M_{mu+alpha}  for every e in n_+ },   M_lambda = 0,

which is exactly the maximal submodule of the Verma module N(lambda) = U(n_-) (x) C_lambda
(a proper submodule cannot contain v_lambda, and the set of u with v_lambda not in
U(n_+)u is a subspace, hence the sum of all proper submodules).  Then

    V(lambda)_mu = N(lambda)_mu / M_mu.

Weight spaces are computed on demand, top down.
"""

from fractions import Fraction as F
from le import (pmul, padd, pscal, abr, par_le, wt, pwt, rref, nullspace)


def mon(n, a, S, c=1):
    return {(tuple(a), tuple(sorted(S))): F(c)}


def xxi(n, i, j):
    """generating function x_i xi_j"""
    a = [0] * n
    a[i] = 1
    return mon(n, a, [j])


def xx(n, i, j):
    a = [0] * n
    a[i] += 1
    a[j] += 1
    return mon(n, a, [])


def xixi(n, i, j):
    return mon(n, [0] * n, [i, j])


class Pe:
    """L_0 = pe(n) with the triangular decomposition n_- + h + n_+.

    Two Borels contain h + n_+^{gl}; they are not conjugate:
      odd_up=True   n_+ = n_+^{gl} + pe_{+1},  pe_{+1} = Lambda^2 = <D_{xi_i xi_j}>
      odd_up=False  n_+ = n_+^{gl} + pe_{-1},  pe_{-1} = S^2      = <D_{x_i x_j}>
    Kochetkov's list is stated for the second one.
    """

    def __init__(self, n, odd_up=True):
        self.n = n
        self.odd_up = odd_up
        self.h = [xxi(n, i, i) for i in range(n)]                     # e_i
        lam2 = [xixi(n, i, j) for i in range(n) for j in range(i + 1, n)]
        sym2 = [xx(n, i, j) for i in range(n) for j in range(i, n)]
        glpos = [xxi(n, i, j) for i in range(n) for j in range(n) if j < i]
        glneg = [xxi(n, i, j) for i in range(n) for j in range(n) if j > i]
        self.npos = glpos + (lam2 if odd_up else sym2)
        self.nneg = glneg + (sym2 if odd_up else lam2)
        self.nneg_wt = [pwt(f, n) for f in self.nneg]
        self.nneg_par = [par_le(f) for f in self.nneg]
        self.basis = self.nneg + self.h + self.npos
        self.bwt = [pwt(f, n) for f in self.basis]
        self.bpar = [par_le(f) for f in self.basis]

    def ht(self, nu):
        # c_1 > ... > c_n, and c_i + c_j > 0 (odd_up) resp. < 0 (not odd_up),
        # so that ht > 0 on every root of n_+.
        c = (lambda i: self.n + 1 - i) if self.odd_up else (lambda i: -(i + 1))
        return sum(c(i) * nu[i] for i in range(self.n))

    def as_vector(self, f):
        """express a degree-2 generating function in the basis -> dict idx->coef"""
        out = {}
        for m, c in f.items():
            for k, b in enumerate(self.basis):
                (bm, bc), = b.items()
                if bm == m:
                    out[k] = out.get(k, 0) + c / bc
                    break
            else:
                raise KeyError(m)
        return {k: v for k, v in out.items() if v}


class Verma:
    """N(lambda) = U(n_-) v_lambda, PBW words = nondecreasing tuples of n_- indices."""

    def __init__(self, pe, lam):
        self.pe = pe
        self.n = pe.n
        self.lam = tuple(F(x) for x in lam)
        self._basis = {}
        self._maxsub = {}

    # ---- PBW words of a given weight
    def basis(self, mu):
        mu = tuple(F(x) for x in mu)
        if mu in self._basis:
            return self._basis[mu]
        nu = tuple(a - b for a, b in zip(self.lam, mu))   # lam - mu, must be >= 0 in ht
        words = []
        k = len(self.pe.nneg)

        def rec(pos, rem, cur):
            if all(x == 0 for x in rem):
                words.append(tuple(cur))
                return
            if pos == k:
                return
            if self.pe.ht(rem) <= 0 and any(x != 0 for x in rem):
                return
            w = self.pe.nneg_wt[pos]
            odd = self.pe.nneg_par[pos] == 1
            maxrep = 1 if odd else 64
            r = 0
            while r <= maxrep:
                nrem = tuple(a + r * b for a, b in zip(rem, w))
                # -w is a positive root; rem must stay in the positive cone
                if r > 0 and self.pe.ht(nrem) < 0:
                    break
                rec(pos + 1, nrem, cur + [pos] * r)
                r += 1

        rec(0, nu, [])
        words = sorted(set(words))
        self._basis[mu] = words
        return words

    # ---- left multiplication by a n_- basis element
    def lmul(self, i, elt):
        out = {}
        for w, c in elt.items():
            for w2, c2 in self._lmul_word(i, w).items():
                out[w2] = out.get(w2, 0) + c * c2
        return {k: v for k, v in out.items() if v}

    def _lmul_word(self, i, w):
        if not w or i <= w[0]:
            if w and i == w[0] and self.pe.nneg_par[i] == 1:
                return {}
            return {(i,) + w: F(1)}
        j, rest = w[0], w[1:]
        sg = (-1) ** (self.pe.nneg_par[i] * self.pe.nneg_par[j])
        out = {}
        # (-1)^{p_i p_j} Y_j (Y_i rest)
        for w2, c2 in self._lmul_word(i, rest).items():
            for w3, c3 in self._lmul_word(j, w2).items():
                out[w3] = out.get(w3, 0) + sg * c2 * c3
        # [Y_i, Y_j] rest   (lands in n_-)
        br = self.pe.as_vector(abr(self.pe.nneg[i], self.pe.nneg[j], self.n))
        for k, ck in br.items():
            assert k < len(self.pe.nneg), "n_- not closed"
            for w3, c3 in self._lmul_word(k, rest).items():
                out[w3] = out.get(w3, 0) + ck * c3
        return {k: v for k, v in out.items() if v}

    # ---- action of a basis element of pe on N(lambda)
    def act_basis(self, b, elt):
        out = {}
        for w, c in elt.items():
            for w2, c2 in self._act_word(b, w).items():
                out[w2] = out.get(w2, 0) + c * c2
        return {k: v for k, v in out.items() if v}

    def _act_word(self, b, w):
        nneg, nn = self.pe.nneg, len(self.pe.nneg)
        if not w:
            if b < nn:
                return {(b,): F(1)}
            if b < nn + self.n:                      # h
                return {(): self.lam[b - nn]}
            return {}                                # n_+
        j, rest = w[0], w[1:]
        out = {}
        sg = (-1) ** (self.pe.bpar[b] * self.pe.nneg_par[j])
        # (-1)^{p_b p_j} Y_j (b . rest)
        for w2, c2 in self._act_word(b, rest).items():
            for w3, c3 in self._lmul_word(j, w2).items():
                out[w3] = out.get(w3, 0) + sg * c2 * c3
        # [b, Y_j] . rest
        br = self.pe.as_vector(abr(self.pe.basis[b], nneg[j], self.n))
        for k, ck in br.items():
            for w3, c3 in self._act_word(k, rest).items():
                out[w3] = out.get(w3, 0) + ck * c3
        return {k: v for k, v in out.items() if v}

    def act(self, vec, elt):
        out = {}
        for b, c in vec.items():
            for w, cw in self.act_basis(b, elt).items():
                out[w] = out.get(w, 0) + c * cw
        return {k: v for k, v in out.items() if v}

    # ---- maximal submodule, weight by weight
    def maxsub(self, mu):
        """returns (pivots, reduced rows) of a basis of M_mu in the PBW basis of N_mu"""
        mu = tuple(F(x) for x in mu)
        if mu in self._maxsub:
            return self._maxsub[mu]
        words = self.basis(mu)
        if not words:
            self._maxsub[mu] = ([], [])
            return self._maxsub[mu]
        if mu == self.lam:
            self._maxsub[mu] = ([], [])
            return self._maxsub[mu]
        nn = len(self.pe.nneg)
        # equations: coordinates of e.u modulo M_{mu+alpha}, for every e in n_+
        rows = []          # one row per (generator, target coordinate)
        cols = list(range(len(words)))
        eqs = []
        for ei in range(len(self.pe.npos)):
            b = nn + self.n + ei
            alpha = self.pe.bwt[b]
            nu = tuple(a + c for a, c in zip(mu, alpha))
            tw = self.basis(nu)
            if not tw:
                continue
            tpiv, tred = self.maxsub(nu)
            tindex = {w: i for i, w in enumerate(tw)}
            col = {}
            for ci, w in enumerate(words):
                img = self.act_basis(b, {w: F(1)})
                v = {tindex[k]: c for k, c in img.items()}
                # reduce modulo M_nu
                for p, r in zip(tpiv, tred):
                    if p in v:
                        c = v[p]
                        for k, val in r.items():
                            v[k] = v.get(k, 0) - c * val
                            if v[k] == 0:
                                del v[k]
                for k, val in v.items():
                    col.setdefault(k, {})[ci] = val
            eqs.extend(col.values())
        ker = nullspace(eqs, cols)
        piv, red = rref(ker, len(words))
        self._maxsub[mu] = (piv, red)
        return self._matnorm(mu, piv, red)

    def _matnorm(self, mu, piv, red):
        self._maxsub[mu] = (piv, red)
        return piv, red

    # ---- the irreducible quotient
    def quotient_basis(self, mu):
        """returns the list of PBW words that survive as a basis of V(lambda)_mu"""
        words = self.basis(mu)
        piv, red = self.maxsub(mu)
        pset = set(piv)
        return [i for i in range(len(words)) if i not in pset]

    def reduce(self, mu, v):
        """dict word->c  ->  dict surviving-index->c  (coordinates in V_mu)"""
        words = self.basis(mu)
        index = {w: i for i, w in enumerate(words)}
        vv = {}
        for w, c in v.items():
            vv[index[w]] = vv.get(index[w], 0) + c
        piv, red = self.maxsub(mu)
        for p, r in zip(piv, red):
            if p in vv:
                c = vv[p]
                for k, val in r.items():
                    vv[k] = vv.get(k, 0) - c * val
                    if vv[k] == 0:
                        del vv[k]
        return {k: v for k, v in vv.items() if v}


class Irrep:
    """V(lambda): weight spaces with a chosen basis of PBW representatives."""

    def __init__(self, pe, lam):
        self.pe = pe
        self.n = pe.n
        self.lam = tuple(F(x) for x in lam)
        self.V = Verma(pe, lam)
        self._bas = {}

    def wtbasis(self, mu):
        mu = tuple(F(x) for x in mu)
        if mu in self._bas:
            return self._bas[mu]
        nu = tuple(a - b for a, b in zip(self.lam, mu))
        if self.pe.ht(nu) < 0 or (any(x != 0 for x in nu) and self.pe.ht(nu) == 0):
            self._bas[mu] = []
            return []
        try:
            surv = self.V.quotient_basis(mu)
        except (KeyError, AssertionError):
            surv = []
        words = self.V.basis(mu)
        self._bas[mu] = [words[i] for i in surv]
        return self._bas[mu]

    def dim(self):
        """total dimension (only terminates when lambda is dominant)"""
        seen, tot = set(), 0
        stack = [self.lam]
        while stack:
            mu = stack.pop()
            if mu in seen:
                continue
            seen.add(mu)
            b = self.wtbasis(mu)
            tot += len(b)
            if not b:
                continue
            for i, w in enumerate(self.pe.nneg_wt):
                stack.append(tuple(a + c for a, c in zip(mu, w)))
        return tot

    def act(self, b, mu, coord):
        """act by basis element b of pe on a vector given in V_mu coordinates;
        returns coordinates in V_{mu+wt(b)}."""
        words = self.wtbasis(mu)
        elt = {}
        for i, c in coord.items():
            elt[words[i]] = elt.get(words[i], 0) + c
        img = self.V.act_basis(b, elt)
        nu = tuple(a + c for a, c in zip(mu, self.pe.bwt[b]))
        if not img:
            return {}
        red = self.V.reduce(nu, img)
        allw = self.V.basis(nu)
        surv = self.V.quotient_basis(nu)
        pos = {s: i for i, s in enumerate(surv)}
        out = {}
        for k, c in red.items():
            assert k in pos, "reduction left a non-surviving coordinate"
            out[pos[k]] = out.get(pos[k], 0) + c
        return {k: v for k, v in out.items() if v}
