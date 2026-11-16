"""Chevalley--Eilenberg cochain complex of a Lie superalgebra with trivial coefficients.

Model: C^* = S^*(Pi g^*), the free supercommutative superalgebra on the generators
c^a dual to a homogeneous basis x_a of g, with the parity SHIFTED,
p(c^a) = p(x_a) + 1.  So c^a is exterior when x_a is even and polynomial when x_a
is odd; the cohomological degree is the number of c's.

The differential is the odd derivation d with

    d c^a = (1/2) sum_{b,c} eps * (-1)^{p_b} f^a_{bc} c^b c^c,   [x_b, x_c] = f^a_{bc} x_a,

extended by the odd Leibniz rule.  The overall convention eps is FIXED BY TESTING
d^2 = 0 on the generators -- which, d being an odd derivation, is equivalent to
d^2 = 0 on the whole complex.  Nothing is assumed: check_d2() must pass.

Both the internal Z-grading of g and its torus weight are carried along; d
preserves both, so the complex splits into finite blocks C^q_{k,w}.

Over a field of characteristic p the polynomial generators would have to be
replaced by divided powers; as long as q < p the two agree, and every table we
print is computed in that range (and cross-checked against Q).
"""

import sys
from fractions import Fraction
from itertools import combinations_with_replacement, combinations

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))
from exactlin import rank as exact_rank  # noqa: E402


class CE:
    def __init__(self, g, eps=1):
        self.g = g
        self.eps = eps
        d = g.dim
        # parity of the generator c^a
        self.cpar = [(g.par(a) + 1) % 2 for a in range(d)]
        self.cdeg = [-g.deg(a) for a in range(d)]
        self.cwt = [tuple(-t for t in g.weight(a)) for a in range(d)]
        self.dim = d
        self._dgen = None

    # ------------------------------------------------------------ monomials
    def normalize(self, word):
        """Sort a word of generator indices; return (sign, tuple) or (0, None)."""
        w = list(word)
        sign = 1
        for i in range(1, len(w)):
            j = i
            while j > 0 and w[j - 1] > w[j]:
                if self.cpar[w[j - 1]] and self.cpar[w[j]]:
                    sign = -sign
                w[j - 1], w[j] = w[j], w[j - 1]
                j -= 1
        for i in range(1, len(w)):
            if w[i] == w[i - 1] and self.cpar[w[i]]:
                return 0, None
        return sign, tuple(w)

    def mdeg(self, mon):
        return sum(self.cdeg[a] for a in mon)

    def mwt(self, mon):
        r = len(self.cwt[0]) if self.cwt else 0
        w = [0] * r
        for a in mon:
            for i, t in enumerate(self.cwt[a]):
                w[i] += t
        return tuple(w)

    def mpar(self, mon):
        return sum(self.cpar[a] for a in mon) % 2

    # ------------------------------------------------------------ basis
    def basis(self, q):
        """All monomials of cohomological degree q, as sorted tuples."""
        ex = [a for a in range(self.dim) if self.cpar[a]]      # exterior gens
        po = [a for a in range(self.dim) if not self.cpar[a]]  # polynomial gens
        out = []
        for j in range(min(q, len(ex)) + 1):
            for E in combinations(ex, j):
                for M in combinations_with_replacement(po, q - j):
                    s, mon = self.normalize(E + M)
                    if s:
                        out.append(mon)
        return out

    # ------------------------------------------------------------ one block
    def basis_block(self, q, k, w):
        """Monomials of cohomological degree q, internal degree k, torus weight w.

        Enumerated directly, with pruning on the still reachable (degree, weight)
        -- the whole basis of C^q is far too large to sieve for q >= 6.
        """
        d = self.dim
        r = len(w)
        # suffix bounds: over the generators a, a+1, ..., d-1
        lo_deg = [0] * (d + 1)
        hi_deg = [0] * (d + 1)
        lo_w = [[0] * r for _ in range(d + 1)]
        hi_w = [[0] * r for _ in range(d + 1)]
        for a in range(d - 1, -1, -1):
            lo_deg[a] = min(lo_deg[a + 1], self.cdeg[a])
            hi_deg[a] = max(hi_deg[a + 1], self.cdeg[a])
            for i in range(r):
                lo_w[a][i] = min(lo_w[a + 1][i], self.cwt[a][i])
                hi_w[a][i] = max(hi_w[a + 1][i], self.cwt[a][i])
        out = []

        def rec(a, left, dk, dw, acc):
            if left == 0:
                if dk == 0 and all(x == 0 for x in dw):
                    out.append(tuple(acc))
                return
            if a >= d:
                return
            if not (lo_deg[a] * left <= dk <= hi_deg[a] * left):
                return
            for i in range(r):
                if not (lo_w[a][i] * left <= dw[i] <= hi_w[a][i] * left):
                    return
            top = 1 if self.cpar[a] else left
            for m in range(top + 1):
                if m:
                    acc.append(a)
                    dk -= self.cdeg[a]
                    for i in range(r):
                        dw[i] -= self.cwt[a][i]
                rec(a + 1, left - m, dk, dw, acc)
            for m in range(top, 0, -1):
                acc.pop()
                dk += self.cdeg[a]
                for i in range(r):
                    dw[i] += self.cwt[a][i]

        rec(0, q, k, list(w), [])
        return out

    # ------------------------------------------------------------ d
    def dgen(self):
        """d c^a as a dict {monomial: coefficient}, for every generator a."""
        if self._dgen is not None:
            return self._dgen
        g = self.g
        tab = g.table()
        out = [dict() for _ in range(self.dim)]
        for b in range(self.dim):
            sb = (-1) ** g.par(b)
            for c in range(self.dim):
                br = tab[b][c]
                if not br:
                    continue
                s, mon = self.normalize((b, c))
                if not s:
                    continue
                for a, v in br.items():
                    coef = Fraction(self.eps * sb * s * v, 2)
                    out[a][mon] = out[a].get(mon, 0) + coef
        self._dgen = [{m: v for m, v in o.items() if v} for o in out]
        return self._dgen

    def d_monomial(self, mon):
        """d of a monomial, as a dict {monomial: coefficient}."""
        dg = self.dgen()
        res = {}
        par = 0
        for i, a in enumerate(mon):
            sgn = (-1) ** par
            par = (par + self.cpar[a]) % 2
            for m2, v in dg[a].items():
                s, mm = self.normalize(mon[:i] + m2 + mon[i + 1:])
                if not s:
                    continue
                res[mm] = res.get(mm, 0) + sgn * s * v
        return {m: v for m, v in res.items() if v}

    def check_d2(self):
        """d^2 = 0 on the generators <=> d^2 = 0 (d is an odd derivation)."""
        bad = []
        for a in range(self.dim):
            acc = {}
            for mon, v in self.dgen()[a].items():
                for m2, w in self.d_monomial(mon).items():
                    acc[m2] = acc.get(m2, 0) + v * w
            if any(v for v in acc.values()):
                bad.append(a)
        return bad

    # ------------------------------------------------------------ blocks
    def blocks(self, q):
        """Split the degree-q basis into blocks by (internal degree, torus weight)."""
        bl = {}
        for mon in self.basis(q):
            bl.setdefault((self.mdeg(mon), self.mwt(mon)), []).append(mon)
        return bl

    def matrix(self, src, tgt_index):
        """Matrix of d from a list of monomials to an indexed target basis."""
        rows = []
        for mon in src:
            row = [0] * len(tgt_index)
            for m2, v in self.d_monomial(mon).items():
                row[tgt_index[m2]] = v
            rows.append(row)
        return rows


def cohomology(ce, qmax, verbose=True, modp=None, w0=False):
    """dim H^q_{k,w} for q <= qmax.  Returns {(q, k, w): dim}.

    w0=True keeps only the blocks of torus weight zero.  That loses nothing:
    for x in the torus, the Lie derivative L_x = d i_x + i_x d is homotopic to
    zero, so H sits in weight zero -- verified independently for n = 4, where
    the full computation returns no class of nonzero weight.
    """
    bases = {q: ce.blocks(q) for q in range(qmax + 2)}
    if w0:
        for q in bases:
            bases[q] = {key: v for key, v in bases[q].items() if not any(key[1])}
    ranks = {}          # (q, key) -> rank of d : C^q_key -> C^{q+1}_key
    for q in range(qmax + 1):
        for key, mons in bases[q].items():
            tgt = bases[q + 1].get(key, [])
            if not tgt:
                ranks[(q, key)] = 0
                continue
            ti = {m: i for i, m in enumerate(tgt)}
            M = ce.matrix(mons, ti)
            ranks[(q, key)] = exact_rank(M, p=modp) if M else 0
    out = {}
    for q in range(qmax + 1):
        for key, mons in bases[q].items():
            r_out = ranks[(q, key)]
            r_in = ranks.get((q - 1, key), 0) if q > 0 else 0
            h = len(mons) - r_out - r_in
            if h:
                out[(q,) + key] = h
    return out
