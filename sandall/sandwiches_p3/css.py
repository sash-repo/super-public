#!/usr/bin/env python3
"""
The known simple Lie algebras over F_2 of dimension 15 and 16 from the library of
Cushing--Stagg--Stewart, arXiv:2207.01094 (github.com/cushydom88/simple-lie-algebras,
directory SimpleLieAlgebraBases), read into exact F_2 linear algebra.

For every algebra L we compute
  * dim L, simplicity, dim [L,L], dim Der L;
  * Num2Nilps = #{x != 0 : (ad_x)^2 = 0}   (to be matched with Table C.3 of CSS,
    which is how we check that our reading of their data is right);
  * the sandwich elements: x with (ad_x)^2 = 0 AND [[L,x],[L,x]] = 0  (eq. (san),
    (san2) of the Sandwiches text), the sandwich subalgebra s(L) they generate,
    and its normaliser n_L(s);
  * the Weisfeiler filtration defined by L_0 = n_L(s): depth, the growth vector,
    and the associated graded algebra gr L.
This is the p=2 case of the conjecture of the Sandwiches paper: n_L(s) should be
a maximal subalgebra of the smallest codimension.
"""

import ast
import glob
import os
import sys

# ------------------------------------------------------------------ F_2 vectors


def rref(vectors):
    piv = {}
    for v in vectors:
        for p in sorted(piv, reverse=True):
            if (v >> p) & 1:
                v ^= piv[p]
        if v:
            p = v.bit_length() - 1
            for q in list(piv):
                if (piv[q] >> p) & 1:
                    piv[q] ^= v
            piv[p] = v
    return [piv[p] for p in sorted(piv, reverse=True)]


def reduce_vec(v, rows):
    for r in rows:
        if v ^ r < v:
            v ^= r
    return v


def in_span(v, rows):
    return reduce_vec(v, rows) == 0


def all_elements(rows):
    els = [0]
    for r in rows:
        els += [e ^ r for e in els]
    return els


# ------------------------------------------------------------------- the algebra
class LieF2:
    def __init__(self, tab, name="L"):
        """tab[i][j] = bit vector of [e_i, e_j]"""
        self.tab = tab
        self.n = len(tab)
        self.name = name

    # ---- from CSS data
    @staticmethod
    def from_adjoint_basis(mats, name):
        """mats[i] = matrix of ad(e_i) in the basis (e_j); entry mats[i][j][k] is
        read as the coefficient of e_k in [e_i, e_j] (row j = image of e_j)."""
        n = len(mats)
        cols = []
        for M in mats:
            cols.append([sum((M[j][k] & 1) << k for k in range(n)) for j in range(n)])
        tab = [[cols[i][j] for j in range(n)] for i in range(n)]
        L = LieF2(tab, name)
        return L

    @staticmethod
    def from_generators(mats, name, dim):
        """mats = adjoint matrices of two generators; rebuild L = <mats> in gl(n)."""
        n = dim

        def tovec(M):
            v = 0
            for i in range(n):
                for j in range(n):
                    if M[i][j] & 1:
                        v |= 1 << (i * n + j)
            return v

        def tomat(v):
            return [[(v >> (i * n + j)) & 1 for j in range(n)] for i in range(n)]

        def mul(A, B):
            return [[sum(A[i][k] & B[k][j] for k in range(n)) & 1
                     for j in range(n)] for i in range(n)]

        def add(A, B):
            return [[A[i][j] ^ B[i][j] for j in range(n)] for i in range(n)]

        def br(A, B):
            return add(mul(A, B), mul(B, A))

        gens = [tovec(M) for M in mats]
        rows = rref(gens)
        basis = list(rows)
        changed = True
        while changed:
            changed = False
            for a in list(basis):
                for b in list(basis):
                    c = tovec(br(tomat(a), tomat(b)))
                    if c and not in_span(c, rows):
                        rows = rref(rows + [c])
                        basis = list(rows)
                        changed = True
        assert len(rows) == dim, "generated %d-dimensional algebra" % len(rows)
        # structure constants in the basis `rows` (as matrices)
        mats_b = [tomat(r) for r in rows]
        tab = []
        for A in mats_b:
            line = []
            for B in mats_b:
                c = tovec(br(A, B))
                # express in the basis
                coeff = 0
                v = c
                red = [(rows[i], 1 << i) for i in range(dim)]
                red = rref_pairs(red)
                for w, d in red:
                    if v ^ w < v:
                        v, coeff = v ^ w, coeff ^ d
                assert v == 0
                line.append(coeff)
            tab.append(line)
        return LieF2(tab, name)

    # ---- basics
    def mul(self, u, v):
        out = 0
        for i in range(self.n):
            if (u >> i) & 1:
                for j in range(self.n):
                    if (v >> j) & 1:
                        out ^= self.tab[i][j]
        return out

    def ad(self, u):
        return [self.mul(u, 1 << j) for j in range(self.n)]

    def ad2(self, u):
        cols = self.ad(u)
        out = []
        for j in range(self.n):
            w, r = cols[j], 0
            for k in range(self.n):
                if (w >> k) & 1:
                    r ^= cols[k]
            out.append(r)
        return out

    def check_jacobi(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.tab[i][j] != self.tab[j][i]:
                    return False, "not symmetric (p=2)"
            if self.tab[i][i]:
                return False, "[x,x] != 0"
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    s = (self.mul(1 << i, self.tab[j][k]) ^
                         self.mul(1 << j, self.tab[k][i]) ^
                         self.mul(1 << k, self.tab[i][j]))
                    if s:
                        return False, "Jacobi fails at %d,%d,%d" % (i, j, k)
        return True, "ok"

    def ideal(self, gens):
        rows = rref([g for g in gens if g])
        queue = list(rows)
        while queue:
            u = queue.pop()
            for j in range(self.n):
                w = self.mul(u, 1 << j)
                if w and not in_span(w, rows):
                    rows = rref(rows + [w])
                    queue.append(w)
        return rows

    def is_simple(self):
        for u in range(1, 1 << self.n):
            if len(self.ideal([u])) != self.n:
                return False
        return True

    def derived(self):
        return rref([self.tab[i][j] for i in range(self.n) for j in range(self.n)])

    def dim_der(self):
        n = self.n
        eqs = []
        for i in range(n):
            for j in range(i + 1, n):
                bij = self.tab[i][j]
                for t in range(n):
                    row = 0
                    for k in range(n):
                        if (self.tab[i][k] >> t) & 1:
                            row ^= 1 << (k * n + j)
                        if (self.tab[k][j] >> t) & 1:
                            row ^= 1 << (k * n + i)
                    for k in range(n):
                        if (bij >> k) & 1:
                            row ^= 1 << (t * n + k)
                    if row:
                        eqs.append(row)
        return n * n - len(rref(eqs))

    def num_2nilps(self):
        return sum(1 for u in range(1, 1 << self.n) if not any(self.ad2(u)))

    def sandwiches(self):
        out = []
        for u in range(1, 1 << self.n):
            if any(self.ad2(u)):
                continue
            img = rref(self.ad(u))
            ok = True
            els = all_elements(img)
            for a in els:
                if not ok:
                    break
                for b in els:
                    if self.mul(a, b):
                        ok = False
                        break
            if ok:
                out.append(u)
        return out

    def subalgebra_generated(self, gens):
        rows = rref([g for g in gens if g])
        queue = list(rows)
        while queue:
            u = queue.pop()
            for v in list(rows):
                w = self.mul(u, v)
                if w and not in_span(w, rows):
                    rows = rref(rows + [w])
                    queue.append(w)
        return rows

    def normaliser(self, S):
        rows = rref(list(S))
        good = [u for u in range(1, 1 << self.n)
                if all(in_span(self.mul(u, s), rows) for s in rows)]
        return rref(good)

    def is_subalgebra(self, rows):
        return all(in_span(self.mul(a, b), rows)
                   for a in rows for b in rows)

    def is_maximal_subalgebra(self, rows):
        """rows a subalgebra; test that nothing lies strictly between it and L."""
        rows = rref(rows)
        comp = [1 << i for i in range(self.n) if not in_span(1 << i, rows)]
        # all subspaces between: generated by rows + a subset of a complement
        from itertools import combinations
        k = self.n - len(rows)
        base = []
        cnt = 0
        for u in range(1, 1 << self.n):
            if in_span(u, rows):
                continue
            M = rref(rows + [u])
            if len(M) < self.n and self.is_subalgebra(M):
                return False, M
            cnt += 1
        return True, None

    def flat_family(self, S):
        """every K-point of Span(S) is a sandwich?"""
        for a in S:
            if any(self.ad2(a)):
                return False
        for a in S:
            for b in S:
                ca, cb = self.ad(a), self.ad(b)
                for j in range(self.n):
                    x = 0
                    for (c1, c2) in ((ca, cb), (cb, ca)):
                        w = c2[j]
                        for k in range(self.n):
                            if (w >> k) & 1:
                                x ^= c1[k]
                    if x:
                        return False
        return True

    # ---- Weisfeiler filtration attached to a maximal subalgebra L_0
    def w_filtration(self, L0):
        """returns the list of dims of L_{-d} ... L_0 ... and the growth vector"""
        L0 = rref(L0)
        # L_{-1}: minimal L_0-invariant subspace strictly containing L_0
        best = None
        for u in range(1, 1 << self.n):
            if in_span(u, L0):
                continue
            M = rref(L0 + [u])
            # close under ad(L_0)
            changed = True
            while changed:
                changed = False
                for a in list(M):
                    for b in L0:
                        w = self.mul(a, b)
                        if w and not in_span(w, M):
                            M = rref(M + [w])
                            changed = True
            if best is None or len(M) < len(best):
                best = M
        Lm1 = best
        dims = [len(L0), len(Lm1)]
        cur = Lm1
        while len(cur) < self.n:
            new = rref(cur + [self.mul(a, b) for a in Lm1 for b in cur])
            if len(new) == len(cur):
                break
            cur = new
            dims.append(len(cur))
        return dims


def rref_pairs(pairs):
    out = []
    for v, c in pairs:
        for w, d in out:
            if v ^ w < v:
                v, c = v ^ w, c ^ d
        if v:
            out.append((v, c))
            out.sort(key=lambda p: -p[0])
    return out


def load(path):
    s = open(path).read().replace("\n", " ").strip()
    if s.endswith(";"):
        s = s[:-1]
    return ast.literal_eval(s)


def main(paths):
    for path in paths:
        data = load(path)
        name = os.path.basename(path)[:-4]
        dim = len(data[0])
        if len(data) == dim:
            L = LieF2.from_adjoint_basis(data, name)
        else:
            L = LieF2.from_generators(data, name, dim)
        ok, msg = L.check_jacobi()
        print("=" * 70)
        print("%s   dim %d   Lie algebra axioms: %s" % (name, L.n, msg))
        if not ok:
            continue
        print("  simple: %s   dim [L,L] = %d   dim Der L = %d"
              % (L.is_simple(), len(L.derived()), L.dim_der()))
        print("  Num2Nilps = %d" % L.num_2nilps())
        S = L.sandwiches()
        Sp = rref(S)
        print("  # sandwiches (F_2 points) = %d, spanning dim %d, flat family: %s"
              % (len(S), len(Sp), L.flat_family(Sp)))
        if not S:
            continue
        sg = L.subalgebra_generated(Sp)
        print("  dim s(L) = %d" % len(sg))
        nn = L.normaliser(sg)
        print("  dim n_L(s) = %d, codim %d, subalgebra: %s"
              % (len(nn), L.n - len(nn), L.is_subalgebra(nn)))
        if L.is_subalgebra(nn) and len(nn) < L.n:
            mx, wit = L.is_maximal_subalgebra(nn)
            print("  n_L(s) maximal: %s" % mx)
            print("  W-filtration dims from L_0 = n_L(s): %s" % L.w_filtration(nn))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("css-bases/dim15/*.txt"))
    main(args)
