#!/usr/bin/env python3
"""
The graded Lie algebras  g(W) = V + sl(V) + W  inside vect(3; 1) for p=2,
where dim V = 3 and W is a 4- or 5-dimensional sl(3)-submodule of vect(3;1)_1.

These are the candidates for the associated graded algebras of the Weisfeiler
filtrations of the Eick algebras 15:7, 15:8 (dim g_1 = 4) and 16:3, 16:4
(dim g_1 = 5) stated in the bestiary of simple modular Lie algebras.

Computed here, all over F_2 unless stated otherwise:
 * the lattice of submodules W (via vect2.py);
 * transitivity of GL(3, F_2) on the 4- resp. 5-dimensional W's;
 * ideals / simplicity of g(W), its derived series;
 * dim Der g(W)  (an isomorphism invariant);
 * the sandwich elements of g(W), the sandwich subalgebra s(g) and its
   normaliser n_g(s), compared with the Weisfeiler subalgebra g_0 + g_1.
"""

import itertools
import vect2 as V2
from vect2 import bracket, square, show, rref, in_span, all_elements

NV = 3


# ------------------------------------------------------------------ the algebra
class Alg:
    """A Lie algebra over F_2 given by a basis of vector fields."""

    def __init__(self, basis, name="g"):
        self.basis = list(basis)
        self.n = len(self.basis)
        self.name = name
        self.index = {b: i for i, b in enumerate(self.basis)}
        # structure constants as bit vectors
        self.tab = [[self.tov(bracket(a, b)) for b in self.basis]
                    for a in self.basis]
        self.sq = [self.tov(square(a)) for a in self.basis]

    # coordinates: a vector field -> bit vector in the basis of the algebra
    def _tov(self, f):
        # temporary: use a global monomial index
        return f

    def tov(self, f):
        """Express the vector field f in the algebra's basis (must lie in it)."""
        target = f
        out = 0
        rem = set(target)
        # Gaussian elimination in the ambient monomial space
        return self._solve(frozenset(rem))

    def _solve(self, f):
        # build once: echelon form of the basis in the ambient space
        if not hasattr(self, "_ech"):
            amb = sorted({t for b in self.basis for t in b} |
                         {(r, i) for r in V2.MONS for i in range(NV)})
            self._amb = {t: k for k, t in enumerate(amb)}
            piv = {}
            self._ech = []
            rows = []
            for i, b in enumerate(self.basis):
                v = 0
                for t in b:
                    v |= 1 << self._amb[t]
                rows.append((v, 1 << i))
            red = []
            for v, c in rows:
                for w, d in red:
                    if v ^ w < v:
                        v, c = v ^ w, c ^ d
                if v:
                    red.append((v, c))
                    red.sort(key=lambda p: -p[0])
            self._red = red
        v = 0
        for t in f:
            v |= 1 << self._amb[t]
        c = 0
        for w, d in self._red:
            if v ^ w < v:
                v, c = v ^ w, c ^ d
        assert v == 0, "element outside the algebra: %s" % show(f)
        return c

    # ------------------------------------------------------------- linear algebra
    def mul(self, u, v):
        """bracket of two coordinate vectors"""
        out = 0
        for i in range(self.n):
            if (u >> i) & 1:
                for j in range(self.n):
                    if (v >> j) & 1:
                        out ^= self.tab[i][j]
        return out

    def sqr(self, u):
        """u^[2] in char 2:  (sum u_i e_i)^[2] = sum u_i e_i^[2] + sum_{i<j} [e_i,e_j]
        -- only valid over F_2 (coefficients squared = themselves)."""
        out = 0
        idx = [i for i in range(self.n) if (u >> i) & 1]
        for i in idx:
            out ^= self.sq[i]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                out ^= self.tab[idx[a]][idx[b]]
        return out

    def ad(self, u):
        """ad_u as a list of images of the basis vectors"""
        return [self.mul(u, 1 << j) for j in range(self.n)]

    def ad2(self, u):
        cols = self.ad(u)
        out = []
        for j in range(self.n):
            w = cols[j]
            r = 0
            for k in range(self.n):
                if (w >> k) & 1:
                    r ^= cols[k]
            out.append(r)
        return out

    def ideal_generated(self, gens):
        rows = rref([g for g in gens if g])
        queue = list(rows)
        while queue:
            u = queue.pop()
            for j in range(self.n):
                w = self.mul(u, 1 << j)
                if w and not in_span(w, rows):
                    rows = rref(rows + [w])
                    queue.append(w)
        return tuple(rows)

    def is_simple(self):
        for u in range(1, 1 << self.n):
            I = self.ideal_generated([u])
            if len(I) != self.n:
                return False, u, I
        return True, None, None

    def derived(self):
        gens = [self.tab[i][j] for i in range(self.n) for j in range(self.n)]
        return tuple(rref([g for g in gens if g]))

    def derivations(self):
        """dim Der(g) over F_2: solve D[x,y] = [Dx,y] + [x,Dy] linearly."""
        n = self.n
        # unknowns: D_{kj} = coefficient of e_k in D(e_j), n*n unknowns
        eqs = []
        for i in range(n):
            for j in range(i + 1, n):
                bij = self.tab[i][j]
                # sum_k D_kj [e_i,e_k] + sum_k D_ki [e_k,e_j] + D(b_ij) = 0
                for target in range(n):
                    row = 0
                    for k in range(n):
                        c = (self.tab[i][k] >> target) & 1
                        if c:
                            row ^= 1 << (k * n + j)
                        c = (self.tab[k][j] >> target) & 1
                        if c:
                            row ^= 1 << (k * n + i)
                    for k in range(n):
                        if (bij >> k) & 1:
                            row ^= 1 << (target * n + k)
                    if row:
                        eqs.append(row)
        red = rref(eqs)
        return n * n - len(red)

    # ------------------------------------------------------------------ sandwiches
    def sandwiches(self):
        """all s in g(F_2) with (ad_s)^2 = 0 and [[g,s],[g,s]] = 0"""
        out = []
        for u in range(1, 1 << self.n):
            cols = self.ad2(u)
            if any(cols):
                continue
            img = rref([self.mul(u, 1 << j) for j in range(self.n)])
            ok = True
            for a in all_elements(img):
                if not ok:
                    break
                for b in all_elements(img):
                    if self.mul(a, b):
                        ok = False
                        break
            if ok:
                out.append(u)
        return out

    def flat_family(self, S):
        """True if every K-point of Span_K(S) is a sandwich, i.e. if
        (ad_a)^2 = 0 and ad_a ad_b + ad_b ad_a = 0 for all a,b in the basis S."""
        for a in S:
            if any(self.ad2(a)):
                return False
        for a in S:
            for b in S:
                ca, cb = self.ad(a), self.ad(b)
                for j in range(self.n):
                    x = 0
                    w = cb[j]
                    for k in range(self.n):
                        if (w >> k) & 1:
                            x ^= ca[k]
                    w = ca[j]
                    for k in range(self.n):
                        if (w >> k) & 1:
                            x ^= cb[k]
                    if x:
                        return False
        return True

    def normaliser(self, S):
        """n_g(S) = {x : [x, S] subset S} for a subspace S (list of rows)."""
        rows = rref(list(S))
        out = []
        for j in range(self.n):
            pass
        # linear condition on x: solve by testing a basis of g and combining
        good = []
        for u in range(1, 1 << self.n):
            if all(in_span(self.mul(u, s), rows) for s in rows):
                good.append(u)
        return tuple(rref(good))


# ------------------------------------------------------------------- ingredients
def sl3():
    return V2.SL3


def gm1():
    return [frozenset({(0, i)}) for i in range(NV)]


def coord_change(A):
    """The automorphism of vect(3;1) induced by x_i -> sum_j A[i][j] x_j.
    Returns a function on vector fields (A invertible over F_2)."""
    import itertools as it

    def monimage(r):
        """image of the monomial x^r (r a bitmask), as a set of monomials"""
        facs = []
        for k in range(NV):
            if (r >> k) & 1:
                facs.append([j for j in range(NV) if A[k][j]])
        if not facs:
            return {0}
        out = {0}
        for f in facs:
            new = set()
            for m in out:
                for j in f:
                    if not (m >> j) & 1:
                        new ^= {m | (1 << j)}
            out = new
        return out

    Ainv = inv3(A)

    def act(field):
        res = set()
        for (r, i) in field:
            for m in monimage(r):
                for j in range(NV):
                    if Ainv[j][i]:
                        res ^= {(m, j)}
        return frozenset(res)

    return act


def inv3(A):
    """inverse of a 3x3 matrix over F_2"""
    n = 3
    M = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(A)]
    r = 0
    for c in range(n):
        p = None
        for i in range(r, n):
            if M[i][c]:
                p = i
                break
        assert p is not None, "singular"
        M[r], M[p] = M[p], M[r]
        for i in range(n):
            if i != r and M[i][c]:
                M[i] = [(a ^ b) for a, b in zip(M[i], M[r])]
        r += 1
    return [row[n:] for row in M]


def gl3_f2():
    out = []
    for bits in itertools.product([0, 1], repeat=9):
        A = [list(bits[0:3]), list(bits[3:6]), list(bits[6:9])]
        try:
            inv3(A)
        except AssertionError:
            continue
        out.append(A)
    return out


def main():
    subs = V2.all_submodules(maxdim=5)
    by_dim = {}
    for s in subs:
        by_dim.setdefault(len(s), []).append(s)
    W4 = [tuple(sorted(V2.vec_to_field(v) for v in s)) for s in by_dim[4]]
    W5 = [tuple(sorted(V2.vec_to_field(v) for v in s)) for s in by_dim[5]]
    print("4-dim submodules: %d   5-dim submodules: %d" % (len(W4), len(W5)))

    # --- transitivity of GL(3,F_2)
    G = gl3_f2()
    print("|GL(3,F_2)| =", len(G))
    for name, Ws in (("4", W4), ("5", W5)):
        spaces = [canon(W) for W in Ws]
        orb = {spaces[0]}
        frontier = [spaces[0]]
        spaces = set(spaces)
        while frontier:
            S = frontier.pop()
            for A in G:
                act = coord_change(A)
                T = canon([act(f) for f in S])
                if T not in orb:
                    assert T in spaces, "orbit left the list!"
                    orb.add(T)
                    frontier.append(T)
        print("  GL(3,F_2)-orbit of one %s-dimensional W has size %d (of %d)"
              % (name, len(orb), len(Ws)))

    # --- the two graded algebras
    for Ws, dim in ((W4, 15), (W5, 16)):
        W = Ws[0]
        basis = gm1() + sl3() + list(W)
        g = Alg(basis, "g(W) dim %d" % dim)
        assert g.n == dim
        print("\n===== g(W), dim %d, W = %s" % (dim, [show(w) for w in W]))
        simple, wit, I = g.is_simple()
        print("  simple:", simple, end="")
        if not simple:
            print("   -- proper ideal of dim %d, generated by %s"
                  % (len(I), show_vec(g, wit)))
            print("     ideal =", [show_vec(g, r) for r in I])
        else:
            print()
        der = g.derived()
        print("  dim [g,g] =", len(der))
        print("  dim Der(g) over F_2 =", g.derivations())
        S = g.sandwiches()
        Sp = rref(S)
        print("  # sandwich elements over F_2 =", len(S),
              "  they span a subspace of dim", len(Sp))
        print("     span:", [show_vec(g, r) for r in Sp])
        print("     every K-point of the span is a sandwich:", g.flat_family(Sp))
        # sandwich subalgebra: subalgebra generated by the sandwich set
        rows = rref(Sp)
        queue = list(rows)
        while queue:
            u = queue.pop()
            for v in list(rows):
                w = g.mul(u, v)
                if w and not in_span(w, rows):
                    rows = rref(rows + [w])
                    queue.append(w)
        print("  dim s(g) =", len(rows))
        n = g.normaliser(rows)
        print("  dim n_g(s) =", len(n), " codim =", g.n - len(n))
        g0g1 = rref([g.tov(b) for b in sl3() + list(W)])
        print("  n_g(s) == g_0 + g_1 :", set(n) == set(g0g1))
        print("     n_g(s) =", [show_vec(g, r) for r in n])


def canon(fields):
    """canonical label of the F_2-span of a set of vector fields:
    the frozenset of all its elements."""
    els = [frozenset()]
    for f in fields:
        els += [frozenset(set(e) ^ set(f)) for e in els]
    return frozenset(els)


def show_vec(g, u):
    terms = []
    for i in range(g.n):
        if (u >> i) & 1:
            terms.append(show(g.basis[i]))
    return " + ".join(terms) if terms else "0"


if __name__ == "__main__":
    main()
