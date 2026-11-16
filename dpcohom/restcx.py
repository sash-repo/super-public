r"""restcx.py -- THE RESTRICTED COCHAIN COMPLEX IN ALL DEGREES, p = 2.

Engine for the restricted cochains: explicit cochains,
in EVERY degree, computing  H^n_res(g;M) = Ext^n_{u(g)}(K;M).

THE ANSWER.  It is the divided power complex of the text -- with the divided
powers taken on the WHOLE of the algebra, not only on its odd part.  Namely, let
p = 2, g a Lie algebra, g' a subspace of g, and

    [2] : g' -> g      with   (ax)^{[2]} = a^2 x^{[2]},
                              (x+y)^{[2]} = x^{[2]} + y^{[2]} + [x,y],
                              [x^{[2]},y] = [x,[x,y]]

--- i.e. exactly the axioms of the squaring of a Lie superalgebra for p = 2, but
with the map defined on a subspace of g and with values in g.  Put

    u = U(g)/(x^2 + x^{[2]} | x in g'),      C = Gamma(g') (x) Lambda(g/g'),

and let d be the differential of the main theorem of the text, i.e.

    d(1 (x) c) = sum_k  x_k (x) partial_k(c) + 1 (x) delta(c),
    delta = mult o (beta (x) id) o Delta_{2,.},
    beta:  x/\y, x gamma_1(y), gamma_1(x)gamma_1(y) |-> [x,y],  gamma_2(x) |-> x^{[2]}.

Then V = u (x) C is a free resolution of K over u and the dual complex computes
Ext_u(K;M) in ALL degrees.  Since the data (g', [2]) are literally the data of a
squaring, THE ENGINE IS dpco.py ITSELF: one feeds it the height-2 generators in
place of the odd ones.  That is what this file does; it adds

  * the restricted closure of a (2,4)-structure:  u_{2,4}(g) = u_2(L),
    L = g_+ (+) g_- (+) g_-^{(2)},  the (2,4)-structure becoming an honest
    2-structure on L.  L is computed INSIDE u_{2,4}(g) (so its structure
    constants are not guessed but read off the multiplication of u);
  * the comparison of the resulting cohomology with Ext over u, computed
    independently by resco.py from a free resolution.

The list of cases covered (see [BLLS, Sec. 2] = arXiv:1407.1695):

    non-restricted superalgebra   g' = g_od,  [2] = the squaring s
    2-structure                   g' = g
    2|2- and 2|4-structures       g' = g   (on g_od the map is s; the two
                                  restrictednesses give the SAME algebra u)
    (2,-)-structure               g' = g_+
    (2,4)-structure               g' = L    (after the closure)
    (2,4)|4-structure             g' = L    (same, with the odd part of g_-)
"""

import sys

sys.setrecursionlimit(100000)

import dpco
import resco
from dpco import bits, rank2_int, span_int, in_span


# ------------------------------------------------------------------ the data

class RLA:
    """A Lie algebra over F_2 with a squaring on a subspace.

    dim, br as in dpco; the first `nfree` basis vectors span a complement of g'
    (height infinity), the remaining ones span g' (height 2) and carry pmap.
    Internally this is a dpco.SLA2 with m = nfree "even" and n = dim - nfree
    "odd" generators -- the parity being a bookkeeping device only.
    """

    def __init__(self, dim, br, nfree, pmap, name="g"):
        self.dim, self.nfree, self.name = dim, nfree, name
        self.br = br
        self.pmap = pmap                 # length dim - nfree, for the height-2 part
        brd = {}
        for i in range(dim):
            for j in range(i + 1, dim):
                if br[i][j]:
                    brd[(i, j)] = br[i][j]
        self.sla = dpco.SLA2(nfree, dim - nfree, brd, pmap, name)
        self.res = dpco.Resolution(self.sla)

    # ---- the axioms actually used by the construction --------------------
    def check(self):
        bad = []
        g, d = self.sla, self.dim
        for i in range(d):
            if self.br[i][i]:
                bad.append("[x_%d,x_%d] != 0" % (i, i))
            for j in range(d):
                if self.br[i][j] != self.br[j][i]:
                    bad.append("bracket not symmetric (%d,%d)" % (i, j))
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    if (g.bracket(1 << i, self.br[j][k])
                            ^ g.bracket(1 << j, self.br[i][k])
                            ^ g.bracket(1 << k, self.br[i][j])):
                        bad.append("Jacobi(%d,%d,%d)" % (i, j, k))
        # the squaring axioms on the height-2 subspace
        m, n = self.nfree, self.dim - self.nfree
        for x in range(1, 1 << n):
            xm = x << m
            sx = g.squaring(xm)
            for y in range(d):
                if g.bracket(sx, 1 << y) != g.bracket(xm, g.bracket(xm, 1 << y)):
                    bad.append("[x^[2],y] != [x,[x,y]] at x=%d, y=%d" % (x, y))
            for y in range(1, 1 << n):
                if g.squaring((x ^ y) << m) != (g.squaring(x << m)
                                                ^ g.squaring(y << m)
                                                ^ g.bracket(x << m, y << m)):
                    bad.append("polarization at (%d,%d)" % (x, y))
        return bad

    # ---- modules ---------------------------------------------------------
    def trivial(self):
        return self.sla.trivial_module()

    def adjoint(self):
        return self.sla.adjoint_module()

    def module_from_action(self, dim, act):
        """act(k, v): the action of the k-th basis vector of g on M"""
        return dpco.Module(dim, [[act(k, 1 << i) for k in range(self.dim)]
                                 for i in range(dim)],
                           [0] * dim, "M")

    def check_restricted(self, M):
        """rho(x^{[2]}) = rho(x)^2 on the height-2 basis vectors"""
        bad = []
        m = self.nfree
        for j in range(self.dim - m):
            for t in range(M.dim):
                lhs = M.apply_vec(self.pmap[j], 1 << t) if hasattr(M, "apply_vec") \
                    else _apply_vec(M, self.pmap[j], 1 << t)
                rhs = M.apply(j + m, M.apply(j + m, 1 << t))
                if lhs != rhs:
                    bad.append("not restricted at x_%d, v_%d" % (j + m, t))
        return bad

    # ---- the two things we want ------------------------------------------
    def dsquared(self, maxdeg):
        """d^2 = 0 on the resolution, in degrees <= maxdeg"""
        bad = 0
        for n in range(2, maxdeg + 1):
            for c in self.res.ce.basis(n):
                acc = {}
                for (w, c1) in self.res.d_gen(c):
                    for (w2, c2) in self.res.d_gen(c1):
                        for red in self.res.env.reduce(w + w2):
                            acc[(red, c2)] = acc.get((red, c2), 0) ^ 1
                if any(acc.values()):
                    bad += 1
        return bad

    def cohom(self, M, maxdeg):
        return dpco.cohomology_dims(self.res, M, maxdeg)

    def u(self):
        """u = U(g)/(x^2+x^{[2]}) as a finite dimensional algebra; only when
        every generator has height 2."""
        assert self.nfree == 0, "u is infinite dimensional unless g' = g"
        return resco.RestAlg(self.dim, self.br, [2] * self.dim, self.pmap)

    def ext(self, M, length):
        u = self.u()
        maps = resco.resolution(u, length)
        Mact = resco.module_action(u, lambda k, v: M.apply(k, v), M.dim)
        return u, resco.ext_dims(u, maps, Mact, M.dim, length)


def _apply_vec(M, mask, v):
    out = 0
    for k in bits(mask):
        out ^= M.apply(k, v)
    return out


# ------------------------------------------- the restricted closure of (2,4)

class Closure:
    """For a (2,4)-structure on g = g_+ (+) g_-, the restricted closure

        L = g_+ (+) g_- (+) g_-^{(2)},   the third summand spanned by the
        squares y^2 taken IN u_{2,4}(g),

    which is a Lie algebra with an honest 2-structure, and u_2(L) = u_{2,4}(g)
    (equal dimensions 2^{dim g_+} 4^{dim g_-}, and L generates u).

    Everything is computed inside u = u_{2,4}(g): the brackets and the squares
    of the basis of L are read off the multiplication of u and expressed in the
    span of that basis, so no formula is assumed.
    """

    def __init__(self, dimp, dimm, br, pmap2, pmap4):
        """basis of g: 0..dimp-1 spans g_+, dimp..dimp+dimm-1 spans g_-;
        pmap2[i] = x_i^{[2]} for i in g_+, pmap4[j] = y_j^{[4]} for j in g_-"""
        self.dimp, self.dimm = dimp, dimm
        d = dimp + dimm
        self.dim_g = d
        self.br_g = br
        self.u = resco.RestAlg(d, br, [2] * dimp + [4] * dimm,
                               list(pmap2) + list(pmap4))
        n = len(self.u.basis)
        # the basis of L inside u: x_i (i < d), then y_j^2 (j < dimm)
        self.Lbasis = [self._word(( i,)) for i in range(d)]
        for j in range(dimm):
            self.Lbasis.append(self._word((dimp + j, dimp + j)))
        self.dimL = len(self.Lbasis)
        self.piv = span_int(self.Lbasis)
        self.problems = []
        self._extract()

    def _word(self, w):
        out = 0
        for red in self.u.reduce(w):
            out ^= 1 << self.u.index[red]
        return out

    def _coords(self, v):
        """express v in the basis Lbasis; None if not in the span"""
        # Gaussian elimination keeping track of the combination
        piv = {}
        for a, b in enumerate(self.Lbasis):
            cur, comb = b, 1 << a
            while cur:
                h = cur.bit_length() - 1
                if h in piv:
                    cur ^= piv[h][0]
                    comb ^= piv[h][1]
                else:
                    piv[h] = (cur, comb)
                    break
        cur, comb = v, 0
        while cur:
            h = cur.bit_length() - 1
            if h not in piv:
                return None
            cur ^= piv[h][0]
            comb ^= piv[h][1]
        return comb

    def _extract(self):
        n = len(self.u.basis)
        dL = self.dimL
        self.br = [[0] * dL for _ in range(dL)]
        for a in range(dL):
            for b in range(a + 1, dL):
                v = self.u.mul(self.Lbasis[a], self.Lbasis[b]) \
                    ^ self.u.mul(self.Lbasis[b], self.Lbasis[a])
                c = self._coords(v)
                if c is None:
                    self.problems.append("[L_%d,L_%d] leaves L" % (a, b))
                    c = 0
                self.br[a][b] = c
                self.br[b][a] = c
        self.pmap = []
        for a in range(dL):
            v = self.u.mul(self.Lbasis[a], self.Lbasis[a])
            c = self._coords(v)
            if c is None:
                self.problems.append("L_%d^2 leaves L" % a)
                c = 0
            self.pmap.append(c)

    def rla(self):
        return RLA(self.dimL, self.br, 0, self.pmap, "L")

    def module_L(self, Mdim, act_g):
        """lift a restricted g-module to L: the twist y^2 acts as rho(y)^2"""
        def act(k, v):
            if k < self.dim_g:
                return act_g(k, v)
            j = k - self.dim_g
            return act_g(self.dimp + j, act_g(self.dimp + j, v))
        return dpco.Module(Mdim,
                           [[act(k, 1 << i) for k in range(self.dimL)]
                            for i in range(Mdim)], [0] * Mdim, "M")

    def ext_over_u24(self, Mdim, act_g, length):
        maps = resco.resolution(self.u, length)
        Mact = resco.module_action(self.u, act_g, Mdim)
        return resco.ext_dims(self.u, maps, Mact, Mdim, length)
