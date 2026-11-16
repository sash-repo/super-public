#!/usr/bin/env python3
"""seq5.py -- the analog of Hochschild's six-term
sequence for the restrictednesses of characteristic 2, including the
(2,4)-structures.  Everything exact over F_2.

THE STATEMENT.  Let p = 2 and let g carry one of the restrictednesses, u(g) the
restricted enveloping algebra, and M a restricted module.  Write C_res for the
divided power complex Gamma(L)^* (x) M of restcx.py (L = g for a 2-, 2|2- or
2|4-structure, L = g (+) g_-^{(2)} the restricted closure for a (2,4)-structure)
and C for the ordinary Chevalley-Eilenberg complex Lambda(g)^* (x) M.  Then

    0 --> N --> C_res --> C --> 0                                          (*)

is a short exact sequence of complexes, N being the cochains that vanish on the
square-free monomials in the letters of g, and its long exact sequence is the
wanted comparison sequence.  Since H^0(N) = H^1(N) = 0 it begins

    0 -> H^1_res -> H^1 -> H^2(N) -> H^2_res -> H^2,

and H^2(N) is the space of the "semilinear" maps: for a 2-structure

    H^2(N) = Hom_Fr(g; M^g),          psi |--> (x |-> psi(gamma_2(x))),

and for a (2,4)-structure

    H^2(N) = Hom_{Fr^2}(g_+; M^g) (+) Hom_{Fr^4}(g_-; M^g),
             psi |--> (x |-> psi(gamma_2(x)),  y |-> psi(gamma_2(y^2))),

the connecting map being, on a 1-cocycle c,

    theta(c) = ( x |-> c(x^{[2]}) + x.c(x),  y |-> c(y^{[4]}) + y^3.c(y) ),

which is Hochschild's map with the second component added.  Over the prime field
the semilinearity is vacuous (a^2 = a), so what the machine sees is additivity
and the values lying in M^g; the semilinearity itself is one line by hand.

WHAT IS CHECKED HERE:
  1) (*) is a short exact sequence of complexes: the restriction is onto and
     commutes with the differentials;
  2) H^0(N) = H^1(N) = 0;
  3) dim H^2(N) = dim Hom(g_+;M^g) + dim Hom(g_-;M^g) -- one summand for a
     2-structure -- and the readoff map above is an isomorphism onto it;
  4) the connecting map of (*) equals theta, cocycle by cocycle;
  5) exactness of the five-term sequence at each of its four inner spots, by
     ranks.
"""

import sys

sys.setrecursionlimit(100000)

import dpco
from dpco import bits, span_int, in_span, rank2_int
from restcx import RLA, Closure

LOG = []


def out(s=""):
    print(s)
    LOG.append(s)
    sys.stdout.flush()


# ------------------------------------------------------------ linear algebra

def kernel(images, nsrc):
    """basis (as ints over the source indices) of the kernel of the map given
    by images[j] = the image of the j-th source basis vector"""
    piv = {}
    ker = []
    for j in range(nsrc):
        cur, comb = images[j], 1 << j
        while cur:
            h = cur.bit_length() - 1
            if h in piv:
                cur ^= piv[h][0]
                comb ^= piv[h][1]
            else:
                piv[h] = (cur, comb)
                break
        if cur == 0:
            ker.append(comb)
    return ker


def apply_map(images, v):
    out_ = 0
    for j in bits(v):
        out_ ^= images[j]
    return out_


def dim_span(vecs):
    return rank2_int(list(vecs))


def dim_sum(a, b):
    """dim (span a + span b)"""
    return rank2_int(list(a) + list(b))


# ---------------------------------------------------------------- a complex

class Cplx:
    """the cochain complex Hom(C_n, M) of a dpco.Resolution, in ints.

    basis index of degree n:  a * M.dim + t,  a the monomial index."""

    def __init__(self, res, M, maxdeg):
        self.res, self.M, self.maxdeg = res, M, maxdeg
        self.mono = [res.ce.basis(n) for n in range(maxdeg + 2)]
        self.dim = [len(self.mono[n]) * M.dim for n in range(maxdeg + 2)]
        self.d = []
        for n in range(maxdeg + 1):
            rows, ns, nd = res.codiff_matrix(n, M)
            self.d.append([_tobits(r) for r in rows])

    def Z(self, n):
        return kernel(self.d[n], self.dim[n])

    def B(self, n):
        if n == 0:
            return []
        return list(self.d[n - 1])

    def H(self, n):
        z = self.Z(n)
        b = self.B(n)
        return dim_span(z) - dim_span(b)


def _tobits(row):
    v = 0
    for j, x in enumerate(row):
        if x:
            v |= 1 << j
    return v


# ------------------------------------------------ the comparison of complexes

class Comparison:
    """C_res = Gamma(L)^* (x) M   -->>   C = Lambda(g)^* (x) M"""

    def __init__(self, L, g_ord, dim_g, Mres, Mord, maxdeg=3):
        """L: the RLA with all heights 2 (all generators "odd" for dpco);
        g_ord: the RLA with all heights infinity (all generators "even");
        dim_g: the first dim_g basis vectors of L are those of g."""
        self.L, self.g_ord, self.dim_g = L, g_ord, dim_g
        self.res = Cplx(L.res, Mres, maxdeg + 1)
        self.ord = Cplx(g_ord.res, Mord, maxdeg + 1)
        self.Mdim = Mres.dim
        assert Mres.dim == Mord.dim
        self.maxdeg = maxdeg
        # the restriction map, degree by degree
        self.R = []
        self.Nidx = []
        for n in range(maxdeg + 2):
            src = self.res.mono[n]
            dst = self.ord.mono[n]
            dsti = {c: b for b, c in enumerate(dst)}
            imgs = []
            nidx = []
            for a, c in enumerate(src):
                key = self._squarefree_g(c)
                for t in range(self.Mdim):
                    if key is None:
                        imgs.append(0)
                    else:
                        b = dsti[key]
                        imgs.append(1 << (b * self.Mdim + t))
                if key is None:
                    for t in range(self.Mdim):
                        nidx.append(a * self.Mdim + t)
            self.R.append(imgs)
            self.Nidx.append(nidx)

    def _squarefree_g(self, c):
        """the Lambda^n(g)-monomial corresponding to c, or None"""
        S, al = c                              # S is empty: all heights are 2
        supp = []
        for i, a in enumerate(al):
            if a == 0:
                continue
            if a > 1 or i >= self.dim_g:
                return None
            supp.append(i)
        return (frozenset(supp), ())

    # -- N as a coordinate subspace ---------------------------------------
    def N_basis(self, n):
        return [1 << j for j in self.Nidx[n]]

    def in_N(self, n, v):
        mask = 0
        for j in self.Nidx[n]:
            mask |= 1 << j
        return (v & ~mask) == 0

    def Z_N(self, n):
        """cocycles of N in degree n"""
        nb = self.Nidx[n]
        imgs = [self.res.d[n][j] for j in nb]
        ker = kernel(imgs, len(nb))
        # re-express in the coordinates of C_res^n
        out_ = []
        for v in ker:
            w = 0
            for a in bits(v):
                w |= 1 << nb[a]
            out_.append(w)
        return out_

    def B_N(self, n):
        if n == 0:
            return []
        return [self.res.d[n - 1][j] for j in self.Nidx[n - 1]]

    def H_N(self, n):
        return dim_span(self.Z_N(n)) - dim_span(self.B_N(n))

    # -- the three maps of the long exact sequence -------------------------
    def check_chain_map(self):
        """R o d_res = d_ord o R"""
        bad = 0
        for n in range(self.maxdeg + 1):
            for j in range(self.res.dim[n]):
                a = apply_map(self.R[n + 1], self.res.d[n][j])
                b = apply_map(self.ord.d[n], self.R[n][j])
                if a != b:
                    bad += 1
        return bad

    def check_onto(self):
        """R is surjective in every degree"""
        bad = 0
        for n in range(self.maxdeg + 2):
            r = dim_span(self.R[n])
            if r != self.ord.dim[n]:
                bad += 1
        return bad

    def lift(self, n, v):
        """a preimage under R of v in C^n_ord: the cochain that agrees with v on
        the square-free g-monomials and vanishes elsewhere"""
        w = 0
        for j in range(self.res.dim[n]):
            img = self.R[n][j]
            if img and (v & img):
                w |= 1 << j
        return w

    def connecting(self, n, z):
        """the connecting map H^n_ord -> H^{n+1}(N) on a cocycle z"""
        return apply_map(self.res.d[n], self.lift(n, z))

    def five_term(self):
        """the dimensions and the exactness of
        0 -> H^1_res -> H^1 -> H^2(N) -> H^2_res -> H^2"""
        info = {}
        info["H1res"] = self.res.H(1)
        info["H1"] = self.ord.H(1)
        info["H2N"] = self.H_N(2)
        info["H2res"] = self.res.H(2)
        info["H2"] = self.ord.H(2)
        info["H0N"] = self.H_N(0)
        info["H1N"] = self.H_N(1)

        # rank of iota_*: H^2(N) -> H^2_res
        Bres2 = self.res.B(2)
        rk_iota = dim_sum(self.Z_N(2), Bres2) - dim_span(Bres2)
        # rank of rho_*: H^n_res -> H^n_ord
        def rk_rho(n):
            Zres = self.res.Z(n)
            imgs = [apply_map(self.R[n], v) for v in Zres]
            Bord = self.ord.B(n)
            return dim_sum(imgs, Bord) - dim_span(Bord)
        # rank of the connecting map H^n_ord -> H^{n+1}(N)
        def rk_delta(n):
            Zord = self.ord.Z(n)
            imgs = [self.connecting(n, v) for v in Zord]
            BN = self.B_N(n + 1)
            return dim_sum(imgs, BN) - dim_span(BN)
        info["rk_rho1"] = rk_rho(1)
        info["rk_rho2"] = rk_rho(2)
        info["rk_delta1"] = rk_delta(1)
        info["rk_iota2"] = rk_iota
        # exactness
        ex = []
        ex.append(("at H^1_res", info["H1res"] - info["rk_rho1"], 0))
        ex.append(("at H^1", info["H1"] - info["rk_delta1"], info["rk_rho1"]))
        ex.append(("at H^2(N)", info["H2N"] - info["rk_iota2"], info["rk_delta1"]))
        ex.append(("at H^2_res", info["H2res"] - info["rk_rho2"], info["rk_iota2"]))
        info["exactness"] = ex
        return info


# --------------------------------------------- the read-off and theta on H^2(N)

def semilinear_check(cmp_, cl, L, Mres, Mord, dimp, dimm):
    """3) and 4): the read-off map H^2(N) -> Hom(g_+;M^g) (+) Hom(g_-;M^g) is an
    isomorphism, and the connecting map is theta."""
    Mdim = Mres.dim
    dim_g = cmp_.dim_g
    # M^g
    inv = []
    for v in range(1, 1 << Mdim):
        if all(_apply(Mres, k, v) == 0 for k in range(L.dim)):
            inv.append(v)
    dim_inv = rank2_int(list(inv))
    expected = (dimp + dimm) * dim_inv

    # the read-off: psi |-> (psi(gamma_2(x_i)))_{i<dimp} + (psi(gamma_2(y_j^2)))
    mono2 = cmp_.res.mono[2]
    idx2 = {c: a for a, c in enumerate(mono2)}

    def gamma2(i):
        al = [0] * L.dim
        al[i] = 2
        return (frozenset(), tuple(al))

    slots = [idx2[gamma2(i)] for i in range(dimp)]                 # g_+
    slots += [idx2[gamma2(dim_g + j)] for j in range(dimm)]        # g_-^{(2)}

    def readoff(psi):
        w = 0
        for s, a in enumerate(slots):
            blk = (psi >> (a * Mdim)) & ((1 << Mdim) - 1)
            w |= blk << (s * Mdim)
        return w

    ZN = cmp_.Z_N(2)
    BN = cmp_.B_N(2)
    # a basis of the annihilator of M^g inside M^*
    Mgbasis = _echelon(inv)
    ann = kernel([_pair_col(t, Mgbasis) for t in range(Mdim)], Mdim)
    # the subspace of Z^2(N) whose read-off values all lie in M^g
    eqs = []
    for j, z in enumerate(ZN):
        val = 0
        bitpos = 0
        for s, a in enumerate(slots):
            blk = (z >> (a * Mdim)) & ((1 << Mdim) - 1)
            for f in ann:
                if _dotf(f, blk):
                    val |= 1 << bitpos
                bitpos += 1
        eqs.append(val)
    nice_coords = kernel(eqs, len(ZN))
    Znice = []
    for c in nice_coords:
        w = 0
        for j in bits(c):
            w ^= ZN[j]
        Znice.append(w)
    # injectivity: {psi in Z : readoff(psi) = 0} is contained in B
    zeros = []
    eqs0 = [readoff(z) for z in ZN]
    for c in kernel(eqs0, len(ZN)):
        w = 0
        for j in bits(c):
            w ^= ZN[j]
        zeros.append(w)
    dimZ, dimB = rank2_int(list(ZN)), rank2_int(list(BN))
    return dict(dim_inv=dim_inv, expected=expected,
                dim_H2N=cmp_.H_N(2),
                rank_readoff_nice=rank2_int([readoff(v) for v in Znice]),
                every_class_has_nice_rep=(rank2_int(Znice + list(BN)) == dimZ),
                injective=(rank2_int(zeros + list(BN)) == dimB))


def _echelon(vecs):
    piv = span_int(list(vecs))
    return list(piv.values())


def _pair_col(t, basis):
    """the column of the pairing <e_t, v> for v running over `basis`"""
    col = 0
    for i, v in enumerate(basis):
        if (v >> t) & 1:
            col |= 1 << i
    return col


def _dotf(f, v):
    return bin(f & v).count("1") % 2


def theta_check(cmp_, cl, L, Mres, Mord, dimp, dimm, br_g, p2, p4):
    """4): on every ordinary 1-cocycle c, the connecting class equals
    theta(c) = (x |-> c(x^{[2]}) + x.c(x),  y |-> c(y^{[4]}) + y^3.c(y))."""
    Mdim = Mord.dim
    dim_g = cmp_.dim_g
    mono1 = cmp_.ord.mono[1]
    idx1 = {c: a for a, c in enumerate(mono1)}
    mono2 = cmp_.res.mono[2]
    idx2 = {c: a for a, c in enumerate(mono2)}

    def gamma2(i):
        al = [0] * L.dim
        al[i] = 2
        return (frozenset(), tuple(al))

    def cval(z, i):
        """the value c(x_i) as a vector of M, for a 1-cochain z of the ordinary
        complex (basis index a*Mdim+t with a the monomial index)"""
        a = idx1[(frozenset([i]), ())]
        return (z >> (a * Mdim)) & ((1 << Mdim) - 1)

    bad = 0
    tested = 0
    for z in cmp_.ord.Z(1):
        tested += 1
        dz = cmp_.connecting(1, z)
        for i in range(dimp + dimm):
            a = idx2[gamma2(i if i < dimp else dim_g + (i - dimp))]
            got = (dz >> (a * Mdim)) & ((1 << Mdim) - 1)
            if i < dimp:
                # c(x^{[2]}) + x.c(x)
                want = 0
                for k in bits(p2[i]):
                    want ^= cval(z, k)
                want ^= _apply(Mord, i, cval(z, i))
            else:
                j = i - dimp
                want = 0
                for k in bits(p4[j]):
                    want ^= cval(z, k)
                v = cval(z, dimp + j)
                for _ in range(3):
                    v = _apply(Mord, dimp + j, v)
                want ^= v
            if got != want:
                bad += 1
    return tested, bad


def _apply(M, k, v):
    return M.apply(k, v)
