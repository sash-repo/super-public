"""
The cocycle on vect^L(1|2) written WITHOUT the embedding: the display

    c(X_1, X_2) = (-1)^{p(X_1)} Res( i(X_1) i(S(X_2)) ),

what i is here, and how to define this pairing correctly.

The row vect^L(1|2) of Table (61) prints the same cocycle twice: as
(-1)^{p(X_1)} Res(g_1 ~g_2' - g_2 ~g_1'), i.e., in the coordinates (x|theta_1,
theta_2) themselves, and as Res(i(X_1) d_t i(X_2)) through the embedding i of
eq. (28).  The display belongs to the FIRST shape -- the one the map
S was introduced for -- so the i in it is one too many, and the pairing in it
is the residue pairing of the odd components alone.

What is computed here, over every pair of basis vector fields with |deg x| <= 4
(11664 pairs for each claim; the kernel is read off the sub-range |deg x| <= 3,
in which every partner of a field is still in the list):

 1) Res( i(X_1) d_t i(X_2) ) = (-1)^{p(X_1)} Res( g_1 ~g_2' - g_2 ~g_1' ),
    the two shapes Table (61) prints -- the sign and the normalization of
    eq. (28) included, and with Res the BEREZIN residue, i.e., the coefficient
    of x^{-1} theta_1 theta_2;
 2) with  S(X) := g_2' d_theta_1 - g_1' d_theta_2  and with

        (X, Y) := Res( sum_i g_i(X) g_i(Y) )        [the odd components alone]

    one has c(X_1, X_2) = (-1)^{p(X_1)} (X_1, S(X_2));
 3) what may be claimed of (-,-): it is EVEN, SUPERSYMMETRIC, and DEGENERATE,
    its kernel being exactly the d_x-part -- so it is the residue pairing of the
    odd components and is NOT an invariant form on vect^L(1|2);
 4) S is parity-preserving, S^2 = 0, and Ker S is eq. (100).

Exact arithmetic throughout (rationals over the monomials x^a theta_S).
"""
import sys
from itertools import product

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from check_bases import make_i, vect_basis, name_vect, par_vect
from check_cocycles import L_res
from stringy import mono, add, mul, smul, d_t, d_odd

N = 2
DEG = 4
i_dx, i_dth = make_i(N)

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1
        print("  ok   %s %s" % (name, extra))
    else:
        BAD += 1
        print("  FAIL %s %s" % (name, extra))
    sys.stdout.flush()
    return cond


def emb(tok):
    kind, m, j, T = tok
    return i_dx(m, T) if kind == 'dx' else i_dth(m, j, T)


def comps(tok):
    """(f; g_1, g_2) of the token, as elements of Lambda(theta)[x^{-1},x]."""
    kind, m, j, T = tok
    f = mono(m, tuple(T))
    if kind == 'dx':
        return f, {}, {}
    return ({}, f, {}) if j == 1 else ({}, {}, f)


def RES(h):
    """the Berezin residue on (x|theta_1,theta_2): coefficient of x^{-1} th1 th2."""
    return h.get((-1, (1, 2)), 0)


def S_map(A):
    """S(X) = g_2' d_theta_1 - g_1' d_theta_2, on a triple (f; g_1, g_2)."""
    f, g1, g2 = A
    return {}, d_t(g2), smul(-1, d_t(g1))


def pairing(A, B):
    """(X, Y) := Res( sum_i g_i(X) g_i(Y) ): the odd components alone."""
    _, a1, a2 = A
    _, b1, b2 = B
    return RES(add(mul(a1, b1), mul(a2, b2)))


def bracket(A, B):
    """[X, Y] on vect^L(1|2), X = f d_x + sum g_i d_theta_i, as a triple."""
    fa, g1a, g2a = A
    fb, g1b, g2b = B
    ga, gb = (g1a, g2a), (g1b, g2b)

    def act(F, comp):
        """(f d_x + sum g_i d_th_i) applied to the function comp."""
        out = mul(F[0], d_t(comp))
        for j in (1, 2):
            out = add(out, mul(F[1 + j - 1], d_odd(j, comp)))
        return out

    # parities: p(X) = p(f) = p(g_i) + 1 on a homogeneous token
    def par(F):
        for c in F:
            for (m, T), v in c.items():
                return len(T) % 2 if c is F[0] else (len(T) + 1) % 2
        return 0

    pa, pb = par(A), par(B)
    sgn = (-1) ** (pa * pb)
    f = add(act(A, fb), smul(-sgn, act(B, fa)))
    g1 = add(act(A, g1b), smul(-sgn, act(B, g1a)))
    g2 = add(act(A, g2b), smul(-sgn, act(B, g2a)))
    return f, g1, g2


if __name__ == "__main__":
    print("=" * 70)
    print("the cocycle on vect^L(1|2) without the embedding")
    print("=" * 70)
    toks = vect_basis(N, DEG)
    print("  %d basis fields, %d pairs per claim\n" % (len(toks), len(toks) ** 2))

    bad1 = bad2 = bad3 = 0
    ex1 = ex2 = None
    for a, b in product(toks, toks):
        pa = par_vect(a)
        lhs = L_res(2 * N, mul(emb(a), d_t(emb(b))))        # Res(i(X1) d_t i(X2))
        A, B = comps(a), comps(b)
        _, g1a, g2a = A
        _, g1b, g2b = B
        tab = (-1) ** pa * RES(add(mul(g1a, d_t(g2b)),
                                   smul(-1, mul(g2a, d_t(g1b)))))
        if lhs != tab:
            bad1 += 1
            ex1 = ex1 or (name_vect(a), name_vect(b), lhs, tab)
        viaS = (-1) ** pa * pairing(A, S_map(B))
        if lhs != viaS:
            bad2 += 1
            ex2 = ex2 or (name_vect(a), name_vect(b), lhs, viaS)
        if lhs != pairing(A, S_map(B)):
            bad3 += 1

    check("1) Res(i(X_1) d_t i(X_2)) = (-1)^{p(X_1)} Res(g_1 ~g_2' - g_2 ~g_1')",
          bad1 == 0, "(%d mismatches; %s)" % (bad1, ex1 or ""))
    check("2) the same = (-1)^{p(X_1)} (X_1, S(X_2)),  (X,Y) := Res(sum g_i g_i)",
          bad2 == 0, "(%d mismatches; %s)" % (bad2, ex2 or ""))
    check("   and the sign (-1)^{p(X_1)} is not absorbed by the pairing",
          bad3 != 0, "(%d mismatches without it)" % bad3)

    # ---------------------------------------------------- the pairing itself
    anti = sym = even = 0
    for a, b in product(toks, toks):
        A, B = comps(a), comps(b)
        s = (-1) ** (par_vect(a) * par_vect(b))
        if pairing(A, B) != -s * pairing(B, A):
            anti += 1
        if pairing(A, B) != s * pairing(B, A):
            sym += 1
        if pairing(A, B) != 0 and (par_vect(a) + par_vect(b)) % 2:
            even += 1
    check("3) (-,-) is ANTI-symmetric: (X,Y) = -(-1)^{p(X)p(Y)}(Y,X)",
          anti == 0, "(%d failures)" % anti)
    check("   and not symmetric", sym != 0, "(%d failures)" % sym)
    check("   and EVEN: it vanishes on a pair of opposite parities",
          even == 0, "(%d failures)" % even)

    sub = vect_basis(N, DEG - 1)
    ker = [a for a in sub if all(pairing(comps(a), comps(b)) == 0 for b in toks)]
    dxs = [a for a in sub if a[0] == 'dx']
    check("   and DEGENERATE: its kernel is exactly the d_x-part",
          sorted(map(str, ker)) == sorted(map(str, dxs)),
          "(kernel %d of the %d fields with |deg x| <= %d, the d_x-part being %d)"
          % (len(ker), len(sub), DEG - 1, len(dxs)))

    # is it invariant?  ([Z,X],Y) + (-1)^{p(Z)p(X)} (X,[Z,Y]) = 0 ?
    bad_inv = 0
    small = vect_basis(N, 1)
    for z, a, b in product(small, small, small):
        Z, A, B = comps(z), comps(a), comps(b)
        pz, pa = par_vect(z), par_vect(a)
        lhs = pairing(bracket(Z, A), B) + (-1) ** (pz * pa) * pairing(A, bracket(Z, B))
        if lhs != 0:
            bad_inv += 1
    check("   and NOT invariant", bad_inv != 0,
          "(%d of the %d triples with |deg x| <= 1 break invariance)"
          % (bad_inv, len(small) ** 3))

    # --------------------------------------------------------------- the map S
    bad_p = 0
    for a in toks:
        _, s1, s2 = S_map(comps(a))
        for g in (s1, s2):
            for (m, T), c in g.items():
                if (len(T) + 1) % 2 != par_vect(a):
                    bad_p += 1
    check("4) S is parity-preserving", bad_p == 0, "(%d failures)" % bad_p)

    bad_sq = 0
    for a in toks:
        _, g1, g2 = comps(a)
        SSA = S_map(S_map(comps(a)))
        want = (smul(-1, d_t(d_t(g1))), smul(-1, d_t(d_t(g2))))
        if (SSA[1], SSA[2]) != want:
            bad_sq += 1
    check("   S^2 = -d_x^2 on the odd components, so S is no differential",
          bad_sq == 0, "(%d failures)" % bad_sq)

    kerS = [a for a in toks if not S_map(comps(a))[1] and not S_map(comps(a))[2]]
    ok_ker = all(a[0] == 'dx' or a[1] == 0 for a in kerS)
    check("   Ker S = {f d_x} + {g_i(theta) d_theta_i}, i.e., eq. (100)",
          ok_ker, "(%d fields of the %d)" % (len(kerS), len(toks)))

    print("\n%d ok, %d FAILED" % (OK, BAD))
    sys.exit(1 if BAD else 0)
