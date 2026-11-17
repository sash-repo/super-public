#!/usr/bin/env python3
"""torus.py -- the two facts the formula 2^{(dim L - dim t)/2} needs
(Lemma 2.2.1 and Proposition 4.1.1 of the text):

  1) the maximal torus t of g = o^{(1)}_I(2n+1) stays a maximal torus of the
     restricted closure L, so that the same dim t may be used on both sides;
  2) the desuperization of oo_{IPi}(1|2n) is o_G(2n+1) for the form
     G = diag(1, Pi_{2n}), which is the identity form up to a change of basis
     (Lebedev's normal shapes) -- this is what identifies the "method 2"
     superization of the series.

For 1) it suffices to look inside the centralizer C_L(t): a torus containing t
lies there, and an element of C_L(t) is either in t or has a nilpotent square.
The sweep below is over the F_2-points of C_L(t), which is small.

Output: torus.log
"""

import itertools

from wk import (GF, Space, flat, mat_mul, mat_zero, oI, order_of,
                restricted_closure, triangular)

LOG = [None]


def out(s=""):
    print(s, flush=True)
    if LOG[0]:
        LOG[0].write(s + "\n")
        LOG[0].flush()


def toral_elements(alg, sub):
    """the x in the span of sub (over F_2) with x^2 = x as matrices."""
    F = alg.F
    tor = []
    for coefs in itertools.product([0, 1], repeat=len(sub)):
        if not any(coefs):
            continue
        M = mat_zero(alg.m)
        for c, B in zip(coefs, sub):
            if c:
                for i in range(alg.m):
                    for j in range(alg.m):
                        if B[i][j]:
                            M[i][j] = F.add(M[i][j], B[i][j])
        if mat_mul(F, M, M) == M:
            tor.append(list(coefs))
    return tor


def main():
    LOG[0] = open("torus.log", "w")
    F = GF(1)
    for n in (1, 2, 3):
        alg, deg = oI(F, n)
        alg.build_struct()
        neg, tor, pos = triangular(alg, deg)
        L, odd = restricted_closure(alg)
        L.build_struct()
        out("n = %d: dim g = %d, dim L = %d, dim t = %d"
            % (n, alg.n, L.n, len(tor)))
        # the centralizer of t in L
        tmats = [alg.basis[i] for i in tor]
        cent = []
        for i in range(L.n):
            if all(not any(any(r) for r in _comm(F, L.basis[i], T))
                   for T in tmats):
                cent.append(i)
        out("   dim C_L(t) = %d" % len(cent))
        sub = [L.basis[i] for i in cent]
        tel = toral_elements(L, sub)
        # a torus is an abelian subalgebra of toral elements; count how many
        # toral elements there are and whether they all lie in t
        tspace = Space(F, L.m * L.m)
        for T in tmats:
            tspace.add(flat(T))
        outside = 0
        for coefs in tel:
            M = mat_zero(L.m)
            for c, B in zip(coefs, sub):
                if c:
                    for i in range(L.m):
                        for j in range(L.m):
                            if B[i][j]:
                                M[i][j] = F.add(M[i][j], B[i][j])
            if flat(M) not in tspace:
                outside += 1
        out("   nonzero toral elements of C_L(t) over F_2: %d, of which %d lie"
            " outside t" % (len(tel), outside))
        out("   (t has %d nonzero F_2-points)" % (2 ** len(tor) - 1))
        out("   => t is a maximal torus of L: %s" % (outside == 0))
        # the (2,4)-parities: the odd part of L is g_-, of dim 2n
        out("   sdim s(g) = (%d|%d);  oo_{IPi}(1|%d) predicts (%d|%d)"
            % (sum(1 for o in odd if not o), sum(1 for o in odd if o),
               2 * n, n * (2 * n + 1), 2 * n))
        out()
    LOG[0].close()


def _comm(F, A, B):
    from wk import mat_comm
    return mat_comm(F, A, B)


if __name__ == "__main__":
    main()
