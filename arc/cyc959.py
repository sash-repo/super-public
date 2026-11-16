"""Where the constant -3 of [Ra] comes from.

Radul's Subsec. 5.2 prints  S_3(G) = -3 m^{3/2} (v_1(v_2), v_3) alpha^{1/2},
whereas the constant that continues the normalization of his own rows n <= 2 is
-1 (run3.py).  Radul's pages 40-46 fix every ingredient
of the formula from his own print -- the contact form (1.2), the frame 1.4, the
multiplier m and the matrix (a_ij) of n. 1.4, the modules F_lambda = {a alpha^l}
of n. 1.4 -- and they agree with those of the text.  So the difference is in the constant alone, and
the question is what a 3 can count.

The candidate: the expression is cyclically symmetric in (1,2,3), so a
definition that sums over the three cyclic positions is 3 times the one that
picks one.  Checked here, together with the whole symmetry type of
(v_i(v_j), v_k), for an honest finite contact diffeomorphism of S^{1|3}.

Run:  ../venv/bin/python cyc959.py   (or system python3; sympy is enough)
"""
import sys
import sympy as sp

import radul as R
from radul import (e, d, eadd, esub, emul, escal, eeq, einv, epow,
                   Diffeo, Composite, multiplier, D_op, clean)
from test_radul import gen
from run import amatrix, vfield, apply_to_row, sprod, S3_frame, cocycle

OK = BAD = 0


def check(name, ok):
    global OK, BAD
    if ok:
        OK += 1
        print("  ok   %s" % name, flush=True)
    else:
        BAD += 1
        print("  FAIL %s" % name, flush=True)


def main():
    R.setup(3)
    R.set_order(2)
    n = 3
    G = Diffeo(gen(3), e, 3)
    m = multiplier(G, n)
    mi = einv(m)
    a = amatrix(G, n)
    v = [vfield(a[i], n) for i in range(n)]
    print("frame built for a finite contact diffeomorphism of S^{1|3}",
          flush=True)

    # T[i][j][k] = (v_i(v_j), v_k)
    vv = [[apply_to_row(v[i], a[j], n) for j in range(n)] for i in range(n)]
    T = [[[sprod(vv[i][j], a[k], n) for k in range(n)]
          for j in range(n)] for i in range(n)]

    base = T[0][1][2]                       # (v_1(v_2), v_3), the Schwarzian
    check("the Schwarzian (v_1(v_2), v_3) is not identically zero", bool(base))

    # 1) cyclic symmetry
    check("(v_1(v_2), v_3) = (v_2(v_3), v_1)", eeq(base, T[1][2][0]))
    check("(v_2(v_3), v_1) = (v_3(v_1), v_2)", eeq(T[1][2][0], T[2][0][1]))

    # 2) the odd permutations: antisymmetric in the first two by Comment 4)
    for (i, j, k) in ((1, 0, 2), (0, 2, 1), (2, 1, 0)):
        check("(v_%d(v_%d), v_%d) = -(v_1(v_2), v_3)" % (i + 1, j + 1, k + 1),
              eeq(T[i][j][k], escal(-1, base)))

    # 3) so the whole tensor is the totally antisymmetric one, and every
    #    repeated-index component is NOT in it: those are the (v_i(v_i), v_k),
    #    which Comment 4) expresses through m.
    cyc = eadd(T[0][1][2], T[1][2][0], T[2][0][1])
    check("the cyclic sum is 3 (v_1(v_2), v_3)", eeq(cyc, escal(3, base)))
    alt = eadd(*[escal(1, T[i][j][k]) for (i, j, k) in
                 ((0, 1, 2), (1, 2, 0), (2, 0, 1))] +
               [escal(-1, T[i][j][k]) for (i, j, k) in
                ((1, 0, 2), (0, 2, 1), (2, 1, 0))])
    check("the full antisymmetrization is 6 (v_1(v_2), v_3)",
          eeq(alt, escal(6, base)))

    # 4) what the diagonal components are -- they are not the Schwarzian: by
    #    Comment 4) and (v_i, v_i) = m^{-1} they are expressed through m alone.
    check("(v_1(v_1), v_1) = -(m^{-2}/2) v_1(m)",
          eeq(T[0][0][0],
              escal(sp.Rational(-1, 2), emul(mi, emul(mi, v[0](m))))))

    # 5) the constant, once more and in this file: the linearization
    want = D_op(1)(D_op(2)(D_op(3)(gen(3))))
    got = clean({S: sp.expand(sp.diff(c, e).subs({e: 0, d: 0}))
                 for S, c in S3_frame(G, 3, 1).items()})
    ratios = set()
    for S in set(list(got) + list(want)):
        x, y = got.get(S), want.get(S)
        if y in (None, 0):
            ratios.add(sp.nan if x else sp.Integer(0))
        else:
            ratios.add(sp.simplify(sp.cancel((x or 0) / y)))
    ratios.discard(sp.Integer(0))
    print("     m^{3/2}(v_1(v_2),v_3) linearized / D_1D_2D_3(g) = %s"
          % sorted(ratios, key=str), flush=True)
    check("one constant continues the rows n <= 2, and it is -1",
          len(ratios) == 1 and sp.simplify(list(ratios)[0] + 1) == 0)
    check("the cyclic sum would therefore carry -3",
          len(ratios) == 1 and sp.simplify(3 * list(ratios)[0] + 3) == 0)

    # 6) both constants give a cocycle of weight 1/2, and of no other weight:
    #    the cocycle identity is homogeneous, so it cannot decide the constant.
    H = Diffeo(gen(3, seed=3), d, 3)
    for c in (-1, -3):
        S3 = lambda X, c=c: S3_frame(X, 3, c)
        check("S_3 with the constant %d is a 1-cocycle of weight 1/2" % c,
              cocycle(S3, sp.Rational(1, 2), 3, G, H))

    print("\n%d ok, %d FAILED" % (OK, BAD))
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
