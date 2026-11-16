#!/usr/bin/env python3
"""Which elements carry the obstruction, and the even-size non-alternating case.

Companion of nonalt276.py:
  1) name the generating functions f with (ad_{H_f})^2 not inner, for
     omega_I = (dx)^(2) + sum dp_i dq_i in m = 2n+1 indeterminates;
  2) the derived series of span{H_f} and of h(omega_I);
  3) the same questions for a non-alternating form in an EVEN number of
     indeterminates (sampled among the closed non-degenerate ones).

Run:  python3 extra276.py
"""
import sys, os, time, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nalt2
from nalt2 import (Alg, d2, lie, act, bracket, gram, det, reduce_basis,
                   preserving_algebra, closed_space, alternating, nondeg,
                   derived_series, addform)
from nonalt276 import omega_I, ham_fields, square, in_span, ad_inner, xdeg


def name_obstructions(n):
    m = 2 * n + 1
    alg = Alg(m, (1,) * m)
    om = omega_I(alg, n)
    H = ham_fields(alg, n)
    hbasis = reduce_basis(alg, [D for D in H.values() if any(D)])
    bad = []
    for f, D in H.items():
        if not any(D):
            continue
        if not ad_inner(alg, hbasis, square(alg, D)):
            bad.append(f)
    print("  m=%d: dim span{H_f} = %d;  f with (ad_{H_f})^2 not inner: %s"
          % (m, len(hbasis), [alg.show(f) for f in bad]))
    print("        derived series of span{H_f}: %s"
          % derived_series(alg, hbasis, steps=4))
    pres = preserving_algebra(alg, om)
    print("        dim h(omega) = %d, derived series: %s"
          % (len(pres), derived_series(alg, pres, steps=4)))
    # the square of the offending fields, expressed
    for f in bad:
        D = square(alg, H[f])
        print("        H_%s^2 = %s" % (alg.show(f),
                                       " + ".join("(%s)d%d" % (alg.show(c), i + 1)
                                                  for i, c in enumerate(D) if c)))


def even_case(m, samples=6, seed=276):
    """non-alternating closed non-degenerate forms in an even number of
    indeterminates: is there a 2-structure?"""
    alg = Alg(m, (1,) * m)
    Z = closed_space(alg)
    random.seed(seed)
    print("  m=%d: dim Z^2 = %d" % (m, len(Z)))
    found = 0
    tried = 0
    while found < samples and tried < 4000:
        tried += 1
        om = {}
        for v in Z:
            if random.getrandbits(1):
                addform(om, v)
        if not om or alternating(om) or not nondeg(alg, om):
            continue
        found += 1
        h = preserving_algebra(alg, om)
        hb = reduce_basis(alg, h)
        bad = [D for D in hb if not ad_inner(alg, hb, square(alg, D))]
        print("     sample %d: dim h = %-3d  basis fields with (ad_D)^2 not "
              "inner: %d" % (found, len(hb), len(bad)))


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 78)
    print("1) the obstruction, named")
    print("=" * 78)
    for n in (1, 2):
        name_obstructions(n)
    print("\n" + "=" * 78)
    print("2) non-alternating forms in an even number of indeterminates")
    print("=" * 78)
    even_case(2)
    even_case(4)
    print("\n[%.1f s]" % (time.time() - t0))
