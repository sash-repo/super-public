#!/usr/bin/env python3
"""The groupoid of a generalized root system, part C: two further items.

(1) rank of span(R_re) versus dim V for every GRS in the list.  The r-group generated
    by the affine reflections in the anisotropic walls has a translation lattice of
    rank = rank span(R_re), so it is crystallographic only when the real roots span V.
    Claim to be tested: rank span(R_re) = dim V - 1 exactly for A(m,n) and C(n)
    (the type I superalgebras, whose even part has a one-dimensional centre) and
    = dim V for all the others.

(2) The defective GRS A(1,1) = psl(2|2) and Serganova's Remark 6.3: the sets
    obtained from linear orderings of eps_1,eps_2,delta_1,delta_2 have 2m+1 = 3
    elements, but do they satisfy Definition 6.1?  We test the uniqueness clause.
"""
from fractions import Fraction as F
from itertools import combinations, product

from check_iv_a import (dot, rank, solve, sysA, sysA11, sysB, sysC, sysD, sysD21,
                        sysF4, sysG3)


def re_span(sysd):
    G, roots = sysd["G"], sysd["roots"]
    Vb = sysd["Vbasis"] or [tuple(F(1) if i == j else F(0) for j in range(len(G))) for i in range(len(G))]
    dim = len(Vb)
    re = [r for r in roots if dot(G, r, r) != 0]
    im = [r for r in roots if dot(G, r, r) == 0]
    # ranks computed inside V: express in Vbasis coordinates
    rre = rank([solve(Vb, r) for r in re]) if re else 0
    rim = rank([solve(Vb, r) for r in im]) if im else 0
    print(f"{sysd['name']:22s} dim V = {dim}   rank span(R_re) = {rre}   "
          f"rank span(R_im) = {rim}   defect = {dim - rre}")
    return dim, rre


def a11_remark63():
    """Serganova's Remark 6.3 for m = 1: the 4! linear orderings of eps_1,eps_2,delta_1,delta_2."""
    # ambient: eps_1,eps_2,delta_1,delta_2 with (eps,eps)=1, (delta,delta)=-1
    G4 = [[F(0)] * 4 for _ in range(4)]
    for i in range(2):
        G4[i][i] = F(1)
    for k in (2, 3):
        G4[k][k] = F(-1)
    # V = the orthogonal complement of sum(eps) and sum(delta); project
    def proj(v):
        # subtract the components along sum(eps) and sum(delta)
        se = (F(1), F(1), F(0), F(0))
        sd = (F(0), F(0), F(1), F(1))
        out = list(v)
        for s in (se, sd):
            c = dot(G4, v, s) / dot(G4, s, s)
            out = [x - c * y for x, y in zip(out, s)]
        return tuple(out)
    sym = [0, 1, 2, 3]                     # eps_1, eps_2, delta_1, delta_2
    def e(i):
        return tuple(F(1) if j == i else F(0) for j in range(4))
    # the GRS in the plane, in the coordinates a = eps_1-eps_2, b = delta_1-delta_2
    a = proj(tuple(x - y for x, y in zip(e(0), e(1))))
    b = proj(tuple(x - y for x, y in zip(e(2), e(3))))
    basis = [a, b]
    def coords(v):
        return tuple(solve(basis, proj(v)))
    roots = set()
    for i, j in product(range(4), repeat=2):
        if i == j:
            continue
        r = tuple(x - y for x, y in zip(e(i), e(j)))
        if (i < 2) == (j < 2):             # even root
            roots.add(coords(r))
        else:                              # odd root
            roots.add(coords(r))
    roots = sorted(roots)
    print(f"\nA(1,1): |R| = {len(roots)} roots in the plane: {roots}")

    def reps(B, r, bound=5):
        out = []
        for c in product(range(-bound, bound + 1), repeat=len(B)):
            if not (all(x >= 0 for x in c) or all(x <= 0 for x in c)):
                continue
            v = tuple(sum(F(c[i]) * B[i][j] for i in range(len(B))) for j in range(2))
            if v == r:
                out.append(c)
        return out

    # the 24 orderings of Remark 6.3
    triples = set()
    for perm in product(*[range(4)] * 4):
        if len(set(perm)) != 4:
            continue
        B = tuple(coords(tuple(x - y for x, y in zip(e(perm[k]), e(perm[k + 1]))))
                  for k in range(3))
        triples.add(tuple(sorted(B)))
    print(f"sets from the 24 linear orderings, deduplicated: {len(triples)}")
    uniq_ok = weak_ok = 0
    for B in triples:
        Bl = list(B)
        nrep = [len(reps(Bl, r)) for r in roots]
        if all(n == 1 for n in nrep):
            uniq_ok += 1
        if all(n >= 1 for n in nrep):
            weak_ok += 1
    print(f"  of them satisfying Definition 6.1 verbatim (unique representation): {uniq_ok}")
    print(f"  of them satisfying it with 'unique' dropped (some representation):  {weak_ok}")
    B0 = sorted(triples)[0]
    print(f"  sample: B = {list(B0)};  representations of {roots[0]}: {reps(list(B0), roots[0])[:4]}")
    # an explicit positive relation in such a triple
    for B in triples:
        Bl = list(B)
        for c in product(range(0, 4), repeat=3):
            if sum(c) == 0:
                continue
            v = tuple(sum(F(c[i]) * Bl[i][j] for i in range(3)) for j in range(2))
            if v == (F(0), F(0)):
                print(f"  positive relation in B = {Bl}: coefficients {c} sum to zero")
                return


def main():
    print("--- (1) real-root span ---")
    for s in [sysA(1, 0), sysA(2, 0), sysA(2, 1), sysA(3, 1), sysA11(), sysB(1, 1),
              sysB(2, 1), sysB(1, 2), sysC(2), sysC(3), sysD(2, 1), sysD(2, 2),
              sysD21(1), sysD21(3), sysG3(), sysF4()]:
        re_span(s)
    print("\n--- (2) A(1,1) and Serganova's Remark 6.3 ---")
    a11_remark63()
    multipliable()


def multipliable():
    """Which roots gamma have 2*gamma also a root?  Needed for Theorem 8.11."""
    print("\n--- (3) multipliable roots (2*gamma in R) ---")
    for s in [sysA(1, 0), sysA(2, 1), sysA11(), sysB(1, 1), sysB(2, 1), sysB(1, 2), sysC(2),
              sysC(3), sysD(2, 1), sysD(2, 2), sysD21(1), sysG3(), sysF4()]:
        G, roots = s["G"], s["roots"]
        rs = set(roots)
        re2 = [r for r in roots if dot(G, r, r) != 0 and tuple(2 * x for x in r) in rs]
        im2 = [r for r in roots if dot(G, r, r) == 0 and tuple(2 * x for x in r) in rs]
        print(f"{s['name']:22s} multipliable real: {len(re2):2d}   multipliable imaginary: {len(im2)}")


if __name__ == "__main__":
    main()
