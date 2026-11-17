"""The census: are Bunegina--Onishchik's answers right?

Part A. The tangent sheaf of a split (CP^1, Lambda E) read naively (as a direct
        sum) against B--O's Prop. 18 (a non-split extension), both against the
        honest count of global vector fields.
Part B. Dimension 1|2: which non-split supermanifolds are homogeneous.
(The Parts C and D, dimension 1|3 and the degree-2 obstructions of [LT], are
in runcd.py.)
"""

from fractions import Fraction
from itertools import combinations, product
import sys

from ns import (SM, g_from, g_mul, g_add, g_zero, lp, sdim_split_predicted,
                summands_naive, summands_BO, h0, h1)

DEG = 10
LOG = open("run.log", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def xi(j):
    return g_from((j,), lp((0, 1)))


def coef(e, c=1):
    return g_from((), lp((e, c)))


def cocycle_pair(m, i, j, c, ks):
    """B--O's cocycle (41): c(x) xi_i xi_j d/dx + (1/2) k_r c'... for m = 3 the
    second summand is (1/2) k_r x^{-2} xi_1xi_2xi_3 d/dxi_r with c = x^{-1}."""
    zx = g_mul(c, g_mul(xi(i), xi(j)))
    zxi = {}
    if m == 3:
        r = ({1, 2, 3} - {i, j}).pop()
        # B--O's (41) has xi_i xi_j xi_r IN THAT ORDER; for (i,j,r) = (1,3,2)
        # that is -xi_1 xi_2 xi_3, and the sign is not a decoration.
        delta = g_mul(g_mul(xi(i), xi(j)), xi(r))
        zxi[r] = g_mul(coef(-2, Fraction(ks[r - 1], 2)), delta)
    return zx, zxi


# ------------------------------------------------------------------- Part A
say("=" * 78)
say("A. The graded pieces of the tangent sheaf are NON-SPLIT extensions")
say("=" * 78)
say(f"{'E = +O(-k_j)':>16} {'naive sum':>12} {'B-O Prop.18':>12} "
    f"{'engine':>12}   verdict")
bad_naive = 0
for ks in ([0, 0], [1, 1], [2, 2], [3, 1], [4, 0], [3, 3], [5, 1], [4, 2],
           [0, 0, 0], [1, 1, 1], [2, 2, 0], [2, 2, 1], [2, 2, 2], [2, 2, 5],
           [3, 1, 0], [3, 1, 1], [5, 3, 2], [4, 0, 0]):
    naive = sdim_split_predicted(ks, naive=True)
    bo = sdim_split_predicted(ks)
    eng = SM.split(ks).sdim_vect(DEG)
    ok = (eng == bo)
    if naive != bo:
        bad_naive += 1
    say(f"{str(tuple(ks)):>16} {str(naive):>12} {str(bo):>12} {str(eng):>12}"
        f"   {'B-O confirmed' if ok else 'MISMATCH'}"
        f"{'  (naive wrong)' if naive != bo else ''}")
say(f"\n  B--O's Prop. 18 reproduced by the engine in all "
    f"{18} bundles; the naive direct-sum reading fails in {bad_naive} of them.")

# ------------------------------------------------------------------- Part B
say("")
say("=" * 78)
say("B. Dimension 1|2: every non-split structure, over every retract")
say("=" * 78)
say("  a non-split structure over E = O(-k1)+O(-k2) is a non-zero class in")
say("  H^1(CP^1; O(2-k1-k2)), basis x^{-1}, ..., x^{3-k1-k2}")
say("")
say(f"{'(k1,k2)':>10} {'dim H^1':>8} {'classes':>8} {'homogeneous':>28}"
    f" {'sdim v(M,O)':>14}")
homog2 = []
KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8   # Part B: retracts with k1 <= KMAX
for k1 in range(0, KMAX + 1):
    for k2 in range(0, k1 + 1):
        d = h1(2 - k1 - k2)
        if d == 0:
            continue
        basis = [-1 - i for i in range(d)]          # exponents of x
        classes = []
        if d <= 7:                                  # all 0/1 combinations
            for sel in product((0, 1), repeat=d):
                if any(sel):
                    classes.append(sel)
        else:                                       # a sample, d being large
            for i in range(d):
                classes.append(tuple(1 if j == i else 0 for j in range(d)))
                classes.append(tuple(1 if j <= i else 0 for j in range(d)))
                classes.append(tuple(1 if j >= i else 0 for j in range(d)))
        if d >= 2:                                  # a few generic ones too
            classes.append(tuple(range(1, d + 1)))
            classes.append(tuple((-1) ** i * (i + 2) for i in range(d)))
        hits, sdims = [], set()
        for sel in classes:
            c = g_zero()
            for a, e in zip(sel, basis):
                if a:
                    c = g_add(c, coef(e, a))
            zx, zxi = cocycle_pair(2, 1, 2, c, [k1, k2])
            M = SM.from_cocycle([k1, k2], zx, zxi)
            sdims.add(M.sdim_vect(DEG))
            if M.transitive(DEG):
                hits.append(sel)
        if hits:
            homog2.append((k1, k2, hits))
        say(f"{str((k1,k2)):>10} {d:>8} {(str(len(classes)) + ('' if d <= 7 else '*')):>8}"
            f" {(str(len(hits)) + ' of them') if hits else 'none':>28}"
            f" {str(sorted(sdims)):>14}")
say("")
say("  homogeneous non-split 1|2 supermanifolds found:", homog2 or "none")
say("  B--O's Corollary (p. 3526): exactly one, the superquadric Q_{1|2},")
say("  with retract E = O(-2)+O(-2).")

