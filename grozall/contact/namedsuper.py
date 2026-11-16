"""The two families over k(1|n), written out and verified as formulas in (lam, mu).

    B_1(F,G) = lam (1 (x) D)(F,G) - mu (D (x) 1)(F,G)                    [n = 1, W = 1]
    B_2(F,G) = -(1/2) sum_i (D_i (x) D_i)(F,G) + mu (RF)G - lam F(RG)    [any n, W = 2]

and B_2 at lam = mu = -1 is the contact bracket itself.  Checked at many weights and
for several n, against the invariant operators the census produces.
"""

import os, sys
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank                                   # noqa: E402

from super import add, scal, mul, kbracket, mono            # noqa: E402
from gensuper import pencil, apply_term, rand_fun           # noqa: E402
from affine import dim_at, ops_at                           # noqa: E402

WEIGHTS = [(Fr(0), Fr(0)), (Fr(1), Fr(2)), (Fr(-1), Fr(-1)), (Fr(2, 3), Fr(-5, 7)),
           (Fr(17, 13), Fr(31, 11)), (Fr(-3), Fr(1, 2))]


def coeff_vector(n, W, lam, mu, rule):
    """The coefficient vector of the asserted formula, in the term basis."""
    terms, _, _, _, _ = pencil(n, W)
    return terms, [rule(t, lam, mu) for t in terms]


def rule1(term, lam, mu):
    S1, s1, S2, s2 = term
    if (S1, s1, S2, s2) == ((), 0, (1,), 0):
        return Fr(lam)
    if (S1, s1, S2, s2) == ((1,), 0, (), 0):
        return -Fr(mu)
    return Fr(0)


def rule2(term, lam, mu):
    S1, s1, S2, s2 = term
    if s1 == 0 and s2 == 1 and not S1 and not S2:
        return -Fr(lam)
    if s1 == 1 and s2 == 0 and not S1 and not S2:
        return Fr(mu)
    if len(S1) == 1 and len(S2) == 1 and S1 == S2 and s1 == s2 == 0:
        return Fr(-1, 2)
    return Fr(0)


def check(n, W, rule, label):
    print("  %s over k(1|%d), W=%d:" % (label, n, W))
    terms, V0, N0, N1, N2 = pencil(n, W)
    ok = True
    for (lam, mu) in WEIGHTS:
        vec = [rule(t, lam, mu) for t in terms]
        d = dim_at(V0, N0, N1, N2, lam, mu)
        basis = ops_at(terms, V0, N0, N1, N2, lam, mu)
        if not any(vec):
            print("     (%s,%s): the formula is zero" % (lam, mu))
            continue
        r0 = rank([list(b) for b in basis]) if basis else 0
        r1 = rank([list(b) for b in basis] + [vec])
        good = (r0 == r1) and d >= 1
        ok = ok and good
        print("     (%s,%s): dim=%d, the formula is invariant: %s" % (lam, mu, d, good))
    return ok


print("=" * 74)
allok = True
allok &= check(1, 1, rule1, "B_1 = lam (1 x D) - mu (D x 1)")
for n in (1, 2, 3):
    allok &= check(n, 2, rule2, "B_2 = -(1/2) sum (D_i x D_i) + mu (R x 1) - lam (1 x R)")

print("=" * 74)
print("B_2 at lam = mu = -1 is the contact bracket:")
for n in (1, 2, 3):
    terms, _, _, _, _ = pencil(n, 2)
    vec = [rule2(t, Fr(-1), Fr(-1)) for t in terms]
    bad = 0
    Fs = [rand_fun(n, p, 900 + i, maxdeg=4) for p in (0, 1) for i in range(3)]
    Gs = [rand_fun(n, p, 950 + i, maxdeg=4) for p in (0, 1) for i in range(3)]
    for F in Fs:
        for G in Gs:
            if not F or not G:
                continue
            out = {}
            for c, t in zip(vec, terms):
                if c:
                    out = add(out, scal(c, apply_term(t, F, G)))
            if add(out, scal(-1, kbracket(n, F, G))):
                bad += 1
    print("   n=%d: %s" % (n, "yes" if bad == 0 else "NO (%d pairs differ)" % bad))
    allok &= (bad == 0)

print("\nall checks passed" if allok else "\nSOMETHING FAILED")
