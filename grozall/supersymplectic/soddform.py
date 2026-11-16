"""The index formulas of the three order-2 operators over a periplectic form.

soddexplicit.py fits a coefficient per parity class to the shapes

    C1_{ij} = X^a d_a B_{ij} + (d_i X^a) B_{aj} + (d_j X^a) B_{ia},  X = X_{ptr A}
    C3_{ij} = (d_i pi)(d_j pi') + (d_j pi)(d_i pi')

This script does three further things, and the first of them is the reason it
exists.

1) A PARITY CLASS CAN BE VACUOUS BY ACCIDENT.  At k = 2 four of the 24 classes of
   the C1 ansatz receive no contribution whatever -- not because the term is
   absent, but because sdim (2|2) is too small to produce it.  Fitting there gives
   a table with four holes, and a table with holes is a formula that is WRONG
   wherever the holes are live: the 20-coefficient table fitted at k = 2 misses
   24 of the 72 components at k = 3.  So the fit is done where every class is
   live, and the count of vacuous classes is printed rather than assumed to be 0.

2) It fits at three independent random seeds and requires the same table each
   time, and it fits at k = 2 as well as at k = 3.

3) It then reads the table as a CLOSED SIGN RULE and verifies the rule -- not the
   table -- at k = 2, 3 and 4.  The formula printed in the paper is the rule, so
   the rule is what has to be checked; a hand-reading of a sign table is exactly
   the kind of step this folder has twice been bitten by.

Usage:  python3 soddform.py
"""

import os
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from ssuper import SSetup                                  # noqa: E402
from sodd0 import ptr                                      # noqa: E402
from soddexplicit import comps, fit, verify                # noqa: E402


# --------------------------------------------------------------- the ansatz

def terms_C1(S):
    """L_{X_pi}B with X = X_{ptr A} given, one coefficient per (p_i,p_j,p_a)."""
    A = S.A

    def transport(i, j, X, BC):
        out = []
        v = BC.get((i, j))
        if v is None:
            return out
        for a in range(S.nc):
            if not X[a]:
                continue
            dv = S.dz(v, a)
            if dv:
                out.append(((S.par[i], S.par[j], S.par[a]), A.mul(X[a], dv)))
        return out

    def first(i, j, X, BC):
        out = []
        for a in range(S.nc):
            dX = S.dz(X[a], i)
            v = BC.get((a, j))
            if dX and v:
                out.append(((S.par[i], S.par[j], S.par[a]), A.mul(dX, v)))
        return out

    def second(i, j, X, BC):
        out = []
        for a in range(S.nc):
            dX = S.dz(X[a], j)
            v = BC.get((i, a))
            if dX and v:
                out.append(((S.par[i], S.par[j], S.par[a]), A.mul(dX, v)))
        return out

    return [("transport", transport), ("first", first), ("second", second)]


def terms_C3(S):
    A = S.A

    def t_ij(i, j, pi, pip):
        v = A.mul(S.dz(pi, i), S.dz(pip, j))
        return [((S.par[i], S.par[j]), v)] if v else []

    def t_ji(i, j, pi, pip):
        v = A.mul(S.dz(pi, j), S.dz(pip, i))
        return [((S.par[i], S.par[j]), v)] if v else []

    return [("ij", t_ij), ("ji", t_ji)]


def targets_C1(S, rng, npair, fdeg=3):
    out = []
    for _ in range(npair):
        P = S.random_field(fdeg, rng, nterms=3)
        Q = S.random_field(fdeg, rng, nterms=3)
        pi = ptr(S, P)
        out.append((S.lie(pi, Q), (S.ham(pi), comps(S, Q))))
    return out


def targets_C3(S, rng, npair, fdeg=3):
    A = S.A
    out = []
    for _ in range(npair):
        P = S.random_field(fdeg, rng, nterms=3)
        Q = S.random_field(fdeg, rng, nterms=3)
        pi, pip = ptr(S, P), ptr(S, Q)
        out.append((A.mul(S.dform(pi), S.dform(pip)), (pi, pip)))
    return out


# --------------------------------------------------------------- the rules

def rule_C1(lab, cls):
    """The closed form of the fitted C1 table:

        transport, second :  +1
        first             :  (-1)^{p_j (p_i + p_a)}

    i.e. the classical formula for the Lie derivative of a covariant 2-tensor,
    with ONE Koszul sign, on the middle term.
    """
    pi_, pj, pa = cls
    if lab == "first":
        return Fraction(-1 if (pj * (pi_ + pa)) % 2 else 1)
    return Fraction(1)


def rule_C3(lab, cls):
    """ij : (1/2)(-1)^{p_j(1+p_i)},   ji : (1/2)(-1)^{p_i}."""
    pi_, pj = cls
    if lab == "ij":
        s = -1 if (pj * (1 + pi_)) % 2 else 1
    else:
        s = -1 if pi_ % 2 else 1
    return Fraction(s, 2)


def table_from_rule(rule, labels, slots):
    out = {}
    for lab in labels:
        for mask in range(2 ** slots):
            cls = tuple((mask >> t) & 1 for t in range(slots))
            out[(lab, cls)] = rule(lab, cls)
    return out


# --------------------------------------------------------------- driver

def fit_at(name, k, maker, targeter, slots, seeds, npair, ncls, grow=6):
    """Fit at k, with each seed given ENOUGH targets that no class is vacuous.

    A vacuous class is a hole in the table, and a table with a hole is not a
    formula -- so the number of targets is raised until the fit sees all ncls
    classes, and the run says so if it never does.
    """
    S = SSetup(k, k, form="odd")
    tabs = []
    for seed in seeds:
        n = npair
        for _ in range(grow):
            rng = random.Random(seed)
            c, dead = fit(f"{name} k={k} seed={seed} n={n}", S,
                          targeter(S, rng, n), maker(S), slots, rng,
                          with_dead=True)
            if c is not None and not dead:
                break
            n += 2
        else:
            print(f"   {name} k={k} seed={seed}: {len(dead)} classes are still "
                  f"vacuous at n={n} -- sdim ({k}|{k}) cannot produce them")
        if c is None:
            return S, None
        tabs.append(c)
    same = all(t == tabs[0] for t in tabs)
    print(f"   {name} k={k}: the {len(seeds)} seeds agree: {same} "
          f"({len(tabs[0])} of {ncls} classes seen)")
    return S, (tabs[0] if same else None)


def terms_C1_ordered(S, order):
    """The same ansatz with each of the three products written in either order.

    Swapping the two factors of a product costs a Koszul sign, so "the" formula
    is only defined once the order is fixed; this shows that no order removes the
    sign -- the best is 2 coefficients of -1 out of 24, which is what the formula
    in the paper has.
    """
    A = S.A
    o0, o1, o2 = order

    def pr(p, q, o):
        return A.mul(q, p) if o else A.mul(p, q)

    def transport(i, j, X, BC):
        out = []
        v = BC.get((i, j))
        if v is None:
            return out
        for a in range(S.nc):
            if not X[a]:
                continue
            dv = S.dz(v, a)
            if dv:
                out.append(((S.par[i], S.par[j], S.par[a]), pr(X[a], dv, o0)))
        return out

    def first(i, j, X, BC):
        out = []
        for a in range(S.nc):
            dX, v = S.dz(X[a], i), BC.get((a, j))
            if dX and v:
                out.append(((S.par[i], S.par[j], S.par[a]), pr(dX, v, o1)))
        return out

    def second(i, j, X, BC):
        out = []
        for a in range(S.nc):
            dX, v = S.dz(X[a], j), BC.get((i, a))
            if dX and v:
                out.append(((S.par[i], S.par[j], S.par[a]), pr(dX, v, o2)))
        return out

    return [("transport", transport), ("first", first), ("second", second)]


def order_sweep():
    import itertools
    S = SSetup(3, 3, form="odd")
    print("=== can the Koszul sign be ordered away?  all 8 factor orders, k=3 ===")
    best = []
    for order in itertools.product((0, 1), repeat=3):
        for n in (3, 7, 11):
            rng = random.Random(11)
            c, dead = fit("x", S, targets_C1(S, rng, n),
                          terms_C1_ordered(S, order), 3, rng,
                          verbose=False, with_dead=True)
            if c is not None and not dead:
                break
        nbad = None if c is None else sum(1 for v in c.values() if v != 1)
        print(f"   order {order}: {len(c) if c else 0} classes, "
              f"{nbad} coefficients equal to -1")
        best.append(nbad)
    live = [b for b in best if b is not None]
    print(f"   the minimum over all orders is {min(live)} -- the sign does not "
          f"come from the way the products are written")


def main():
    print("=== C1: the fit ===")
    _, t2 = fit_at("C1", 2, terms_C1, targets_C1, 3, (11, 22, 33), 4, 24)
    _, t3 = fit_at("C1", 3, terms_C1, targets_C1, 3, (11, 22, 33), 3, 24)
    print(f"   k=2 gives {len(t2) if t2 else 0} coefficients, "
          f"k=3 gives {len(t3) if t3 else 0}")
    if t2 and t3:
        common = set(t2) & set(t3)
        agree = all(t2[u] == t3[u] for u in common)
        print(f"   they agree on the {len(common)} classes both see: {agree}")
        print(f"   the classes k=2 cannot see: "
              f"{sorted(u for u in set(t3) - set(t2))}")

    print("\n=== C1: the table, read off at k = 3 ===")
    for (lab, cls), v in sorted(t3.items()):
        print(f"        {lab:10s} {cls} : {v}")

    print("\n=== C1: the closed rule against the table ===")
    rule = table_from_rule(rule_C1, ("transport", "first", "second"), 3)
    ok = all(rule[u] == v for u, v in t3.items())
    print(f"   rule reproduces every fitted coefficient: {ok}")

    print("\n=== C3: the fit ===")
    _, s2 = fit_at("C3", 2, terms_C3, targets_C3, 2, (11, 22, 33), 4, 8)
    _, s3 = fit_at("C3", 3, terms_C3, targets_C3, 2, (11, 22, 33), 3, 8)
    print(f"   k=2 and k=3 tables identical: {s2 == s3}")
    for (lab, cls), v in sorted(s3.items()):
        print(f"        {lab:10s} {cls} : {v}")
    rule3 = table_from_rule(rule_C3, ("ij", "ji"), 2)
    print(f"   rule reproduces every fitted coefficient: "
          f"{all(rule3[u] == v for u, v in s3.items())}")

    print("\n=== the RULES verified on fields they were not fitted to ===")
    allok = True
    for k in (2, 3, 4):
        S = SSetup(k, k, form="odd")
        rng = random.Random(20260905 + 7 * k)
        allok &= verify(f"C1 rule, k={k}", S, rule, terms_C1(S),
                        targets_C1(S, rng, 2))
        allok &= verify(f"C3 rule, k={k}", S, rule3, terms_C3(S),
                        targets_C3(S, rng, 2))
    print(f"\n   ALL RULES VERIFIED: {bool(allok)}\n")
    order_sweep()


if __name__ == "__main__":
    main()
