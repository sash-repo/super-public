#!/usr/bin/env python3
"""
v_aff.py -- the affine certificate against the shortcut it replaces.

For a type given on the command line, T_3 is run BOTH ways -- with the affine
treatment of each degree (classify_affine) and with the quotient-by-T
shortcut (classify(..., affine_mode=False)) -- and the two answers are
confronted:

  * each representative of the old list is matched, by T_2's orbit test, to a
    representative of the new one; a partition that lost nothing must match
    every one of them;
  * a representative of the new list matched by no old one is a class the
    shortcut MISSED, and is printed as such;
  * random points of the layer are matched against the new list, which is the
    completeness test the dimension count only approximates.

Usage:  python3 -u v_aff.py III_1,3,2 [III_1,4,2 ...]
"""
import sys
import random
from fractions import Fraction as F

import stab, orb, t3
from probe import TYPES


def match(mats, degs, v, reps, cache):
    """
    (index, verdict) for the rep whose orbit contains v.  T_2's test returns
    True, False or None (undecided); a definite True is preferred, and an
    undecided one is reported as such rather than silently taken for a match.
    """
    undecided = None
    for i, r in enumerate(reps):
        verdict, why = orb.same_orbit(mats, v, r, degs=degs, cache=cache)
        if verdict is True:
            return i, True
        if verdict is None and undecided is None:
            undecided = i
    return (undecided, None) if undecided is not None else (None, False)


def confront(tname, n, m, D=None, samples=25):
    p, q, terms = TYPES[tname](n, m)
    if D is None:
        D = 2 * n + 4
    res = stab.analyse(tname, p, q, terms, D, verbose=False)
    L, new, cnew, nnew = t3.classify(res)
    _, old, cold, nold = t3.classify(res, affine_mode=False)
    print(f"\n=== {tname}({n},{m})   dim Nt_f = {L.n}")
    print(f"    shortcut: {len(old)} classes, "
          f"{sum(1 for c in old if c['family'])} families, "
          f"{'certified' if cold else 'NOT certified: ' + '; '.join(sorted(set(nold)))}")
    print(f"    affine  : {len(new)} classes, "
          f"{sum(1 for c in new if c['family'])} families, "
          f"{'certified' if cnew else 'NOT certified: ' + '; '.join(sorted(set(nnew)))}")
    nv = [c["v"] for c in new]
    cache = {}
    hit = set()
    for c in old:
        i, verdict = match(L.mats, L.degs, c["v"], nv, cache)
        if i is None:
            print(f"    !! the old class {L.show(c['v'])} (dim {c['dim']}) "
                  f"matches NO class of the new list")
        else:
            hit.add(i)
            tag = "" if verdict is True else "   (T_2 undecided)"
            print(f"    old {L.show(c['v'])!r:44s} -> new {L.show(nv[i])}{tag}")
    for i, c in enumerate(new):
        if i not in hit:
            print(f"    ++ MISSED BY THE SHORTCUT: {L.show(c['v'])} "
                  f"(orbit dim {c['dim']})")
    # completeness: random points must all land in the new list
    rnd = random.Random(23)
    lost = 0
    for _ in range(samples):
        v = [F(rnd.randint(-4, 4)) for _ in range(L.n)]
        i, verdict = match(L.mats, L.degs, v, nv, cache)
        if i is None:
            lost += 1
            print(f"    ?? the random point {L.show(v)} matches no class")
    print(f"    {samples - lost}/{samples} random points of the layer matched "
          f"a class of the new list")
    return len(old), len(new)


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        head, _, dd = spec.partition(":")
        t, n, m = head.split(",")
        confront(t, int(n), int(m), D=int(dd) if dd else None)
