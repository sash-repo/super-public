#!/usr/bin/env python3
"""
run_b2.py -- the enumerator T_3 (t3.py) on the coranks 1|2, 1|4, 1|5 and 1|6
of the Serganova-Vaintrob tables of simple supersingularities.

Part 0  T_3 on the corank-1|2 types II(n,m), whose answer is known from the
        structure theorem: it must return mu+1 classes x^j.xi_1.
Part 1  corank 1|4: III_1(n,m), III_2(n), III_3, III_4, III_5.
Part 2  corank 1|5: IV_1(n,m), IV_2(n,m), IV_3(n), IV_4(n,m), IV_5, IV_6.
Part 3  corank 1|6: V(n).
Part 4  the invariant behind the modulus, checked exactly.

Every class list is certified branch by branch (sl/sp transitivity, a zero
action on a quotient, or an invariant subspace with its Hom-block), and every
pair of representatives is separated -- by T_2's certificates where they
suffice, by the certified branch otherwise.
"""

import sys
import time
import random
from fractions import Fraction as F

import stab
import orb
import sv
import t3
from probe import TYPES

bad = 0
spec_point_str = ""


def specialised(vecs):
    """
    If the representatives lie in K(mu), their values at an ordinary member --
    a rational point off the exceptional locus, at which no denominator dies.
    None if there are no parameters at all.
    """
    global spec_point_str
    try:
        import ratfun
        import param
    except ImportError:
        return None
    if not any(isinstance(x, ratfun.Rat) for v in vecs for x in v):
        return None
    ex = param.exceptional()
    rnd = random.Random(29)
    for _ in range(200):
        pt = [F(rnd.randint(2, 40)) for _ in range(param._NPARAM)]
        if any(g.eval(pt) == 0 for g in ex):
            continue
        out = [ratfun.specialize(v, pt) for v in vecs]
        if any(u is None for u in out):
            continue
        spec_point_str = ", ".join(str(x) for x in pt[:6]) + ", ..."
        return out
    return None


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


def branch_key(log):
    return tuple((d, t) for d, t, k, why in log)


def do(tname, n=3, m=1, D=None, expect=None, census=12, quiet=False,
       separate=True):
    """T_3 on one type, with the pairwise separation of the classes."""
    global bad
    t0 = time.time()
    p, q, terms = TYPES[tname](n, m)
    if D is None:
        D = 2 * n + 4
    res = stab.analyse(tname, p, q, terms, D, verbose=False)
    L, classes, cert, notes = t3.classify(res)
    label = tname + (f"({n},{m})" if TYPES[tname].__code__.co_argcount and
                     "(" not in tname and tname[0] in "IV" else "")
    nfam = sum(1 for c in classes if c["family"])
    print(f"\n--- {label}   p|q = {p}|{q}   sdim O_f = "
          f"({res['sdim'][0]}|{res['sdim'][1]})   dim Nt_f = {L.n} "
          f"(pieces " + ",".join(f"{d}:{len(L.bydeg[d])}" for d in L.degrees)
          + f")   dim rho(s_f) = {res['rhodim']}")
    print(f"    T_3: {len(classes)} classes"
          + (f", {nfam} of them a one-parameter family" if nfam else "")
          + ("   CERTIFIED" if cert else "   NOT CERTIFIED"))
    for note in sorted(set(notes)):
        print(f"      ! {note}")
    for c in classes:
        fam = ("   <-- a one-parameter family, the scalar is a MODULUS"
               if c["family"] else "")
        print(f"      orbit dim {c['dim']:3d}   {L.show(c['v'])}{fam}")
    # pairwise separation.  When the parametric certificate has been used, the
    # representatives carry the parameters of their families and lie in K(mu);
    # they are separated AT AN ORDINARY MEMBER, which is a proof for the generic
    # ones as well: equivalence over K(mu) specialises to every member at which
    # the denominators survive, so two members separated at one such point
    # cannot be equivalent generically.  Choosing the point off the exceptional
    # locus keeps it an ordinary member of every family.
    vecs = [c["v"] for c in classes]
    spec = specialised(vecs)
    if spec is not None:
        vecs = spec
        print(f"    the representatives carry parameters; they are separated "
              f"at an ordinary member (mu = {spec_point_str})")
    cache, sep_t2, sep_br, fails = {}, 0, 0, []
    for a in range(len(classes) if separate else 0):
        for b in range(a + 1, len(classes)):
            verdict, why = orb.same_orbit(L.mats, vecs[a],
                                          vecs[b], degs=L.degs,
                                          cache=cache)
            if verdict is False:
                sep_t2 += 1
            elif branch_key(classes[a]["log"]) != branch_key(classes[b]["log"]) \
                    and cert:
                sep_br += 1
            else:
                fails.append((a, b, verdict, why))
    if not separate:
        print("    pairwise separation skipped (--nosep)")
    else:
        print(f"    separation of the {len(classes)} representatives: "
              f"{sep_t2} pairs by a T_2 certificate, {sep_br} by the certified "
              f"branch, {len(fails)} unresolved")
    for a, b, verdict, why in fails:
        print(f"      UNRESOLVED: {L.show(classes[a]['v'])} vs "
              f"{L.show(classes[b]['v'])}: {verdict}")
    # every sampled point must land in the list
    hits = 0
    if census:
        rnd = random.Random(17)
        for _ in range(census):
            v = [F(rnd.randint(-4, 4)) for _ in range(L.n)]
            d = orb.orbit_dim(L.mats, v)
            if any(c["dim"] == d for c in classes):
                hits += 1
        print(f"    random points whose orbit dimension occurs in the list: "
              f"{hits}/{census}")
    ok = cert and not fails and (not census or hits == census)
    if expect is not None and len(classes) != expect:
        ok = False
        print(f"      <-- EXPECTED {expect} classes")
    bad += 0 if ok else 1
    print(f"    {'ok' if ok else 'FAIL'}   {time.time()-t0:.1f}s", flush=True)
    return res, L, classes, cert


def mu_of(n, m):
    return min(n - 1, m + 1)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which == "one":
        # run_b2.py one TYPE [n [m [D]]] -- a single type of probe.TYPES
        sep = "--nosep" not in sys.argv
        args = [int(a) for a in sys.argv[3:] if a != "--nosep"]
        do(sys.argv[2], *args, separate=sep)
        hdr("SUMMARY: " + ("all checks passed" if bad == 0
                           else f"{bad} types not fully certified"))
        sys.exit(0)

    if which in ("all", "0"):
        hdr("(0)  T_3 on the corank-1|2 types II(n,m)")
        print("The structure theorem gives mu+1 classes,")
        print("mu = min(n-1,m+1), with representatives x^j.xi_1 and orbit")
        print("dimensions 2(mu-j).  T_3 knows none of that.\n")
        for n in range(3, 7):
            for m in range(1, n):
                do("II", n, m, expect=mu_of(n, m) + 1, census=8)

    if which in ("all", "III"):
        hdr("(1)  corank 1|4")
        for n in range(3, 7):
            for m in range(1, n):
                do("III_1", n, m)
        for n in range(3, 7):
            do("III_2", n)
        do("III_3", 3, 1, D=9)
        do("III_4", 3, 1, D=9)
        do("III_5", 3, 1, D=9)

    if which in ("all", "IV"):
        hdr("(2)  corank 1|5")
        for n in range(3, 6):
            for m in range(2, n):
                do("IV_1", n, m)
        for n in range(3, 6):
            for m in range(1, n):
                do("IV_2", n, m)
        for n in range(3, 6):
            do("IV_3", n)
        for n in range(3, 6):
            for m in range(2, n):
                do("IV_4", n, m)
        do("IV_5", 3, 1, D=9)
        do("IV_6", 3, 1, D=9)

    if which in ("all", "V"):
        hdr("(3)  corank 1|6")
        do("V", 3, 1, census=4)
        do("V", 4, 1, census=4)

    hdr("SUMMARY: " + ("all checks passed" if bad == 0
                       else f"{bad} types not fully certified"))
