#!/usr/bin/env python3
"""
v_par.py -- the checks of the PARAMETRIC certificate (param.py, ratfun.py).

Three things are verified, on the types that need the certificate:

  (1) CONTROL.  At random members of a family that are NOT on the exceptional
      locus, the fibre is classified again from scratch over Q, and its answer
      must agree with the specialisation of the generic answer -- same number
      of classes, same multiset of family degrees.  This is an independent
      test of the theorem behind the certificate: it is not used anywhere in
      the proof, and it would fail loudly if the specialisation were wrong.

  (2) SPECIAL MEMBERS.  At every rational root of the polynomials the node
      inverted, the fibre is classified over Q as well.  A member whose answer
      differs from the generic one is a genuine exception and belongs in the
      census as its own line; a member whose answer agrees is one at which the
      certificate merely happened to divide by something that vanishes.

  (3) THE FAMILY OBSTRUCTION, MEASURED.  For each node, dim S(mu) at the
      generic member against dim S* -- the uniformity criterion: the
      stabiliser is constant iff the two agree.

Usage:  python3 -u v_par.py [type,n,m ...]        (default IV_1,3,2)
"""

import random
import sys
import time
from fractions import Fraction as F

import stab
import t3
import orb
import param
import ratfun
from probe import TYPES


def profile(res):
    """what is compared between two members: how many classes, and of what kind"""
    reps, why = res
    if reps is None:
        return None
    return (len(reps), tuple(sorted(fam for _, fam, _ in reps)))


def rational_roots(polys, idx):
    """the rational roots in the parameter `idx` of the factors involving it"""
    out = []
    for g in polys:
        if g.degree_in(idx) == 0:
            continue
        c = param._uni_coeffs(g, idx)
        if c is None or len(c) != 2:
            continue
        out.append(-c[0] / c[1])
    return sorted(set(out))


def run(tname, n, m, D=None, ntest=4):
    p, q, terms = TYPES[tname](n, m)
    if D is None:
        D = 2 * n + 4
    t0 = time.time()
    res = stab.analyse(tname, p, q, terms, D, verbose=False)
    L, classes, cert, notes = t3.classify(res)
    print(f"\n=== {tname}({n},{m}):  {len(classes)} classes, "
          f"{'CERTIFIED' if cert else 'NOT CERTIFIED'}, "
          f"{len(param.NODES)} parametric node(s)   [{time.time()-t0:.1f}s]")
    for nt in notes:
        print("    !", nt)
    ex = param.exceptional()
    print(f"    the run inverted {len(ex)} irreducible polynomials in the "
          f"parameters: " + ", ".join(str(g) for g in ex))

    rnd = random.Random(11)
    agree = differ = failed = 0
    for i, nd in enumerate(param.NODES):
        gen = profile(nd.get("generic"))
        pars = nd.get("params", [])
        loc = param.node_exceptional(i)
        roots = sorted(set(sum([rational_roots(loc, j) for j in pars], [])))
        print(f"\n    node {i} ({nd['kind']}): a {nd['fam']}-parameter family, "
              f"parameters {pars}, fibre of dimension "
              f"{nd.get('m', nd.get('w'))}")
        print(f"        generic answer: {gen[0]} classes, family degrees "
              f"{list(gen[1])}" if gen else "        generic answer: none")
        print(f"        it inverted {len(loc)} polynomials; rational special "
              f"members: {[str(x) for x in roots]}")
        # (1) the control at random ordinary members
        vals = []
        while len(vals) < ntest:
            a = [F(rnd.randint(-9, 9), rnd.choice([1, 2, 3]))
                 for _ in range(nd["fam"])]
            if all(g.eval([a[pars.index(j)] if j in pars else F(1)
                           for j in range(param._NPARAM)]) != 0
                   for g in loc if any(g.degree_in(j) for j in pars)):
                vals.append(a)
        for a in vals:
            pr = profile(nd["run"](a))
            same = (pr == gen)
            agree += same
            failed += (not same)
            print(f"        mu = {[str(x) for x in a]}: "
                  f"{'agrees' if same else 'DIFFERS'}  {pr}")
        # (2) the special members
        for x in roots:
            pr = profile(nd["run"]([x] + [F(0)] * (nd["fam"] - 1)))
            same = (pr == gen)
            differ += (not same)
            print(f"        SPECIAL mu = {x}: "
                  f"{'agrees with the generic answer' if same else 'DIFFERS'}"
                  f"  {pr}")
    print(f"\n    control: {agree} random members agree, {failed} disagree; "
          f"{differ} special members genuinely differ")
    return failed


if __name__ == "__main__":
    specs = sys.argv[1:] or ["IV_1,3,2"]
    bad = 0
    print("backend:", ratfun.backend())
    for s in specs:
        head, _, dd = s.partition(":")
        t, n, m = head.split(",")
        bad += run(t, int(n), int(m), D=int(dd) if dd else None)
    print(f"\nSUMMARY: {bad} disagreements between the generic answer and an "
          f"ordinary member")
