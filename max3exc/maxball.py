#!/usr/bin/env python3
"""Run the highest-weight maximality test of maxb.py on EVERY answer of the
G3/F4 sweeps flagged MAX(b).  Verdict MAX if: u_ev semisimple of rank = rank(u cap h)
(checked: derived algebra of u_ev = u_ev), every u_ev-highest weight space of g/u
is a line, and each such line generates g together with u."""
import sys, os, pickle, glob
from fractions import Fraction as F
from maxb import *
from sweep3 import derived

def uev_basis(g, E):
    # parity projection of u's basis (u is parity-homogeneous)
    return Ech(g.N, [[x if g.PAR[i] == 0 else F(0) for i, x in enumerate(v)] for v in E.basis()]).basis()

res = {}
for f in sorted(glob.glob(os.path.join(LM, "runs", "[GF]*.pkl"))):
    if "hom" in f: continue
    which = "G3" if "G3" in f else "F4"
    g = exc3.build_G3() if which == "G3" else exc3.build_F4()
    d = pickle.load(open(f, "rb"))
    for tags, basis, s, pc, verdict in d["answers"]:
        if verdict != "MAX(b)": continue
        ub = [[F(x) for x in v] for v in basis]
        E, tor = analyse(g, ub, "%s %s %s %s" % (os.path.basename(f), s, pc, tags[0]))
        ue = uev_basis(g, E)
        der = Ech(g.N, [g.br(x, y) for x in ue for y in ue]).dim()
        ss = der == len(ue)
        ep, hw = hw_spaces(g, E, tor)
        mults = [len(m) for w, m in hw]
        ok = ss and all(m == 1 for m in mults)
        gens = []
        if ok:
            for w, mod in hw:
                gens.append(closure(g, E.basis() + mod).dim())
            ok = all(x == g.N for x in gens)
        print("   u_ev dim %d semisimple %s, #pos roots %d, hw mults %s, closures %s  ==> %s"
              % (len(ue), ss, len(ep), mults, gens, "MAX" if ok else "UNDECIDED"))
        res.setdefault((which, s), set()).add("MAX" if ok else "UNDECIDED")
print("\nSUMMARY")
for k, v in sorted(res.items()): print(k, v)
