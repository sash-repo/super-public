#!/usr/bin/env python3
"""compare.py -- the two invariants the paper selects models by, across every
model computed: the number of weight-0 tensors in degree 2 (the "scalar
curvatures", whose linear combination is the left-hand side of eq. (7), so that
the parameter runs over CP^{s-1}), and the copies of the Minkowski space."""
import re, sys, glob

def cells(log):
    out, par = {}, None
    for ln in open(log):
        m = re.match(r'Par = \((\w+)\)', ln)
        if m:
            par = m.group(1); out[par] = {}
        m = re.match(r'\s+(-?\d+)\s{2,}(\S.*?)\s{2,}\s*(\S.*)?$', ln)
        if m and par:
            out[par][int(m.group(1))] = m.group(2).strip()
    return out

allc = {}
for f in sys.argv[1:]:
    allc.update(cells(f))

print(f"{'Par':<14} {'N':>2} {'deg range':<12} {'zero-wt in deg 2':>16}  {'CP^{s-1}'}")
for par, d in allc.items():
    N = len(par) - 4 if par.count('0') == 4 else par.count('0') - 0
    N = len(par) - 4
    degs = [k for k, v in d.items() if v not in ('---', 'not defined')]
    c2 = d.get(2, '')
    m = re.search(r'\(0+\)(?:\^(\d+))?', c2)
    s = 0 if not m else int(m.group(1) or 1)
    rng = f"{min(degs)}..{max(degs)}" if degs else "-"
    print(f"({par})".ljust(15) + f"{N:>2} {rng:<12} {s:>16}  "
          + ("CP^%d" % (s - 1) if s else "none -- (B) fails"))
