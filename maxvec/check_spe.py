#!/usr/bin/env python3
"""Is spe^a(n) + C.E a maximal subalgebra of sl(n|n)?  (the 0th component of
the new row sle(n) + C.E < svect(n|n)).  Exact; output to stdout.
Usage: check_spe.py [n ...]  (default 2 3 4)."""
import sys
from fractions import Fraction as F
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from run_cartan import preserving, form_odd, sl_fields, euler_field
from run_case3 import is_maximal, sdim_of
from run_case3b import independent
from run_pairs import intersect_layers

def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)

for n in ([int(x) for x in sys.argv[1:]] or [2, 3, 4]):
    ne = no = n
    sl = independent(sl_fields(ne, no), ne, no)
    pa = preserving(ne, no, form_odd(ne, no, -1))
    E = euler_field(ne, no)
    # spe^a + C.E  =  (pe^a + C.E) cap sl   (E in sl since m = n)
    cpa = independent(list(pa) + [E], ne, no)
    u0 = intersect_layers({0: cpa}, {0: sl}, ne, no, 0)[0]
    v, dq, dA = is_maximal(u0, sl, ne, no)
    say(f"n={n}: dim sl({n}|{n}) = {len(sl)}, dim pe^a = {len(pa)}, "
        f"dim (c(pe^a) cap sl) = {len(u0)}, dim of the quotient = {dq}")
    say(f"      {v}")
