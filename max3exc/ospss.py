#!/usr/bin/env python3
"""osp_a(4|2): the semi-simple MAX(b) answers of the D21a sweeps (a = 1, 1/3, 5),
tested with casmax.test (Lemma 2.8), and the (5|4), (6|2), (6|6) entries listed."""
import sys, os, pickle, glob
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__))
from check_d21a import D21a
from casmax import test
from sweep3 import Ech, closure
A = {"-1.pkl": F(1), "-1_3.pkl": F(1, 3), "-5.pkl": F(5)}
for f in sorted(glob.glob(os.path.join(HERE, "runs", "D21a*.pkl"))):
    a = next(v for k, v in A.items() if f.endswith(k))
    g = D21a(a)
    d = pickle.load(open(f, "rb"))
    for tags, basis, s, pc, verdict in d["answers"]:
        if verdict != "MAX(b)": continue
        ub = [[F(x) for x in v] for v in basis]
        v, info = test(g, ub)
        print(os.path.basename(f), "a =", a, s, pc, tags[0], "->", v, info, flush=True)
