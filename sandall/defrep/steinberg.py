#!/usr/bin/env python3
"""The Steinberg sl(3)-module L^(2,2) for p = 3: H^1(sl(3); M (x) M^*), by parity and
weight.  Also the modules L^(0,1), L^(1,0), L^(0,2), L^(1,1), for which H^1 = 0."""
from h1gen import h1

sl3 = [[2, -1], [-1, 2]]
for lam in ([0, 1], [1, 0], [0, 2], [1, 1], [2, 2]):
    h1(sl3, [0, 0], 3, lam, "end")
