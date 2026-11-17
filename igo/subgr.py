#!/usr/bin/env python3
"""Calibration of sub.py against Igonin's proved Theorem over Gr_s^n (odd dim
<= n-1, 2 <= s <= n-2): no block of the sub half may carry H^1 except over Gr_2^4.
The last sentence of Remark 4.5.6 of the text."""
import sys
from retract import (named, levi_weyl, levi_modules_up_to, weights, wedge,
                     add_weight, cohomology_of, dims_of)
from bott import omega
from sub import dual, tensor
for n in range(4, 8):
    for s in range(2, n - 1):
        X = named("Gr_%d^%d" % (s, n)); W = levi_weyl(X); r = X.R.rank
        zero = tuple([0] * r); hits = []
        for nm, lam, dV in levi_modules_up_to(X, n - 1):
            V = weights(X, lam); Vd = dual(V)
            for a in range(0, dV + 1):
                La = {zero: 1} if a == 0 else wedge(V, a)
                for tag, base in (("A", La), ("B", tensor(Vd, La))):
                    if tag == "A" and a == 0:
                        continue
                    for j in range(-20, 21):
                        v = dims_of(cohomology_of(X, add_weight(base, omega(r, X.node, j)), W)).get(1, 0)
                        if v:
                            hits.append((tag, a, nm, j, v))
        print("Gr_%d^%d:" % (s, n), hits or "no block with H^1"); sys.stdout.flush()
