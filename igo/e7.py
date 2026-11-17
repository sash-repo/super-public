#!/usr/bin/env python3
"""The cotangent retract over the space E = E_7/P_7, the one space of the census
of cot.py that is run separately (its Levi Weyl group W(E_6) has 51840 elements).

Prints, for p = 1, 2, 3, the dimensions of H^q(X; Omega^p (x) Theta_X) and of
their G-invariant parts.  For Statement 4.4.3 of the text: p = 1 gives
H^0 = C and H^1 = g (case I); p = 2 gives dim H^1 = 1, all of it invariant;
p = 3 gives H^1 = 0.
Takes about 15 seconds.
"""
from bott import chss_list
from cot import levi_weyl, dims, invariants

X = [Y for Y in chss_list() if Y.typ == "E" and Y.rank == 7][0]
W = levi_weyl(X)
print("%s: dim X %d |W_l| %d" % (X.name, X.dimX, len(W)))
for p in (1, 2, 3):
    print(" p=%d dims %s inv %s" % (p, dims(X, p, W), invariants(X, p, W)))
