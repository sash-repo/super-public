#!/usr/bin/env python3
"""
Addendum to check_v.py: the signed graph of a generalized root system.

Claim (Lemma 9.11 of the text, a corollary of the finiteness of W (Proposition 8.5) and of
Theorem 9.8): in a generalized root system no two real roots of
opposite sign are joined, i.e. (alpha,alpha) > 0 > (beta,beta) forces
(alpha,beta) = 0.  Hence the sign eps is constant on every connected component
of the graph of real roots, and the signed graphs of Section 9 that actually occur in
Lie superalgebra theory are Popov's unsigned ones on each component.

Verified by brute force on the systems of the table of Subsec. 8.8 plus the defective A(1,1).
"""

from fractions import Fraction as F
from check_iv_a import (dot, sysA, sysA11, sysB, sysC, sysD, sysD21, sysF4, sysG3)

systems = [sysA(1, 0), sysA(2, 0), sysA(2, 1), sysA(3, 1), sysB(1, 1), sysB(2, 1),
           sysB(1, 2), sysC(2), sysC(3), sysD(2, 1), sysD(2, 2), sysD21(1), sysD21(2),
           sysD21(3), sysG3(), sysF4(), sysA11()]

bad_total = 0
print(f"{'system':<24} {'#re+':>5} {'#re-':>5} {'#mixed pairs':>13} {'#nonorth':>9}")
for s in systems:
    G, roots = s["G"], s["roots"]
    re_pos = [a for a in roots if dot(G, a, a) > 0]
    re_neg = [a for a in roots if dot(G, a, a) < 0]
    mixed, nonorth = 0, 0
    for a in re_pos:
        for b in re_neg:
            mixed += 1
            if dot(G, a, b) != 0:
                nonorth += 1
    bad_total += nonorth
    print(f"{s['name']:<24} {len(re_pos):>5} {len(re_neg):>5} {mixed:>13} {nonorth:>9}")

print()
print("no real roots of opposite sign are ever joined:", bad_total == 0)
print("(so eps is constant on each connected component of the real-root graph,")
print(" as the finiteness of W and Theorem 9.8 predict)")
