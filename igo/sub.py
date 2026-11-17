#!/usr/bin/env python3
"""The other half of T_p: the subsheaf E^* (x) Lambda^{p+1}(E), Theorem 4.5.4 of the text.

retract.py computes H^1 of the quotient Theta_X (x) Lambda^p(E) only.  The
degree-p even derivations of Lambda(E) form an extension
    0 -> E^* (x) Lambda^{p+1}(E) -> T_p -> Theta_X (x) Lambda^p(E) -> 0,
so splitness needs H^1 of the sub as well.  Over LGr_n at most one non-character
constituent V fits under 2n-1, so E^* (x) Lambda^q(E) is a sum of the blocks
    A(a, j) = Lambda^a(V) (x) d^j,      B(a, j) = V^* (x) Lambda^a(V) (x) d^j.
This lists every block with H^1 != 0, for |j| <= 30.
"""
import sys
from retract import (named, levi_weyl, levi_modules_up_to, weights, wedge,
                     add_weight, cohomology_of, dims_of, direct_sum)
from bott import omega


def dual(m):
    return {tuple(-x for x in mu): c for mu, c in m.items()}


def tensor(m1, m2):
    out = {}
    for a, c in m1.items():
        for b, e in m2.items():
            k = tuple(x + y for x, y in zip(a, b))
            out[k] = out.get(k, 0) + c * e
    return out


def run(name, jr=range(-30, 31)):
    X = named(name)
    W = levi_weyl(X)
    r = X.R.rank
    zero = tuple([0] * r)
    n = int(name.split("_")[1])
    print("%s:" % name)
    got = [j for j in jr
           if dims_of(cohomology_of(X, {tuple(omega(r, X.node, j)): 1}, W)).get(1, 0)]
    print("  d^j alone: H^1 != 0 for j in %s" % (got or "none"))
    for nm, lam, dV in levi_modules_up_to(X, 2 * n - 1):
        V = weights(X, lam)
        Vd = dual(V)
        for a in range(0, dV + 1):
            La = {zero: 1} if a == 0 else wedge(V, a)
            for tag, base in (("A", La), ("B", tensor(Vd, La))):
                if tag == "A" and a == 0:
                    continue
                hits = []
                for j in jr:
                    v = dims_of(cohomology_of(
                        X, add_weight(base, omega(r, X.node, j)), W)).get(1, 0)
                    if v:
                        hits.append((j, v))
                if hits:
                    print("  %s(a=%d), V=%s: %s" % (tag, a, nm, hits))
    sys.stdout.flush()


if __name__ == "__main__":
    for n in range(2, 8):
        run("LGr_%d" % n)
