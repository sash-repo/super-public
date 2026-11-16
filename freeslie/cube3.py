#!/usr/bin/env python3
"""cube3.py -- is 3*alpha ever a root?

Companion to freelie.py.  In characteristic 3 the map

        T(x) = [x, [x, x]]      (x odd)

is additive and 3-semilinear (freelie.py, Lemma), so a root-graded Lie
superalgebra satisfies the identity T = 0 on the whole odd part as soon as T
kills every odd ROOT vector; and T kills the root vector of an odd root alpha
automatically whenever 3*alpha is not a root, because T(g_alpha) is contained
in g_{3 alpha}.

This script runs that test over the sample of refl.py: for every algebra, and
every odd root alpha, it asks whether 3*alpha is a root.  Where the answer is
"never", the two candidate definitions of a Lie superalgebra in characteristic
3 -- with and without the identity [x,[x,x]] = 0 -- agree on that algebra.
"""
import sys
import os

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from refl import DATA                                          # noqa: E402


def root_parity(root, parity):
    return sum(c * p for c, p in zip(root, parity)) % 2


def main():
    print("algebra                              p  #roots  #odd roots  3a a root?  2a a root?")
    print("-" * 88)
    bad = []
    n_odd = n_two = 0
    for name, alg in DATA.items():
        p = alg["p"]
        pos = alg["pos"]
        parity = alg["parity"]
        roots = set(pos) | {tuple(-c for c in r) for r in pos}
        odd = [r for r in roots if root_parity(r, parity)]
        three = [r for r in odd if tuple(3 * c for c in r) in roots]
        two = [r for r in odd if tuple(2 * c for c in r) in roots]
        print("%-36s %2d  %6d  %10d  %10s  %10s"
              % (name, p, len(roots), len(odd),
                 "NO" if not three else "YES: %s" % three[:3],
                 "no" if not two else "yes (%d)" % len(two)))
        if three:
            bad.append(name)
        n_odd += len(odd)
        n_two += 1 if two else 0
    print()
    print("entries: %d;  odd roots inspected: %d;  entries with 2a a root for some odd a: %d"
          % (len(DATA), n_odd, n_two))
    if bad:
        print("3*alpha IS a root for an odd alpha in: %s" % ", ".join(bad))
    else:
        print("For every algebra of the sample and every odd root alpha, 3*alpha is NOT a root.")
        print("Hence T(g_alpha) is contained in g_{3 alpha} = 0 for every odd root vector, and,")
        print("T being additive, T = 0 on the whole odd part: the identity [x,[x,x]] = 0 holds")
        print("in each of them whether or not it is imposed.")
    p3 = [n for n, a in DATA.items() if a["p"] == 3]
    print()
    print("The characteristic 3 members of the sample are: %s" % ", ".join(p3))


if __name__ == "__main__":
    main()
