#!/usr/bin/env python3
r"""Question 2.4 of arXiv:1906.01380: the super analogue of eq. (8).

The composition (7) of the paper,

    (a_k * a_l)(X_1,...,X_{k+l-1}) =
        sum over (l, k-1)-shuffles s of  sign(s) a_k(a_l(X_{s(1)},..,X_{s(l)}), ...),

is superized by replacing sign(s) with the sign sign(s) sign(s,P) of eq. (14)-(15),
i.e., by the very rule that turns (2) into (14).  We verify on the FREE associative
superalgebra (so the identity holds in every associative superalgebra) that with
this convention eq. (8) survives verbatim:

    a_k * a_l = 0            if k, l are even,
              = k a_{k+l-1}  if l is odd,
              = a_{k+l-1}    if l is even and k is odd.
"""

from itertools import combinations
from sant import amult, wmul, wadd, gen, a_word


def shuffle_sign(order, par):
    """sign(s) sign(s,P) for the permutation sending position i to order[i]."""
    s = 1
    n = len(order)
    for i in range(n):
        for j in range(i + 1, n):
            if order[i] > order[j]:                    # an inversion
                s = -s
                if par[order[i]] and par[order[j]]:
                    s = -s
    return s


def star(k, l, par):
    """(a_k * a_l)(X_1,...,X_{k+l-1}) in the free associative superalgebra."""
    m = k + l - 1
    total = {}
    for A in combinations(range(m), l):
        B = [i for i in range(m) if i not in A]
        order = list(A) + B
        eps = shuffle_sign(order, par)
        inner = amult([gen(i) for i in A], [par[i] for i in A], wmul, wadd, {})
        pin = sum(par[i] for i in A) % 2
        args = [inner] + [gen(i) for i in B]
        pars = [pin] + [par[i] for i in B]
        term = amult(args, pars, wmul, wadd, {})
        for w, c in term.items():
            total[w] = total.get(w, 0) + eps * c
    return {w: c for w, c in total.items() if c != 0}


def predicted(k, l, par):
    m = k + l - 1
    a = a_word(list(range(m)), par)
    if k % 2 == 0 and l % 2 == 0:
        return {}
    coeff = k if l % 2 == 1 else 1
    return {w: coeff * c for w, c in a.items()}


if __name__ == "__main__":
    ok = bad = 0
    for k in range(2, 6):
        for l in range(2, 6):
            m = k + l - 1
            if m > 7:
                continue
            for bits in range(1 << m):
                par = [(bits >> i) & 1 for i in range(m)]
                got = star(k, l, par)
                want = predicted(k, l, par)
                if got == want:
                    ok += 1
                else:
                    bad += 1
                    if bad < 6:
                        print("MISMATCH", k, l, par)
                        print("  got ", dict(list(got.items())[:3]))
                        print("  want", dict(list(want.items())[:3]))
            print(f"a_{k} * a_{l}: all {1 << m} parity vectors checked"
                  f"  ({'OK' if bad == 0 else 'FAILED'})")
    print(f"\ntotal: {ok} agreements, {bad} mismatches")
