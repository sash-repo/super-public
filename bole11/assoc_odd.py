#!/usr/bin/env python3
"""The odd brackets [[ , ]]_k, k = 1, 3, 5, are not associative by themselves.

For the multiplication f*g = sum_n s_n [[f,g]]_{2n+1} on Pi(F_.), let m be the least index with
s_m != 0.  In the lowest total order 2(2m+1), associativity of * is the associativity of the
single bracket B = [[ , ]]_{2m+1}, up to the normalization of B on each pair of weights.  So it
suffices to show that the two trilinear operators
    (f,g,h) -> B(B(f,g),h)   and   (f,g,h) -> B(f,B(g,h))
are not proportional, already on even f, g, h.  We test it for generic rational weights, for both
sign conventions sigma = +-1 of the odd invariance condition.  Exact rational arithmetic.

    python3 assoc_odd.py
"""
from fractions import Fraction as F
from grc_ops import invariant_ops, apply_op


def bracket(k, w1, w2, sigma):
    keys, B = invariant_ops(k, w1, w2, sigma)
    assert len(B) == 1, (k, w1, w2, sigma, len(B))
    return lambda f, g: apply_op(B[0], keys, f, g)


def proportional(u, v):
    keys = sorted(set(u) | set(v))
    a = [u.get(x, 0) for x in keys]
    b = [v.get(x, 0) for x in keys]
    return all(a[i] * b[j] == a[j] * b[i] for i in range(len(keys)) for j in range(len(keys)))


def main():
    w1, w2, w3 = F(1, 7), F(2, 11), F(3, 13)
    fails = 0
    for k in (1, 3, 5):
        for sigma in (1, -1):
            B12 = bracket(k, w1, w2, sigma)
            B12_3 = bracket(k, w1 + w2 + k, w3, sigma)
            B23 = bracket(k, w2, w3, sigma)
            B1_23 = bracket(k, w1, w2 + w3 + k, sigma)
            L, R = {}, {}
            for a in range(k + 2):
                for b in range(k + 2):
                    for c in range(k + 2):
                        f, g, h = {(a, 0): F(1)}, {(b, 0): F(1)}, {(c, 0): F(1)}
                        for mono, x in B12_3(B12(f, g), h).items():
                            L[(a, b, c) + mono] = x
                        for mono, x in B1_23(f, B23(g, h)).items():
                            R[(a, b, c) + mono] = x
            prop = proportional(L, R)
            print(f"k={k} sigma={sigma:+d}: B(B(f,g),h) and B(f,B(g,h)) proportional: {prop}")
            fails += prop
    print("brackets of orders 1, 3, 5 associative up to normalization:", "none" if fails == 0 else fails)


if __name__ == "__main__":
    main()
