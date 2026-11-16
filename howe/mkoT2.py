"""
mkoT2.py -- Kochetkov's Theorem 2 (singular vectors for M_{a,b}(n)), family by family.

usage:  python3 mkoT2.py <n> [family ...]
        families: 2a 2b 2c 2d 3a 3b 3c 4 5
"""

import sys
from fractions import Fraction as F

from mab import scan_mab, lift
from mimod import Mn


def hdr(t):
    print("\n-- " + t)


def run(n, which):
    Z = [0] * (n - 1)
    E = [0] * (n - 2) + [1] if n >= 2 else [1]

    if '2a' in which:
        hdr("2a)  v h.w. (0,...,0; lam+2a),  wt y = (0,...,0; lam),  a w = -(na-nb+lam) v"
            "\n     y = (sum 'p_i'd_i)(x)v + sum_{p<q} 'd_p'd_q (x) w_{p,q} + D (x) w")
        for (a, b) in [(1, 0), (1, 1), (1, 2), (2, 1), (n, -1), (1, -1)]:
            for lm in [-2, 0, 1, 3]:
                scan_mab(n, Z, lm + 2 * a, 2, a, b, want=(tuple(Z), F(lm)))

    if '2b' in which:
        hdr("2b)  nb + a = 0 (SM(n); b=-1, a=n):  v_n h.w. ((0,..,0,1); lam+2n),"
            "\n     wt y = ((0,..,0,1); lam),  n w = -(3n-1+lam) v")
        for lm in [-2, 0, 1, 3]:
            scan_mab(n, E, lm + 2 * n, 2, n, -1, want=(tuple(E), F(lm)))

    if '2c' in which:
        hdr("2c)  2a + nb = 0:  v_{n-1} h.w. ((0,..,0,1); -2b),  wt y = ((0,..,0,2); 2(b-a))"
            "\n     y = sum_{i<=n-1} 'p_i'd_n (x) v_i + sum 'd_i'd_n (x) w_i")
        a, b = n, -2                      # 2a + nb = 2n - 2n = 0
        scan_mab(n, E, -2 * b, 2, a, b, want=(tuple([0] * (n - 2) + [2]), F(2 * (b - a))))
        print("    ... and the whole degree-2 layer of that module:")
        scan_mab(n, E, -2 * b, 2, a, b)

    if '2d' in which and n == 2:
        hdr("2d)  n=2:  y = 'd1'd2 (x) v + D (x) w;  a+b != 0: v h.w. (0; 2a+2b), wt y = (0;0);"
            "\n     a+b = 0: v h.w. (lam; 0), wt y = (lam; 0)")
        for (a, b) in [(1, 0), (1, 1), (1, 2), (2, -1)]:
            scan_mab(2, [0], 2 * a + 2 * b, 2, a, b, want=((0,), F(0)))
        for lm in [0, 1, 2]:
            scan_mab(2, [lm], 0, 2, 1, -1, want=((lm,), F(0)))

    if '3a' in which:
        hdr("3a)  a != 0:  v_n h.w. ((0,..,0,1); (2n-1)b+3a),  wt y = ((0,...,0); 2nb)"
            "\n     y = sum 'p_i'p_j'd_i (x) v_j + sum 'p_i'd_i'd_j (x) w_j"
            "\n         + sum 'p_i D (x) v_{i,0} + sum 'd_i D (x) w_{i,0}")
        for (a, b) in [(1, 0), (1, 1), (1, 2), (n, -1)]:
            scan_mab(n, E, (2 * n - 1) * b + 3 * a, 3, a, b, want=(tuple(Z), F(2 * n * b)))

    if '3b' in which and n == 2:
        hdr("3b)  n=2, a=1, b=0:  w_1 h.w. (1; 3),  wt y = (0; 0)")
        scan_mab(2, [1], 3, 3, 1, 0, want=((0,), F(0)))

    if '3c' in which:
        hdr("3c)  a != 0:  w h.w. ((0,...,0); 2a),  wt y = ((0,...,0); -(a+b))"
            "\n     y = (sum 'p_i'd_i'd_n) (x) w + 'd_n D (x) v")
        for (a, b) in [(1, 0), (1, 1), (1, 2), (n, -1)]:
            scan_mab(n, Z, 2 * a, 3, a, b, want=(tuple(Z), F(-(a + b))))

    if '4' in which:
        hdr("4)  b=0, a=1, m=4:  v h.w. ((0,..,0,1); 3),  wt y = ((0,..,0,1); -1)")
        scan_mab(n, E, 3, 4, 1, 0, want=(tuple(E), F(-1)))

    if '5' in which:
        hdr("5)  a=0, b=1 (B(n)):  the degree-3 and -4 vectors are the degree-1 and -2 ones"
            "\n     for Le(n) with D appended to every term; the weight does not change.")
        # Le(n)-family 1c) of the 1989 paper = 1b) here with k = n: 'd_n (x) m1 at
        # V of pe-h.w. (0,...,0,lam+1); appending D should give a degree-3 vector.
        for lm in [-1, 1]:
            mu = [0] * (n - 2) + [-(lm + 1)] if n >= 2 else [0]
            nu = lm + 1
            scan_mab(n, mu, nu, 3, 0, 1)


if __name__ == '__main__':
    n = int(sys.argv[1])
    which = sys.argv[2:] or ['2a', '2b', '2c', '2d', '3a', '3b', '3c', '4']
    print("=== Kochetkov's Theorem 2 (M_{a,b}(%d)) ===" % n)
    run(n, which)
