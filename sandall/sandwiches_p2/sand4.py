#!/usr/bin/env python3
"""
Sandwich elements over F_4 (hence a first step beyond the prime field towards the
algebraically closed K).

For x = a + w b with a, b in L(F_2) and w a generator of F_4 (w^2 = w + 1),
    (ad_x)^2 = [(ad_a)^2 + (ad_b)^2] + w [(ad_b)^2 + ad_a ad_b + ad_b ad_a],
so, the two F_2-components being independent,
    (ad_x)^2 = 0  <==>  (ad_a)^2 = (ad_b)^2  and  ad_a ad_b + ad_b ad_a = (ad_b)^2.
We enumerate the pairs (a, b) by bucketing a by the value of (ad_a)^2, which makes
the search over F_4^15 feasible.  The second sandwich condition
[[L,x],[L,x]] = 0 is then imposed over F_4 directly.
"""

import sys
import glob
import os
from css import LieF2, rref, in_span
from thin import algebra_from_thin_table, load_table

N = None


def matvec(cols, v, n):
    r = 0
    for k in range(n):
        if (v >> k) & 1:
            r ^= cols[k]
    return r


def compose(c1, c2, n):
    """the matrix of x -> c1(c2(x))"""
    return [matvec(c1, c2[j], n) for j in range(n)]


def key(cols):
    n = len(cols)
    k = 0
    for j, c in enumerate(cols):
        k |= c << (j * n)
    return k


def sandwiches_f4(L, verbose=True):
    n = L.n
    ads = {}
    sq = {}
    for a in range(1, 1 << n):
        cols = L.ad(a)
        ads[a] = cols
        sq[a] = key(compose(cols, cols, n))
    buckets = {}
    for a, s in sq.items():
        buckets.setdefault(s, []).append(a)
    tot = sum(len(v) ** 2 for v in buckets.values())
    if verbose:
        print("   buckets: %d, pairs to test: %d" % (len(buckets), tot))
    found = []
    for s, elts in buckets.items():
        for a in elts:
            ca = ads[a]
            for b in elts:
                cb = ads[b]
                # ad_a ad_b + ad_b ad_a == (ad_b)^2 ?
                ok = True
                for j in range(n):
                    x = matvec(ca, cb[j], n) ^ matvec(cb, ca[j], n) ^ \
                        matvec(cb, cb[j], n)
                    if x:
                        ok = False
                        break
                if ok:
                    found.append((a, b))
    return found


def f4_span_dim(pairs, n):
    """dimension over F_4 of the span of {a + w b}: represent an F_4-vector as a
    pair (a, b) of F_2-vectors; the F_4-span is an F_2-subspace of L + L closed
    under multiplication by w: (a,b) -> (b, a+b)."""
    vecs = []
    for (a, b) in pairs:
        v = a | (b << n)
        w = b | ((a ^ b) << n)          # w*(a + w b) = w a + w^2 b = b*1? see below
        vecs.append(v)
        vecs.append(w)
    return rref(vecs)


def main(paths):
    for path in sorted(paths):
        name = os.path.basename(path)[:-4]
        L = algebra_from_thin_table(load_table(path), name)
        print("=" * 70)
        print(name, "dim", L.n)
        pairs = sandwiches_f4(L)
        # keep only those that also satisfy [[L,x],[L,x]] = 0 over F_4
        good = []
        n = L.n
        for (a, b) in pairs:
            if a == 0 and b == 0:
                continue
            # [L, x] over F_4 = the F_4-span of ([e_j,a], [e_j,b]) pairs
            rows = []
            for j in range(n):
                u = L.mul(1 << j, a)
                v = L.mul(1 << j, b)
                rows.append(u | (v << n))
                # multiplication by w: (u,v) -> (v, u+v)
                rows.append(v | ((u ^ v) << n))
            R = rref(rows)
            ok = True
            for p in R:
                for q in R:
                    a1, b1 = p & ((1 << n) - 1), p >> n
                    a2, b2 = q & ((1 << n) - 1), q >> n
                    # (a1 + w b1)(a2 + w b2) = [a1,a2] + w([a1,b2]+[b1,a2]+[b1,b2])
                    #                          + [b1,b2]   (since w^2 = w+1)
                    re = L.mul(a1, a2) ^ L.mul(b1, b2)
                    im = L.mul(a1, b2) ^ L.mul(b1, a2) ^ L.mul(b1, b2)
                    if re or im:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                good.append((a, b))
        print("   sandwiches over F_4 (non-zero): %d" % len(good))
        S = f4_span_dim(good, L.n)
        print("   their F_4-span has F_2-dimension %d (i.e. F_4-dimension %s)"
              % (len(S), len(S) / 2))


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("css-bases/thin15/*.txt")))
