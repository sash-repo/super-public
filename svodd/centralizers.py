#!/usr/bin/env python3
"""
centralizers.py -- the centralizer Z_x of an odd element x of least centralizer
(a regular odd element) in five series, computed exactly in the matrix
realizations of regodd.py, against the closed formulas

    sl(m|n), m>n :      sdim Z_x = ((m-n)^2+n-1 | n)
    osp(2|2n)    :      sdim Z_x = (2n^2-3n+2 | 1)
    osp(1|2n)    :      sdim Z_x = (2n^2-n | 0)
    osp(2m+1|2m), osp(2m|2m) :  sdim Z_x = (m | m)

Run:  python3 centralizers.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from regodd import osp, sl, regular_sdim

bad = 0


def check(name, g, want):
    global bad
    z = tuple(regular_sdim(g))
    ok = z == tuple(want)
    bad += 0 if ok else 1
    print("  %-14s sdim g = (%d|%d)   Z_x = (%d|%d)   formula (%d|%d)   %s"
          % ((name,) + tuple(g.sdim()) + z + tuple(want) + ("OK" if ok else "MISMATCH",)),
          flush=True)


print("sl(m|n), m>n")
for m in range(2, 6):
    for n in range(1, m):
        check("sl(%d|%d)" % (m, n), sl(m, n), ((m - n) ** 2 + n - 1, n))
print("osp(2|2n)")
for n in (2, 3, 4, 5):
    check("osp(2|%d)" % (2 * n), osp(2, 2 * n), (2 * n * n - 3 * n + 2, 1))
print("osp(1|2n)")
for n in (1, 2, 3, 4):
    check("osp(1|%d)" % (2 * n), osp(1, 2 * n), (2 * n * n - n, 0))
print("osp(2m+1|2m)")
for m in (1, 2, 3):
    check("osp(%d|%d)" % (2 * m + 1, 2 * m), osp(2 * m + 1, 2 * m), (m, m))
print("osp(2m|2m)")
for m in (1, 2, 3):
    check("osp(%d|%d)" % (2 * m, 2 * m), osp(2 * m, 2 * m), (m, m))
print("SUMMARY:", "all agree" if bad == 0 else "%d MISMATCHES" % bad)
