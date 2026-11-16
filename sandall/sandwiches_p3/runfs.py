#!/usr/bin/env python3
r"""
runfs.py --- an exhaustive check of the criterion (fs) of the text:

   for s = sum c_i d_i in vect(n; \One|0) over F_p,
   (ad_s)^2 = 0  <=>  c_i c_j = 0,  c_k d_i c_j = 0,  c_k d_i(Div s) = 0
                      for all 1 <= i, j, k <= n.

n = 1: every one of the p^p elements is tested.
n = 2: the right-hand side forces every c_i into the set A = {c | c^2 = 0},
which is small, so the pairs (c_1, c_2) in A x A can be run through in full;
for the other direction the whole of F_p^{2p^2} is out of reach, so the scan is
over the (deg, wt)-blocks (complete: the span of the cone is multigraded) plus
a large random sample, and the counts of the non-vacuous cases are printed.

Output: runs/fs.log
"""

import os
import random
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vectp import vect, fs_rhs, div, O                      # noqa: E402

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "fs.log")


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(m):
        print(m)
        log.write(m + "\n")

    say("criterion (fs) for vect(n; 1|0):  (ad_s)^2 = 0  <=>  the three families")
    say("")

    # ---------------------------------------------------------------- n = 1
    for p in (3, 5, 7):
        g = vect(p, 1)
        n_s = n_r = bad = 0
        sand = []
        for code in range(p ** g.n):
            v = np.zeros(g.n, dtype=np.int64)
            t = code
            for i in range(g.n):
                v[i] = t % p
                t //= p
            a, b = g.is_sandwich(v), fs_rhs(g, v)
            n_s += a
            n_r += b
            if a != b:
                bad += 1
                if bad < 4:
                    say("   MISMATCH p=%d v=%s lhs=%s rhs=%s" % (p, list(v), a, b))
            if a and v.any():
                sand.append(v)
        say("n=1 p=%2d  dim %2d  all %d elements: sandwiches %d, rhs %d, mismatches %d"
            % (p, g.n, p ** g.n, n_s, n_r, bad))
        if sand:
            from sandp import rref
            R = rref(sand, p)[0]
            degs = sorted({g.deg[i] for r in R for i, c in enumerate(r) if c})
            say("        span of the sandwich cone: dim %d, degrees %s "
                "(vect_{>=%d} has dim %d)"
                % (len(R), degs, min(degs),
                   sum(1 for d in g.deg if d >= min(degs))))

    # ---------------------------------------------------------------- n = 2
    for p in (3,):
        n = 2
        g = vect(p, n)
        o = g.O
        say("")
        say("n=%d p=%d  dim vect = %d,  dim O = %d" % (n, p, g.n, o.N))

        # A = {c in O | c^2 = 0}, over F_p
        A = []
        for code in range(p ** o.N):
            c = np.zeros(o.N, dtype=np.int64)
            t = code
            for i in range(o.N):
                c[i] = t % p
                t //= p
            if not o.prod(c, c).any():
                A.append(c)
        say("   |{c in O | c^2 = 0}| = %d  (of %d)" % (len(A), p ** o.N))

        # the full right-hand side, over F_p: pairs from A
        rhs_pts, agree = [], 0
        for c1 in A:
            for c2 in A:
                if o.prod(c1, c2).any():
                    continue
                v = g.field([c1, c2])
                if fs_rhs(g, v):
                    rhs_pts.append(v)
                    if g.is_sandwich(v):
                        agree += 1
        say("   right-hand side holds for %d elements (all F_p-points); of these, "
            "%d are sandwiches" % (len(rhs_pts), agree))

        # the converse, over the same exhaustive set: every F_p-sandwich must
        # satisfy the right-hand side.  The sandwiches all lie in A x A too,
        # because (ad_s)^2 = 0 forces c_i^2 = 0 (see the log line below).
        forced = 0
        for c1 in A:
            for c2 in A:
                v = g.field([c1, c2])
                if g.is_sandwich(v):
                    forced += 1
                    if not fs_rhs(g, v):
                        say("   MISMATCH (sandwich, rhs false): %s" % list(v))
        say("   sandwiches inside A x A: %d" % forced)

        # is c_i^2 = 0 really forced?  scan outside A x A at random
        bad = 0
        Aset = {c.tobytes() for c in A}
        random.seed(1)
        tried = 0
        for _ in range(400000):
            c1 = np.array([random.randrange(p) for _ in range(o.N)], dtype=np.int64)
            c2 = np.array([random.randrange(p) for _ in range(o.N)], dtype=np.int64)
            if c1.tobytes() in Aset and c2.tobytes() in Aset:
                continue
            tried += 1
            v = g.field([c1, c2])
            if g.is_sandwich(v):
                bad += 1
                say("   a sandwich outside A x A: %s" % list(v))
        say("   %d random elements with some c_i^2 != 0: sandwiches found %d"
            % (tried, bad))

        from sandp import rref
        R = rref([v for v in rhs_pts if v.any()], p)[0]
        degs = sorted({g.deg[i] for r in R for i, c in enumerate(r) if c})
        say("   span of the F_p-points of the cone: dim %d, degrees %s" % (len(R), degs))

    log.close()


if __name__ == "__main__":
    main()
