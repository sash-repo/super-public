#!/usr/bin/env python3
"""Cross-check of the pericontact census: dim Der L computed with no grading.

der_dim() splits the derivation space by (degree, parity); this recomputes the
same number with the full matrix as the unknown, as xcheck.py does for the four
classical series.  A mismatch would mean that the grading assumption is wrong
for m(n) or for the members of the family b_{a,b}(n).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dervect import Echelon, der_dim, derived                      # noqa: E402
from brute import der_dim_brute                                    # noqa: E402
import mvect                                                       # noqa: E402

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mxcheck.log")
LIMIT = 40


def main():
    with open(LOG, "w") as fh:
        def say(s):
            print(s)
            fh.write(s + "\n")
            fh.flush()

        bad = 0
        for p, n in ((3, 1), (5, 1), (3, 2)):
            S = mvect.make(p, n)
            L, fields = mvect.pericontact(S, n)
            cases = [("m(%d)" % n, L), ("m(%d)^(1)" % n, derived(S, L))]
            for (a, b) in [(1, x) for x in range(p)] + [(0, 1)]:
                B = mvect.buttin(S, n, a, b, fields)
                cases.append(("b_(%d:%d)(%d)" % (a, b, n), B))
                cases.append(("b_(%d:%d)(%d)^(1)" % (a, b, n), derived(S, B)))
            for tag, M in cases:
                d = Echelon([S.vec(X) for X in M], S.p).dim
                if d == 0 or d > LIMIT:
                    say(f"p={p} {tag}: dim={d} SKIPPED (limit {LIMIT})")
                    continue
                t0 = time.time()
                g = der_dim(S, M)
                u = der_dim_brute(S, M)
                ok = "ok" if g == u else "MISMATCH"
                if g != u:
                    bad += 1
                say(f"p={p} {tag}: dim={d} graded={g} ungraded={u} {ok} "
                    f"[{time.time() - t0:.0f}s]")
        say(f"mismatches: {bad}")


if __name__ == "__main__":
    main()
