"""The bigraded table dim H^q_k(h^{(1)}(0|n); K), over Q and over F_p.

Usage:  run.py <n> <qmax> [p ...]
Writes both to stdout and to runs/n<n>-q<qmax>.log (never to /tmp).
"""

import sys
import time
from collections import defaultdict
from po import Po
from ce import CE, cohomology


def table(n, qmax, modp=None, weight0_only=False, log=None):
    g = Po(n, "h1")
    ce = CE(g)
    assert not ce.check_d2()
    t0 = time.time()
    h = cohomology(ce, qmax, modp=modp, w0=weight0_only)
    dt = time.time() - t0
    # (q, k, w) -> dim ; collect by (q, k) and record the nonzero torus weights
    byqk = defaultdict(int)
    nonzero_w = set()
    for (q, k, w), v in h.items():
        byqk[(q, k)] += v
        if any(w):
            nonzero_w.add((q, k, w))
    return g, byqk, nonzero_w, dt


def show(n, qmax, byqk, nonzero_w, dt, out):
    ks = sorted({k for (q, k) in byqk})
    def p(s):
        print(s)
        if out:
            out.write(s + "\n")
    p(f"\n  h^(1)(0|{n}):  dim H^q_k with trivial coefficients   [{dt:.1f} s]")
    p("   q \\ k " + "".join(f"{k:5d}" for k in ks) + "   total")
    for q in range(qmax + 1):
        row = "".join(f"{byqk.get((q, k), 0):5d}" if (q, k) in byqk else "    ." for k in ks)
        tot = sum(v for (qq, k), v in byqk.items() if qq == q)
        p(f"  {q:3d}   {row}   {tot:5d}")
    p(f"   classes of nonzero torus weight: {len(nonzero_w)}")


if __name__ == "__main__":
    n = int(sys.argv[1])
    qmax = int(sys.argv[2])
    args = sys.argv[3:]
    w0 = "--w0" in args
    primes = [int(x) for x in args if x != "--w0"]
    first = primes.pop(0) if (w0 and primes) else None
    import os
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/n{n}-q{qmax}.log", "a") as out:
        out.write(f"\n==== n={n} qmax={qmax} primes={primes} w0={w0} ====\n")
        g, byqk, nz, dt = table(n, qmax, modp=first, weight0_only=w0)
        out.write(f"   base field: {'F_' + str(first) if first else 'Q'}\n")
        show(n, qmax, byqk, nz, dt, out)
        out.flush()
        for pp in primes:
            g2, b2, nz2, dt2 = table(n, qmax, modp=pp, weight0_only=w0)
            same = (b2 == byqk)
            msg = (f"   mod {pp}: {'IDENTICAL to Q' if same else 'DIFFERS from Q'}"
                   f"   [{dt2:.1f} s]")
            print(msg)
            out.write(msg + "\n")
            if not same:
                show(n, qmax, b2, nz2, dt2, out)
            out.flush()
