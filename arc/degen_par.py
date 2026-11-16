#!/usr/bin/env python3
"""degen.py, in parallel and with incremental output.

Same scan as degen.py --- for every pair of weights (mu1, mu2) in (1/2)Zee,
the dimension of the space of invariant bilinear differential operators of
order <= N, one entry per shift s = 0, ..., 2N --- but one worker per weight
pair, so the order N = 4 sweep (169 pairs x 9 shifts x ~3 s) fits in minutes
instead of an hour.  Rows are written as they arrive (line-buffered), hence a
killed run still leaves everything it had computed.

    ../venv/bin/python degen_par.py 4 > runs-degen-N4.txt
"""
import sys
import time
from fractions import Fraction as F
from multiprocessing import Pool

import fullops as FO

HALF = [F(n, 2) for n in range(-6, 7)]
GEN = (F(1, 31), F(5, 31))


def dims(job):
    m1, m2, N = job
    return m1, m2, [len(FO.solve(N, m1, m2, F(s))[1]) for s in range(0, 2 * N + 1)]


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    print(f"# order <= {N};  entry = dim Hom for the shift s = 0,1,2,...,{2*N}")
    print(f"# generic reference pair {GEN};  {procs} workers", flush=True)

    _, _, ref = dims((GEN[0], GEN[1], N))
    print(f"{'mu1':>6} {'mu2':>6}   dims per shift s=0..{2*N}")
    print(f"{'gen':>6} {'gen':>6}   {ref}", flush=True)

    jobs = [(m1, m2, N) for m1 in HALF for m2 in HALF]
    out = {}
    with Pool(procs) as pool:
        for m1, m2, d in pool.imap_unordered(dims, jobs, chunksize=1):
            out[(m1, m2)] = d
            if d != ref:
                print(f"# (unsorted, as computed) {m1} {m2} {d}", flush=True)

    odd = 0
    print()
    for m1 in HALF:
        for m2 in HALF:
            d = out[(m1, m2)]
            if d != ref:
                odd += 1
                print(f"{str(m1):>6} {str(m2):>6}   {d}   <-- differs")
    print(f"\n# weight pairs differing from the generic answer: {odd} "
          f"of {len(HALF)**2}")
    print(f"# {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
