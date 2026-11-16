"""The censuses over k(2n+1) for n >= 2.  Usage:  python runn.py <job> [n]

Jobs:
  dens    the weighted densities, W = 0..WMAX
  metric  T(2,l) x T(2,m) -> T(k3,.) for k3 = 0,2,4 and W = 0,2 (and 4 if asked)
  line    the locus of the exceptional operator: a grid of weight pairs, k3 = 0, W = 2
  linear  the linear operators T(k,l) -> T(k',l') (the contact complex)
Output goes to runn-<job>-<n>.log as well as to the screen.
"""

import sys
import time
from fractions import Fraction as Fr

from genn import Problem, target_weight

LOG = None


def say(s):
    print(s, flush=True)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def dim_of(n, k1, l1, k2, l2, k3, W, vpieces=(0, 1, 2), fpdeg=2):
    P = Problem(n, k1, k2, k3, W)
    if not P.schemes:
        return 0, 0
    l3, ns = P.solve(l1, l2, vpieces=vpieces)
    return P.op_dim(ns, deg=fpdeg), P.dim_ansatz()


WEIGHTS = [(Fr(0), Fr(0)), (Fr(2, 3), Fr(-1, 5)), (Fr(-1), Fr(-1)),
           (Fr(1, 2), Fr(3, 7)), (Fr(-3, 2), Fr(5, 4))]


def job_dens(n, WMAX=6):
    say("=== k(%d): the weighted densities, W <= %d ===" % (2 * n + 1, WMAX))
    for W in range(WMAX + 1):
        row = []
        t0 = time.time()
        for l, m in WEIGHTS:
            d, a = dim_of(n, 0, l, 0, m, 0, W)
            row.append(d)
        say("  W=%d  ansatz %2d  dims %s   [%.0f s]"
            % (W, a, row, time.time() - t0))


def job_metric(n, WS=(0, 2)):
    say("=== k(%d): the metrics T(2,l) x T(2,m) -> T(k3,.) ===" % (2 * n + 1))
    pairs = [(Fr(1, 3), Fr(-1, 3)), (Fr(1, 3), Fr(2, 5)), (Fr(0), Fr(0)),
             (Fr(1), Fr(-1)), (Fr(-2), Fr(2))]
    for k3 in (0, 2, 4):
        for W in WS:
            t0 = time.time()
            row = []
            for l, m in pairs:
                d, a = dim_of(n, 2, l, 2, m, k3, W)
                row.append((str(l), str(m), d))
            say("  k3=%d W=%d  ansatz %2d  %s   [%.0f s]"
                % (k3, W, a, row, time.time() - t0))


def job_line(n):
    say("=== k(%d): the locus, k3=0, W=2 ===" % (2 * n + 1))
    vals = [Fr(a, 2) for a in range(-4, 5)]
    for l in vals:
        row = []
        for m in vals:
            d, a = dim_of(n, 2, l, 2, m, 0, 2)
            row.append(d)
        say("  lambda=%-5s  %s" % (l, row))


def job_linear(n, KMAX=2, WMAX=3):
    """Linear operators: the bilinear engine with the first argument a constant is not
    the right tool, so this uses the naive one-argument ansatz built from the schemes
    with G removed -- i.e. the schemes with no G-slot."""
    from genn import Problem as _P
    say("=== k(%d): the linear operators (contact complex) ===" % (2 * n + 1))
    import linn
    linn.run(n, KMAX, WMAX, say)


def main():
    global LOG
    job = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    LOG = open("runn-%s-%d.log" % (job, n), "w")
    extra = sys.argv[3:]
    if job == "dens":
        job_dens(n, int(extra[0]) if extra else 6)
    elif job == "metric":
        job_metric(n, tuple(int(x) for x in extra) if extra else (0, 2))
    elif job == "line":
        job_line(n)
    elif job == "linear":
        job_linear(n)
    else:
        raise SystemExit("unknown job")


if __name__ == "__main__":
    main()
