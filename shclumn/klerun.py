#!/usr/bin/env python3
"""H^2(kle(9|6;K); kle(9|6;K))_k, degree by degree, exactly over Q."""
import sys
import time
from fractions import Fraction
import kle
import klecoh as K

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, backend                                    # noqa: E402

DEPTH = 2


HAT = False


def build(dmax):
    f = kle.hatcomp if HAT else kle.comp
    return {d: f(d) for d in range(-DEPTH, dmax + 1)}


def census(krange, dmax):
    """Just the number of weight-0 2-cochains on the determining components."""
    comp = build(dmax)
    G = K.Graded(kle, comp)
    for k in krange:
        allowed = K.window(k, DEPTH)
        allowed = [(i, j) for (i, j) in allowed
                   if i in comp and j in comp and (i + j + k) in comp]
        C = K.Cocycle(G, allowed, k)
        print('  k = %3d   components %-42s  weight-0 cochains %d'
              % (k, allowed, C.n))


def ext(k, depth, Tmax):
    """All bidegrees with -depth-k <= i+j <= Tmax (the window plus a tail)."""
    out = []
    for T in range(-depth - k, Tmax + 1):
        for i in range(-depth, T + depth + 1):
            j = T - i
            if j < -depth or i > j:
                continue
            out.append((i, j))
    return out


def h2(k, dmax, tmax, Tmax=None):
    comp = build(dmax)
    G = K.Graded(kle, comp)
    allowed = K.window(k, DEPTH) if Tmax is None else ext(k, DEPTH, Tmax)
    allowed = [(i, j) for (i, j) in allowed
               if i in comp and j in comp and (i + j + k) in comp]
    C = K.Cocycle(G, allowed, k)
    degs = sorted({d for p in allowed for d in (p[0], p[1], p[0] + p[1])})
    lo = min(min(p) for p in allowed)
    triples = [(a, b, c) for a in range(lo, tmax + 1)
               for b in range(a, tmax + 1) for c in range(b, tmax + 1)]
    t0 = time.time()
    eqs = K.equations(G, C, triples)
    cob = K.coboundaries(G, C, degs)
    # every coboundary must satisfy every equation -- a check on the signs
    bad = 0
    for r in cob:
        for e in eqs:
            if sum(x * y for x, y in zip(r, e) if x and y):
                bad += 1
                break
    rz = C.n - rank(eqs) if eqs else C.n
    rb = rank(cob) if cob else 0
    print('k=%d: cochains %d, equations %d -> cocycles <= %d, coboundaries %d, '
          'B not in Z: %d  [%.1fs]' % (k, C.n, len(eqs), rz, rb, bad, time.time() - t0))
    return C.n, rz, rb, bad


if __name__ == '__main__':
    if '--hat' in sys.argv:
        globals()['HAT'] = True
        sys.argv.remove('--hat')
    print('exactlin backend:', backend(), '| hat =', HAT)
    if sys.argv[1] == 'census':
        census(range(int(sys.argv[2]), int(sys.argv[3]) + 1), int(sys.argv[4]))
    else:
        h2(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
           int(sys.argv[4]) if len(sys.argv) > 4 else None)
