#!/usr/bin/env python3
"""n4elim.py -- toward Conjecture Conjecture 7.2.8: the level-1 system
as a polynomial map on the kernel of the level-0 system.

Stage 1 (this file, 'build'): q0 = loop, the level-2 directions d2, the
particular solution x0 and kernel ker0 of step 0, and the step-1 matrix M4 of
the level-4 unknowns -- which depends on q0 only (a level-4 term can reach
theta-degree <= 3 only multiplied by level-0 data); that is CHECKED on random
columns at a random point of the kernel.  Everything pickled into this folder.

Usage: ../venv/bin/python n4elim.py build [loop] [a0]
"""
import os, sys, pickle, random, time
import multiprocessing as mp
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction as Q
import n4noeth as N
from n4rec import qadd, qsmul, resid_full
from n4noeth import mzero, C
from n4loop import dirs_abs, q0_of_loop, step_of, E

HERE = os.path.dirname(os.path.abspath(__file__))
NLO, NHI = -3, 4
LOOPS = {"mono": lambda a: E((0, 0, a, 2), (1, 1, a, 2)),
         "diag": lambda a: E((0, 0, a, 1), (1, 1, 0, 1)),
         "uni": lambda a: E((0, 0, 0, 1), (0, 1, a, 1), (1, 1, 0, 1))}
_G = {}


def _init(qp, dp, st):
    _G["q"] = pickle.loads(qp); _G["d"] = pickle.loads(dp); _G["st"] = st


def _col(i):
    q, d, st = _G["q"], _G["d"][i], _G["st"]
    R0 = resid_full(q)
    Rp = resid_full(qadd(q, d))
    out = {}
    for k in set(R0) | set(Rp):
        if step_of(k) != st:
            continue
        v = Rp.get(k, C()) - R0.get(k, C())
        if v:
            out[k] = v
    return i, out


def cols_at(q, ds, st, idx=None, procs=8):
    idx = list(range(len(ds))) if idx is None else idx
    res = {}
    with mp.Pool(procs, _init, (pickle.dumps(q), pickle.dumps(ds), st)) as p:
        for i, c in p.imap_unordered(_col, idx, chunksize=4):
            res[i] = c
    return res


def qof(q0, d2, coeffs):
    u = mzero()
    for j, c in enumerate(coeffs):
        if c:
            u = qadd(u, qsmul(c, d2[j]))
    return qadd(q0, u)


def build(loop, a0, n4lo=NLO, n4hi=NHI):
    t0 = time.time()
    q0 = q0_of_loop(LOOPS[loop](a0))
    d2, _ = dirs_abs(1, NLO, NHI)
    d4, _ = dirs_abs(2, n4lo, n4hi)
    c2 = cols_at(q0, d2, 0)
    cols2 = [c2[i] for i in range(len(d2))]
    F = resid_full(q0)
    tgt0 = {k: -v for k, v in F.items() if step_of(k) == 0}
    x0 = N.sparse_solve(cols2, tgt0)
    ker0 = N.sparse_null(cols2)
    print("step 0: %d level-2 unknowns, kernel %d [%.0fs]"
          % (len(d2), len(ker0), time.time() - t0), flush=True)
    qs = qof(q0, d2, x0)
    c4 = cols_at(qs, d4, 1)
    M4 = [c4[i] for i in range(len(d4))]
    print("M4 built: %d columns [%.0fs]" % (len(d4), time.time() - t0),
          flush=True)
    # constancy check at a random kernel point
    rng = random.Random(7)
    co = list(x0)
    for v in ker0:
        lam = C(rng.randint(-3, 3), rng.randint(-3, 3))
        co = [a + lam * b for a, b in zip(co, v)]
    qr = qof(q0, d2, co)
    idx = rng.sample(range(len(d4)), 24)
    c4r = cols_at(qr, d4, 1, idx)
    bad = sum(1 for i in idx if c4r[i] != M4[i])
    print("M4 constancy: %d of %d sampled columns differ at a random point"
          % (bad, len(idx)), flush=True)
    tag = "%s%d" % (loop, a0) if (n4lo, n4hi) == (NLO, NHI) else \
        "%s%d-w%d_%d" % (loop, a0, n4lo, n4hi)
    with open(os.path.join(HERE, "build-%s.pkl" % tag), "wb") as fh:
        pickle.dump(dict(q0=q0, d2=d2, x0=x0, ker0=ker0, M4=M4,
                         nd4=len(d4), const_bad=bad), fh)
    print("[%.0fs] saved" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        a = sys.argv[2:]
        build(a[0], int(a[1]), *(int(x) for x in a[2:4]))
