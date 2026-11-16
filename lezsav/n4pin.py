#!/usr/bin/env python3
"""n4pin.py -- what the residual freedom in the completion of bold T IS.

n4cons8.py fitted the completion on 17 solutions on all of which d_+ = d_-,
and left 14 complex directions of freedom.  n4mirror.py and n4boost.py add
z-dependent solutions -- the mirror ones, on which d_+ = -d_-, and the
boosted ones, which depend on z and w BOTH -- and on all of them the same
two coefficients work and the freedom is still 14.  A freedom that does not
move when the family triples is not a want of solutions; it is a property of
the candidate list.

This run identifies it.  Write K_- for the space of coefficient vectors x
with sum_j x_j d_-(log f_j) = 0 on every solution of the family, and K_+ for
the same with d_+.  If

    K_- = K_+,

then every vector of the kernel is killed by d_+ and d_- alike, hence by
d_w and d_z alike, so the combination sum_j x_j log f_j is CONSTANT on every
solution of the system.  The completion is then unique as a functional of
the fields -- determined up to constants of the motion, which is the
ambiguity every Noether current has and no family of solutions can remove.

If instead K_- and K_+ meet in less than 14 directions, the intersection is
what a bigger family could still cut down, and the run says by how much.

Usage: ../venv/bin/python n4pin.py   ->  n4pin-out.txt
"""
import pickle
import sys
import time

from fractions import Fraction as Q_

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import exactlin                                                 # noqa: E402

from n4noeth import (C, mono, add, smul, eq, minv, dT, csolve, cnull,
                     d_xm, d_xp, is_solution)                    # noqa: E402
from n4cons7 import logcols                                      # noqa: E402
from n4cons8 import composites                                   # noqa: E402
from n4rec import qadd, qsmul                                    # noqa: E402
from n4emb import cx, I                                          # noqa: E402
import n4mirror as MIR                                           # noqa: E402
import n4boost as BST                                            # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4pin-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def crank(vecs):
    """The complex rank of a list of complex vectors, by realification."""
    if not vecs:
        return 0
    n = len(vecs[0])
    R = []
    for v in vecs:
        R.append([v[j].re for j in range(n)] + [-v[j].im for j in range(n)])
        R.append([v[j].im for j in range(n)] + [v[j].re for j in range(n)])
    return exactlin.rank(R) // 2


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    with open("n4cons-tan.pkl", "rb") as fh:
        tans = pickle.load(fh)

    log("n4pin: what the residual freedom in the completion of bold T is.")
    log("")

    fam = [("vacuum", q)]
    for d in tans:
        for lam in (1, -1, Q_(1, 2), 2, 3, C(0, 1), C(1, 1), Q_(-1, 3)):
            qq = qadd(q, qsmul(lam, d))
            if is_solution(qq):
                fam.append(("q + (%s) d" % lam, qq))
    allf = ([("v:" + nm, qq) for nm, qq in fam]
            + [("s:" + nm, MIR.sigma(qq)) for nm, qq in fam]
            + [("b:" + nm, BST.boost(qq)) for nm, qq in fam])
    bad = [nm for nm, qq in allf if not is_solution(qq)]
    log("the family: %d solutions (%d vacuum, %d mirror, %d boosted);"
        % (len(allf), len(fam), len(fam), len(fam)))
    log("   every one of them an exact solution of (IK4): %s"
        % ("yes" if not bad else "NO: " + str(bad)))
    log("   the three kinds differ in how the two derivatives act:")
    log("      vacuum family:  z-free,        so d_+ = +d_-;")
    log("      mirror family:  w-free,        so d_+ = -d_-;")
    log("      boosted family: z- and w-dependent, so neither.")
    log("   [%.1fs]" % (time.time() - t0))

    # the composites, once per solution
    fss = []
    for nm, qq in allf:
        fs, qqi = composites(qq, "+")
        fss.append((nm, qq, qqi, fs))
    log("   composites built for all %d [%.1fs]" % (len(fss), time.time() - t0))

    ker = {}
    for dop, dopname in ((d_xm, "d_-"), (d_xp, "d_+")):
        blocks = []
        for nm, qq, qqi, fs in fss:
            cols = logcols(fs, dop)
            cols[("log", "w")] = MIR.logw_col(dop)
            cols[("log", "z")] = MIR.logz_col(dop)
            blocks.append((dT(qq, qqi, dop), cols))
        names = sorted(set().union(*[set(c) for _, c in blocks]), key=str)
        cl, tg = MIR.stack(blocks, names)
        x = csolve(cl, tg)
        A = [[cl[j][r] for j in range(len(cl))] for r in range(len(tg))]
        K = cnull(A)
        ker[dopname] = (names, K)
        log("")
        log("dop = %s: %d rows, %d columns -- %s"
            % (dopname, len(tg), len(cl),
               "SOLVED" if x is not None else "INCONSISTENT"))
        if x is not None:
            nz = [(names[j], x[j]) for j in range(len(names)) if x[j]]
            for n_, v in sorted(nz, key=lambda t: str(t[0])):
                log("      log %-40s %s" % (str(n_), v))
        log("      kernel: %d real = %d complex directions"
            % (len(K), len(K) // 2))
        log("      [%.1fs]" % (time.time() - t0))

    # ---- are the two kernels the same subspace? --------------------------
    nm_m, Km = ker["d_-"]
    nm_p, Kp = ker["d_+"]
    log("")
    log("=" * 70)
    log("the two kernels")
    log("=" * 70)
    if nm_m != nm_p:
        log("   the column names differ; not comparable.")
        return
    rm, rp = crank(Km), crank(Kp)
    rboth = crank(Km + Kp)
    log("   dim K_- = %d, dim K_+ = %d (complex)" % (rm, rp))
    log("   dim (K_- + K_+) = %d" % rboth)
    log("   so dim (K_- n K_+) = %d" % (rm + rp - rboth))
    if rboth == rm == rp:
        log("")
        log("   K_- = K_+.  Every vector of the kernel is killed by d_+ and")
        log("   by d_-, hence by d_w and by d_z, so the corresponding")
        log("   combination of logarithms is CONSTANT on every solution.")
        log("   The completion is therefore unique as a functional of the")
        log("   fields, up to constants of the motion -- the ambiguity every")
        log("   Noether current carries, which no family can remove.")
    else:
        log("")
        log("   K_- != K_+: the intersection, of dimension %d, is the part"
            % (rm + rp - rboth))
        log("   that is constant on shell; the rest is genuine freedom that")
        log("   a larger family of solutions could still cut down.")
    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
