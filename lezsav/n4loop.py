#!/usr/bin/env python3
"""n4loop.py -- what the level-0 datum really is: a point of a loop group.

Every previous engine took the theta-free level-0 part of q to be

    q^(0) = A z^{a_0} w^{-1-a_0} / m,     A a constant matrix,

but that is a special case.  What the system actually demands is only that q
be invertible in the ring, i.e. that det q be a UNIT of the Laurent ring
K[z^{+-1}, w^{+-1}]; the units of a Laurent polynomial ring over a field are
exactly the monomials, and det q = det q^(0) + (nilpotent), so:

    det q^(0) must be a monomial,

and nothing more.  Since every entry of m q^(0) is homogeneous of dilatation
degree -1, writing zeta := z/w gives

    m q^(0) = (1/w) F(zeta),     F a 2x2 matrix of Laurent polynomials,
    det(m q^(0)) = (1/w^2) det F(zeta)  a monomial  <=>  det F = c zeta^d,

that is,

    F  in  GL_2( K[zeta^{+-1}] ),

the group of Laurent-polynomial loops.  The old ansatz is the subgroup of
constant matrices times a central monomial; the loop group is much bigger,
and its non-monomial points are genuinely two-variable by construction --
d_z q^(0) and d_w q^(0) are not proportional over the constants unless F is
a monomial times a constant matrix.

This engine runs step 0 -- which carries no earlier choice, so its
solvability is a property of the level-0 datum alone -- over representative
points of the loop group: monomial, diagonal, unipotent, and products.

WINDOW.  The unknowns at theta-level 2r are z^n w^{-1-r-n} m^{-1} for n in
an ABSOLUTE range [NLO, NHI], not a range relative to a_0: n4two3.py shows
that a range tied to a_0 mistakes the rows it cannot reach for an
obstruction (a_0 = -2 is inconsistent in s in [-1,1] and solvable in
[-2,2]).  The rows a window cannot reach are counted and reported.

Usage: ../venv/bin/python n4loop.py [NLO] [NHI] [PROCS]
Writes n4loop-out-n<NLO>_<NHI>.txt.
"""
import itertools
import pickle
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import n4noeth as N
import n4phi
import n4rec
from n4rec import qadd, qsmul, resid_full
from n4noeth import mono, mzero, C, MI, PL, mdet, is_solution
from n4emb import cx

OUT = []
TAG = ""


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4loop-out%s.txt" % TAG, "w") as fh:
        fh.write("\n".join(OUT) + "\n")


# --- the class, with an ABSOLUTE z-window -----------------------------------

def dirs_abs(r, nlo, nhi):
    out, tag = [], []
    for n in range(nlo, nhi + 1):
        for g in (0, 1):
            for b in (0, 1):
                for S in itertools.combinations(MI + PL, 2 * r):
                    mo = mono(n, -1 - r - n, -1, S)
                    if not mo:
                        continue
                    dq = mzero()
                    dq[g][b] = mo
                    out.append(dq)
                    tag.append(n)
    return out, tag


def q0_of_loop(F):
    """F = {(i,j): {d: coeff}} -- the entry (i,j) of F(zeta) as a Laurent
    polynomial sum_d coeff * zeta^d.  Returns q^(0) = F(z/w)/(m w)."""
    q = mzero()
    for (i, j), pol in F.items():
        for d, c in pol.items():
            if c:
                k = (d, -1 - d, -1, ())
                q[i][j] = q[i][j] if q[i][j] else {}
                q[i][j][k] = q[i][j].get(k, C()) + cx(c)
    for i in (0, 1):
        for j in (0, 1):
            q[i][j] = {k: v for k, v in q[i][j].items() if v}
    return q


def det_is_monomial(q):
    d = mdet(q)
    tf = {k: v for k, v in d.items() if k[3] == ()}
    return len(tf) == 1, tf


def level_of(k):
    return len(k[2][3])


def step_of(k):
    lv = level_of(k)
    return (lv - 1) // 2 if k[0] == "C" else lv // 2


_G = {}


def _init(qp, dp):
    _G["q"] = pickle.loads(qp)
    _G["dirs"] = pickle.loads(dp)


def _col(i):
    q, d = _G["q"], _G["dirs"][i]
    V = {t: resid_full(q if t == 0 else qadd(q, qsmul(t, d)))
         for t in (-2, -1, 0, 1, 2)}
    keys = set()
    for t in V:
        keys |= set(V[t])
    c1, nonlin = {}, 0
    for k in keys:
        a = {t: V[t].get(k, C()) for t in V}
        lin = (a[1] - a[-1]) * Q(1, 2)
        if a[1] + a[-1] - a[0] * 2 or (a[2] - a[-2]) != lin * 4:
            nonlin += 1
        if lin:
            c1[k] = lin
    return i, c1, nonlin


def step0(q, nlo, nhi, procs):
    F = resid_full(q)
    F0 = {k: v for k, v in F.items() if step_of(k) == 0}
    ds, tg = dirs_abs(1, nlo, nhi)
    qp, dp = pickle.dumps(q), pickle.dumps(ds)
    cols = [None] * len(ds)
    nonlin = 0
    with mp.Pool(procs, _init, (qp, dp)) as pool:
        for i, c1, nl in pool.imap_unordered(_col, range(len(ds)),
                                             chunksize=4):
            cols[i] = {k: v for k, v in c1.items() if step_of(k) == 0}
            nonlin += nl
    reach = {k for c in cols for k in c}
    unreach = [k for k in F0 if k not in reach]
    x = N.sparse_solve(cols, {k: -v for k, v in F0.items()})
    ker = N.sparse_null(cols) if x is not None else None
    return x, ker, len(F0), unreach, nonlin, len(ds)


def E(*pairs):
    """{(i,j): {d: c}} from (i, j, d, c) tuples."""
    F = {}
    for (i, j, d, c) in pairs:
        F.setdefault((i, j), {})[d] = c
    return F


def main():
    global TAG
    t0 = time.time()
    nlo = int(sys.argv[1]) if len(sys.argv) > 1 else -3
    nhi = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    TAG = "-n%d_%d" % (nlo, nhi)
    log("n4loop: step 0 over points of GL_2(K[zeta^{+-1}]), zeta = z/w.")
    log("absolute z-window for the unknowns: n in [%d, %d]." % (nlo, nhi))
    log("")

    cases = [
        ("A            (a_0 = 0, the vacuum family)",
         E((0, 0, 0, 1), (1, 1, 0, 1))),
        ("A zeta^{-1}  (a_0 = -1, the mirror family)",
         E((0, 0, -1, 1), (1, 1, -1, 1))),
        ("A zeta       (a_0 = 1, monomial)",
         E((0, 0, 1, 1), (1, 1, 1, 1))),
        ("A zeta^{-2}  (a_0 = -2, monomial)",
         E((0, 0, -2, 1), (1, 1, -2, 1))),
        ("A zeta^{2}   (a_0 = 2, monomial)",
         E((0, 0, 2, 1), (1, 1, 2, 1))),
        ("diag(zeta, 1)",
         E((0, 0, 1, 1), (1, 1, 0, 1))),
        ("diag(zeta, zeta^{-1})",
         E((0, 0, 1, 1), (1, 1, -1, 1))),
        ("diag(zeta^{-1}, 1)",
         E((0, 0, -1, 1), (1, 1, 0, 1))),
        ("[[1, zeta], [0, 1]]  (unipotent)",
         E((0, 0, 0, 1), (0, 1, 1, 1), (1, 1, 0, 1))),
        ("[[1, zeta^{-1}], [0, 1]]",
         E((0, 0, 0, 1), (0, 1, -1, 1), (1, 1, 0, 1))),
        ("[[1+zeta, zeta], [1, 1]]  (det = 1)",
         E((0, 0, 0, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 1, 0, 1))),
        ("[[zeta, 1], [-1, 0]]  (det = 1)",
         E((0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 0, -1))),
    ]
    # patch the first case: [[1+zeta, zeta],[1,1]] needs the 1+zeta entry
    cases[10] = ("[[1+zeta, zeta], [1, 1]]  (det = 1)",
                 {(0, 0): {0: 1, 1: 1}, (0, 1): {1: 1},
                  (1, 0): {0: 1}, (1, 1): {0: 1}})

    log("  F(zeta)                                    det F       step 0")
    log("  " + "-" * 72)
    for name, F in cases:
        q = q0_of_loop(F)
        okdet, tf = det_is_monomial(q)
        if not okdet:
            log("  %-42s NOT A UNIT -- det has %d monomials, skipped"
                % (name, len(tf)))
            continue
        (dk, dv), = tf.items()
        dets = "%s zeta^%d" % (dv, dk[0])
        x, ker, nrows, unreach, nonlin, nd = step0(q, nlo, nhi, procs)
        st = ("solvable, kernel %d" % len(ker)) if x is not None \
            else "INCONSISTENT"
        extra = ""
        if unreach:
            byz = {}
            for k in unreach:
                byz[k[2][0]] = byz.get(k[2][0], 0) + 1
            extra = "  [%d rows beyond the window, at n_z = %s]" % (
                len(unreach), sorted(byz.items()))
        log("  %-42s %-11s %s  (%d rows, %d unknowns)%s"
            % (name, dets, st, nrows, nd, extra))
        if nonlin:
            log("      !! %d rows nonlinear in the level-2 unknowns" % nonlin)
    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
