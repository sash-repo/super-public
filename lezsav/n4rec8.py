#!/usr/bin/env python3
"""n4rec8.py -- step 1 decided in full: eliminate ALL 280 affine unknowns.

n4rec6.py/n4rec7.py did the elimination on the blocked sector alone (42 of
the 666 rows) and found that the sector, quadratic in the kernel parameters
y, HAS solutions: the blockage seen at y = 0 is removable.  So the sector is
too weak to decide step 1, and the whole of step 1 must be reduced.

That is what this does, and the reduction is an EQUIVALENCE, not a necessary
condition: step 1 is

    F(y, x) = F0 + sum_i y_i L_i + sum_j x_j M_j + Q(y) = 0,

affine in the 280 level-4 unknowns x.  For a fixed y it is a linear system in
x, solvable if and only if its right-hand side is orthogonal to the left
kernel of the x-block.  Projecting on a basis of that left kernel therefore
gives a system in y ALONE whose solutions are exactly the y that extend to a
solution of step 1.

  step 1 solvable  <==>  the reduced system in y is solvable.

Then the reduced system is decided: its affine rows first (a linear system --
inconsistency there is already a proof of emptiness), then, on the affine
solution space, the quadratic rows, by the linearization relaxation and by a
Groebner basis over Q(i).

The left kernel is computed over Q by realification of Q(i) --
[[Re, -Im], [Im, Re]] -- with FLINT behind lib/exactlin.py; the projected
system is then rebuilt in exact Q(i) arithmetic and every kernel vector is
verified against the x-block before use.

Usage: ../venv/bin/python n4rec8.py [--all] [--nogb]
Writes n4rec8-out.txt.
"""
import itertools
import json
import random
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import exactlin                                  # noqa: E402

import n4noeth as N                              # noqa: E402
import n4vac                                     # noqa: E402
import n4phi                                     # noqa: E402
import n4rec                                     # noqa: E402
import n4rec2                                    # noqa: E402
import n4rec6                                    # noqa: E402
from n4rec import qadd, qsmul, resid_full, q0_of, det2   # noqa: E402
from n4rec6 import build_base, build_model, selftest, dot, dcomb  # noqa: E402
from n4noeth import mzero, C                     # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4rec8-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


# --- the left kernel of the x-block, over Q(i) ------------------------------

def left_kernel(cols, rows):
    """Complex vectors mu with sum_r mu_r * col[r] = 0 for every column.

    cols: list of sparse dicts row -> C.  Returns list of dicts row -> C."""
    ridx = {k: i for i, k in enumerate(rows)}
    nr = len(rows)
    # the matrix of the linear map mu -> (<mu, col_j>)_j is X^T: rows = cols
    re_rows, im_rows = [], []
    for cdict in cols:
        rr = [Q(0)] * nr
        ii = [Q(0)] * nr
        for k, v in cdict.items():
            rr[ridx[k]] = v.re
            ii[ridx[k]] = v.im
        re_rows.append(rr)
        im_rows.append(ii)
    # realified: [[Re, -Im], [Im, Re]] acting on [u; v]
    big = []
    for rr, ii in zip(re_rows, im_rows):
        big.append(list(rr) + [-x for x in ii])
    for rr, ii in zip(re_rows, im_rows):
        big.append(list(ii) + list(rr))
    ns = exactlin.nullspace(big)
    out = []
    for vec in ns:
        u, v = vec[:nr], vec[nr:]
        mu = {}
        for i, k in enumerate(rows):
            if u[i] or v[i]:
                mu[k] = C(u[i], v[i])
        if mu:
            out.append(mu)
    return out


def verify_kernel(mus, cols, n=None):
    """Every mu must annihilate every column."""
    bad = 0
    for mu in mus:
        for cdict in cols:
            if dot(mu, cdict):
                bad += 1
                break
    return bad


# --- the reduced system -----------------------------------------------------

def reduce_system(mus, F0, L, Mx, Qd, Qo):
    const, lin, quad = {}, {}, {}
    for a, mu in enumerate(mus):
        v = dot(mu, F0)
        if v:
            const[a] = v
        for i, D in enumerate(L):
            v = dot(mu, D)
            if v:
                lin.setdefault(i, {})[a] = v
        for i, D in enumerate(Qd):
            v = dot(mu, D)
            if v:
                quad.setdefault((i, i), {})[a] = v
        for ij, D in Qo.items():
            v = dot(mu, D)
            if v:
                quad.setdefault(ij, {})[a] = v
    return const, lin, quad


def decide(const, lin, quad, neq, nvar, t0, nogb):
    """Decide sum lin_i y_i + sum quad_ij y_i y_j + const = 0."""
    qrows = {a for D in quad.values() for a in D}
    aff = set(range(neq)) - qrows
    log("  reduced system: %d equations in %d unknowns; %d affine, %d"
        " quadratic; %d quadratic monomials"
        % (neq, nvar, len(aff), len(qrows), len(quad)))

    # --- 1) the affine rows -------------------------------------------------
    cols = []
    for i in range(nvar):
        D = lin.get(i, {})
        cols.append({a: v for a, v in D.items() if a in aff})
    target = {a: -v for a, v in const.items() if a in aff}
    nontriv = sum(1 for c in cols if c)
    log("  1) the affine rows: %d nonzero columns, %d nonzero right-hand"
        " sides" % (nontriv, len(target)))
    if target:
        y0 = N.sparse_solve(cols, target)
        if y0 is None:
            log("     INCONSISTENT  ==>  PROOF: step 1 has no solution, so no"
                " q of")
            log("     the graded class with this level-0 datum solves (IK4).")
            return "proof"
    else:
        y0 = [C() for _ in range(nvar)]
    ker = N.sparse_null(cols)
    d = len(ker)
    log("     solvable; the affine solution set is %d-dimensional  [%.1fs]"
        % (d, time.time() - t0))
    for a in aff:
        s = const.get(a, C())
        for i in range(nvar):
            c = lin.get(i, {}).get(a)
            if c:
                s = s + c * y0[i]
        if s:
            log("     !! y0 fails affine row %d -- aborting" % a)
            return "error"

    # --- 2) substitute into the quadratic rows ------------------------------
    Ac, Al, Aq = {}, {}, {}
    for a in sorted(qrows):
        s = const.get(a, C())
        for i in range(nvar):
            c = lin.get(i, {}).get(a)
            if c:
                s = s + c * y0[i]
        for (i, j), D in quad.items():
            c = D.get(a)
            if c:
                s = s + c * y0[i] * y0[j]
        if s:
            Ac[a] = s
        for ci in range(d):
            u = ker[ci]
            s = C()
            for i in range(nvar):
                c = lin.get(i, {}).get(a)
                if c:
                    s = s + c * u[i]
            for (i, j), D in quad.items():
                c = D.get(a)
                if c:
                    s = s + c * (y0[i] * u[j] + u[i] * y0[j])
            if s:
                Al.setdefault(ci, {})[a] = s
        for ci in range(d):
            for ei in range(ci, d):
                u, v = ker[ci], ker[ei]
                s = C()
                for (i, j), D in quad.items():
                    c = D.get(a)
                    if c:
                        if ci == ei:
                            s = s + c * u[i] * u[j]
                        else:
                            s = s + c * (u[i] * v[j] + v[i] * u[j])
                if s:
                    Aq.setdefault((ci, ei), {})[a] = s
    live = sorted({a for a in qrows if a in Ac
                   or any(a in D for D in Al.values())
                   or any(a in D for D in Aq.values())})
    tvars = sorted(set(Al) | {c for ce in Aq for c in ce})
    log("  2) after substituting the affine solution: %d nonzero equations,"
        " %d unknowns, %d quadratic monomials  [%.1fs]"
        % (len(live), len(tvars), len(Aq), time.time() - t0))
    if not live:
        log("     every quadratic row is satisfied identically  ==>  step 1"
            " IS SOLVABLE")
        return "solvable"

    # --- 3) the relaxation --------------------------------------------------
    cols = [D for D in Al.values()] + [D for D in Aq.values()]
    target = {a: -v for a, v in Ac.items()}
    if target and N.sparse_solve(cols, target) is None:
        log("  3) the linearization relaxation is INCONSISTENT  ==>  PROOF:")
        log("     step 1 has no solution, so no q of the graded class with")
        log("     this level-0 datum solves (IK4).")
        return "proof"
    log("  3) the linearization relaxation is solvable -- Groebner decides")
    with open("n4rec8-reduced.json", "w") as fh:
        json.dump({"neq": len(live), "d": d,
                   "const": {str(a): [str(v.re), str(v.im)]
                             for a, v in Ac.items()},
                   "lin": {str(c): {str(a): [str(v.re), str(v.im)]
                                    for a, v in D.items()}
                           for c, D in Al.items()},
                   "quad": {"%d,%d" % ce: {str(a): [str(v.re), str(v.im)]
                                           for a, v in D.items()}
                            for ce, D in Aq.items()}}, fh)
    log("     the reduced quadratic system written to n4rec8-reduced.json")
    if nogb:
        return "open"

    # --- 4) Groebner --------------------------------------------------------
    import sympy
    tv = {c: sympy.Symbol("t%d" % c) for c in tvars}

    def sy(v):
        return sympy.Rational(v.re) + sympy.I * sympy.Rational(v.im)

    polys = {}
    for a, v in Ac.items():
        polys[a] = polys.get(a, 0) + sy(v)
    for c, D in Al.items():
        for a, v in D.items():
            polys[a] = polys.get(a, 0) + sy(v) * tv[c]
    for (c, e), D in Aq.items():
        for a, v in D.items():
            polys[a] = polys.get(a, 0) + sy(v) * tv[c] * tv[e]
    P = [sympy.expand(p) for p in polys.values()]
    P = [p for p in P if p != 0]
    log("  4) %d polynomials in %d unknowns handed to the Groebner engine"
        % (len(P), len(tvars)))
    t1 = time.time()
    try:
        G = sympy.groebner(P, *[tv[c] for c in tvars], order="grevlex",
                           domain=sympy.QQ_I)
    except Exception as e:                        # noqa: BLE001
        log("     Groebner failed: %s" % e)
        return "open"
    gl = list(G.exprs)
    log("     basis of %d elements  [%.1fs]" % (len(gl), time.time() - t1))
    if len(gl) == 1 and gl[0].is_number and gl[0] != 0:
        log("     the basis is {1}  ==>  PROOF: step 1 has NO solution,")
        log("     so no q of the graded class with this level-0 datum solves"
            " (IK4).")
        return "proof"
    import sympy as sp
    log("     the basis is NONTRIVIAL (degrees %s): step 1 IS solvable, and"
        % sorted({sp.total_degree(g) for g in gl}))
    log("     the vacuum survives to level 4.")
    return "solvable"


def run_one(A, name, procs, t0, nogb, rng):
    base = build_base(A, procs)
    if base is None:
        log("  step 0 inconsistent")
        return "step0"
    q, kmats, ds4 = base
    F0, L, Mx, Qd, Qo = build_model(q, kmats, ds4)
    rows = sorted({k for D in [F0] + list(L) + list(Mx) + list(Qd)
                   + list(Qo.values()) for k in D})
    log("  model: %d rows, %d + %d + %d columns  [%.1fs]"
        % (len(rows), len(L), len(Mx), len(Qd) + len(Qo), time.time() - t0))
    bad = selftest(q, kmats, ds4, F0, L, Mx, Qd, Qo, 3, rng)
    if bad:
        log("  !! the quadratic model failed its self-test -- aborting")
        return "error"
    log("  model self-test at 3 random integer points: exact")

    mus = left_kernel(list(Mx), rows)
    log("  left kernel of the x-block: %d vectors over Q(i) (%d rows, %d"
        " columns)  [%.1fs]" % (len(mus), len(rows), len(Mx),
                                time.time() - t0))
    bad = verify_kernel(mus, list(Mx))
    log("  kernel verified against all %d x-columns: %d failures"
        % (len(Mx), bad))
    if bad:
        return "error"
    const, lin, quad = reduce_system(mus, F0, L, Mx, Qd, Qo)
    log("  projection done  [%.1fs]" % (time.time() - t0))
    return decide(const, lin, quad, len(mus), len(L), t0, nogb)


def main():
    t0 = time.time()
    procs = max(1, mp.cpu_count() - 2)
    rng = random.Random(7)
    nogb = "--nogb" in sys.argv

    ms0 = n4vac.graded_monos(0, 0, -5, 1, mfree=(-1, -1))
    B = n4vac.constrained_basis(ms0)
    qb, kern = n4vac.build_embedded_base(B)
    Aemb = [[qb[i][j].get((0, -1, -1, ()), C()) for j in (0, 1)]
            for i in (0, 1)]
    trials = [("embedded diag(2,2)", Aemb)]
    if "--all" in sys.argv:
        trials += [("identity", [[C(1), C()], [C(), C(1)]]),
                   ("antidiag", [[C(), C(1)], [C(1), C()]]),
                   ("upper unitriang", [[C(1), C(1)], [C(), C(1)]])]

    log("n4rec8: step 1 of the theta-recursion, ALL 280 affine unknowns")
    log("eliminated; the reduction is an equivalence, so what is left")
    log("decides step 1 outright.")
    log("")
    for name, A in trials:
        log("=" * 68)
        log("level-0 datum: %s   det = %s" % (name, det2(A)))
        v = run_one(A, name, procs, t0, nogb, rng)
        log("  verdict: %s   [%.1fs]" % (v, time.time() - t0))
    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
