#!/usr/bin/env python3
"""n4rec9.py -- exhibit a solution of step 1 and verify it against the TRUE
residual, then continue the recursion.

n4rec8.py eliminated the 280 level-4 unknowns of step 1 exactly and found the
reduced system in the 56 kernel parameters to be SOLVABLE -- its Groebner
basis consists of linear forms, so the solution set is an affine subspace.
So step 1 is not blocked; the blockage was an artifact of testing at the single point
y = 0, where every earlier script sat.

A claim of that size must not rest on the quadratic model, on the projection,
or on sympy.  So this script goes the other way round:

  1. it reconstructs a concrete y from the Groebner basis (an exact point of
     the reduced solution set) -- and checks that y against the reduced
     system it came from;
  2. it forms q1 = q + sum_i y_i k_i and solves for the level-4 part x by the
     TRUE residual, not the model: sum_j x_j M_j = -F(q1) with M_j the honest
     difference columns;
  3. it evaluates the TRUE residual at q2 = q1 + x and requires every step-1
     row to vanish EXACTLY.  Nothing of the machinery above enters this
     check: it is resid_full applied to a matrix of functions;
  4. it then continues the theta-recursion from q2 -- levels 4, 6, 8 -- and
     reports where it stands, the equations at level 8 being the closing
     conditions with no unknowns left to meet them.

Usage: ../venv/bin/python n4rec9.py [--steps 3]
Writes n4rec9-out.txt.
"""
import itertools
import random
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

import n4noeth as N                              # noqa: E402
import n4vac                                     # noqa: E402
import n4phi                                     # noqa: E402
import n4rec                                     # noqa: E402
import n4rec2                                    # noqa: E402
import n4rec6                                    # noqa: E402
import n4rec8                                    # noqa: E402
from n4rec import (qadd, qsmul, resid_full, q0_of, det2, step, dirs_lvl)
from n4rec2 import keep_step1                    # noqa: E402
from n4rec6 import build_base, build_model, selftest, dot, rf, qlin  # noqa
from n4rec8 import left_kernel, reduce_system    # noqa: E402
from n4noeth import mzero, C                     # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4rec9-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def main():
    t0 = time.time()
    procs = max(1, mp.cpu_count() - 2)
    rng = random.Random(7)
    STEPS = 3
    if "--steps" in sys.argv:
        STEPS = int(sys.argv[sys.argv.index("--steps") + 1])

    ms0 = n4vac.graded_monos(0, 0, -5, 1, mfree=(-1, -1))
    B = n4vac.constrained_basis(ms0)
    qb, kern = n4vac.build_embedded_base(B)
    A = [[qb[i][j].get((0, -1, -1, ()), C()) for j in (0, 1)]
         for i in (0, 1)]
    log("n4rec9: a solution of step 1, verified against the true residual.")
    log("")
    log("level-0 datum A = diag(%s, %s), det = %s"
        % (A[0][0], A[1][1], det2(A)))

    q, kmats, ds4 = build_base(A, procs)
    F0, L, Mx, Qd, Qo = build_model(q, kmats, ds4)
    rows = sorted({k for D in [F0] + list(L) + list(Mx) + list(Qd)
                   + list(Qo.values()) for k in D})
    log("model: %d rows, %d kernel + %d level-4 unknowns  [%.1fs]"
        % (len(rows), len(L), len(Mx), time.time() - t0))
    if selftest(q, kmats, ds4, F0, L, Mx, Qd, Qo, 3, rng):
        log("!! model self-test failed -- aborting")
        return
    log("model self-test at 3 random integer points: exact")

    mus = left_kernel(list(Mx), rows)
    const, lin, quad = reduce_system(mus, F0, L, Mx, Qd, Qo)
    nvar = len(L)
    qrows = {a for D in quad.values() for a in D}
    aff = set(range(len(mus))) - qrows
    log("reduced system: %d equations, %d affine, %d quadratic  [%.1fs]"
        % (len(mus), len(aff), len(qrows), time.time() - t0))

    # --- the affine part ----------------------------------------------------
    cols = [{a: v for a, v in lin.get(i, {}).items() if a in aff}
            for i in range(nvar)]
    target = {a: -v for a, v in const.items() if a in aff}
    y0 = N.sparse_solve(cols, target) if target else [C() for _ in range(nvar)]
    if y0 is None:
        log("!! affine rows inconsistent")
        return
    ker = N.sparse_null(cols)
    d = len(ker)
    log("affine solution set: %d-dimensional" % d)

    # --- substitute ---------------------------------------------------------
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
                        s = s + (c * u[i] * u[j] if ci == ei
                                 else c * (u[i] * v[j] + v[i] * u[j]))
                if s:
                    Aq.setdefault((ci, ei), {})[a] = s
    tvars = sorted(set(Al) | {c for ce in Aq for c in ce})
    log("after substitution: %d unknowns t, %d quadratic monomials  [%.1fs]"
        % (len(tvars), len(Aq), time.time() - t0))

    # --- the Groebner basis, and a point of the variety ---------------------
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
    gens = [tv[c] for c in tvars]
    G = sympy.groebner(P, *gens, order="grevlex", domain=sympy.QQ_I)
    gl = list(G.exprs)
    degs = sorted({sympy.total_degree(g) for g in gl})
    log("Groebner basis: %d elements, degrees %s" % (len(gl), degs))
    if degs != [1]:
        log("the basis is not linear; picking a point needs more than a"
            " linear solve -- stopping here")
        return

    # the basis generates the ideal, so V(P) = V(linear forms)
    lcols = [{} for _ in tvars]
    ltar = {}
    for a, g in enumerate(gl):
        p = sympy.Poly(g, *gens)
        cst = p.coeff_monomial(1)
        if cst:
            ltar[a] = -C(Q(sympy.re(cst)), Q(sympy.im(cst)))
        for n, c in enumerate(tvars):
            co = p.coeff_monomial(gens[n])
            if co:
                lcols[n][a] = C(Q(sympy.re(co)), Q(sympy.im(co)))
    tsol = N.sparse_solve(lcols, ltar)
    if tsol is None:
        log("!! the linear Groebner system has no solution -- contradiction")
        return
    log("a point of the reduced variety found  [%.1fs]" % (time.time() - t0))

    # check it against the ORIGINAL reduced polynomials
    subs = {gens[n]: sympy.Rational(tsol[n].re)
            + sympy.I * sympy.Rational(tsol[n].im)
            for n in range(len(tvars))}
    bad = sum(1 for p in P if sympy.simplify(p.subs(subs)) != 0)
    log("the point satisfies %d of the %d reduced equations" % (len(P) - bad,
                                                                len(P)))
    if bad:
        log("!! it does not satisfy them all -- aborting")
        return

    # --- reconstruct y ------------------------------------------------------
    tfull = [C() for _ in range(d)]
    for n, c in enumerate(tvars):
        tfull[c] = tsol[n]
    y = list(y0)
    for ci in range(d):
        if tfull[ci]:
            u = ker[ci]
            for i in range(nvar):
                if u[i]:
                    y[i] = y[i] + tfull[ci] * u[i]
    nz = sum(1 for v in y if v)
    log("y reconstructed: %d of the %d kernel parameters nonzero"
        % (nz, nvar))

    # --- the true residual at q1 = q + y.k ----------------------------------
    q1 = qadd(q, qlin(kmats, y))
    G1 = rf(q1)
    log("TRUE residual at q1: %d nonzero step-1 rows  (was %d at y = 0)"
        % (len(G1), len(F0)))

    # honest level-4 columns, from the true residual
    Mtrue = []
    for D in ds4:
        Fp = rf(qadd(q1, D))
        col = {}
        for k in set(Fp) | set(G1):
            v = Fp.get(k, C()) - G1.get(k, C())
            if v:
                col[k] = v
        Mtrue.append(col)
    x = N.sparse_solve(Mtrue, {k: -v for k, v in G1.items()})
    if x is None:
        log("!! no level-4 completion at this y -- the model and the true")
        log("   residual disagree; aborting")
        return
    log("level-4 completion found  [%.1fs]" % (time.time() - t0))

    q2 = qadd(q1, qlin(ds4, x))
    F2 = rf(q2)
    log("")
    log("=== THE CHECK ===================================================")
    log("TRUE residual at q2 on the step-1 rows: %d nonzero" % len(F2))
    if F2:
        log("!! step 1 is NOT solved -- the reduction is wrong somewhere")
        return
    log("step 1 is SOLVED exactly: every equation at theta-level 2 and every")
    log("constraint at level 3 vanishes.")
    log("=================================================================")
    import pickle
    with open("n4rec9-q2.pkl", "wb") as fh:
        pickle.dump({"A": A, "q2": q2, "y": y, "x": x}, fh)
    log("q2 written to n4rec9-q2.pkl")

    # --- continue the recursion --------------------------------------------
    log("")
    log("continuing the recursion from q2:")
    qq = q2
    for r in (2, 3, 4)[:STEPS]:
        part, kr = step(qq, r, procs, verbose=False)
        if part is None:
            log("   level %d -> %d: INCONSISTENT -- the recursion dies here"
                % (2 * r, 2 * r + 2))
            bad = n4phi.diagnose(
                *_sys(qq, r, procs)) if False else None
            log("   [%.1fs]" % (time.time() - t0))
            break
        log("   level %d -> %d: solvable, kernel %d  [%.1fs]"
            % (2 * r, 2 * r + 2, len(kr), time.time() - t0))
        qq = qadd(qq, part)
    else:
        Fall = resid_full(qq)
        log("   reached the top; %d nonzero rows in all" % len(Fall))
        left = {k: v for k, v in Fall.items() if k[0] == "E"}
        log("   closing check: %d nonzero equation keys" % len(left))
    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
