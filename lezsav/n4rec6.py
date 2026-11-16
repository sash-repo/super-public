#!/usr/bin/env python3
"""n4rec6.py -- eliminate the affine unknowns of step 1 and decide what is
left.

n4rec5.py left the N=4 vacuum as ONE finite object: step 1 of the
theta-recursion is a quadratic system in 56 + 280 unknowns,

    q = A/(mw) + p + sum_i y_i k_i + x,

quadratic in the 56 kernel parameters y of step 0 and AFFINE in the 280
level-4 unknowns x.  Its affine rows alone are solvable, so nothing is
settled by them; the quadratic rows decide.  This does the elimination
taken here.

WHAT IS DONE HERE.

 1. The model is built EXACTLY, not sampled:

        F(y, x) = F0 + sum_i y_i L_i + sum_j x_j M_j
                     + sum_i y_i^2 Q_ii + sum_{i<j} y_i y_j Q_ij,

    with L_i the symmetric difference (exact for a quadratic map), Q_ii the
    second symmetric difference and Q_ij the mixed one.  It is then
    SELF-TESTED against the true residual at random integer points (y, x):
    the model must reproduce it key by key.  That one test certifies at once
    that there is no cubic term in y, no x*x term and no y*x cross term --
    the three assumptions every earlier script made and none verified.

 2. On the blocked sector S -- the rows the linear system at y = 0 cannot
    reach -- the elimination of x is EXACT and it is an equivalence: for a
    fixed y the sector is linear in x, so it is solvable in x if and only if
    its right-hand side is orthogonal to the left kernel of the x-block.
    Projecting on a basis of that left kernel turns the sector into a
    quadratic system in y ALONE, equivalent to it.

 3. That reduced system is then attacked in three ways, in order of
    strength: its affine rows (a linear system -- inconsistency here is a
    proof of emptiness); the linearization relaxation (treat every monomial
    y_i y_j as a free unknown -- inconsistency here is again a proof, and a
    stronger one, since the relaxation is weaker than the system); and, if
    both survive, a Groebner basis of the reduced system over Q(i).

Any of the three returning "inconsistent" proves that NO q of the graded
class with this level-0 datum solves (IK4).

Exact over Q(i) throughout.

Usage:  ../venv/bin/python n4rec6.py [--all] [--trials 5] [--nogb]
Writes n4rec6-out.txt.
"""
import itertools
import random
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import n4noeth as N
import n4vac
import n4phi
import n4rec
import n4rec2
from n4rec import (dirs_lvl, qadd, qsmul, resid_full, q0_of, step, det2)
from n4rec2 import keep_step1
from n4noeth import mzero, C

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4rec6-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


# --- dictionaries over the step-1 rows --------------------------------------

def rf(q):
    """The step-1 rows of the residual at q."""
    return {k: v for k, v in resid_full(q).items() if keep_step1(k)}


def dcomb(*pairs):
    """sum c * D over (c, D) pairs, dropping zeros."""
    out = {}
    for c, D in pairs:
        c = C(c) if not isinstance(c, C) else c
        for k, v in D.items():
            w = out.get(k)
            out[k] = (w + c * v) if w is not None else c * v
    return {k: v for k, v in out.items() if v}


def dot(mu, D):
    """<mu, D> for mu a dict row -> C and D a column dict."""
    s = C()
    for k, v in D.items():
        m = mu.get(k)
        if m:
            s = s + m * v
    return s


def qlin(kmats, y):
    M = mzero()
    for i, c in enumerate(y):
        if c:
            M = qadd(M, qsmul(c, kmats[i]))
    return M


# --- the model --------------------------------------------------------------

def build_base(A, procs):
    """q = A/(mw) + p, the 56 kernel matrices, the 280 level-4 directions."""
    p, ker0 = step(q0_of(A), 0, procs, verbose=False)
    if p is None:
        return None
    ds2 = dirs_lvl(1)
    kmats = []
    for v in ker0:
        M = mzero()
        for i, c in enumerate(v):
            if c:
                M = qadd(M, qsmul(c, ds2[i]))
        kmats.append(M)
    return qadd(q0_of(A), p), kmats, dirs_lvl(2)


def build_model(q, kmats, ds4):
    """F0, L (linear in y), M (affine in x), Qd (y_i^2), Qo (y_i y_j)."""
    F0 = rf(q)
    Fp, Fm = [], []
    for K in kmats:
        Fp.append(rf(qadd(q, K)))
        Fm.append(rf(qadd(q, qsmul(-1, K))))
    L = [dcomb((Q(1, 2), Fp[i]), (Q(-1, 2), Fm[i]))
         for i in range(len(kmats))]
    Qd = [dcomb((Q(1, 2), Fp[i]), (Q(1, 2), Fm[i]), (-1, F0))
          for i in range(len(kmats))]
    Mx = [dcomb((1, rf(qadd(q, D))), (-1, F0)) for D in ds4]
    Qo = {}
    for i, j in itertools.combinations(range(len(kmats)), 2):
        Qo[(i, j)] = dcomb((1, rf(qadd(qadd(q, kmats[i]), kmats[j]))),
                           (-1, Fp[i]), (-1, Fp[j]), (1, F0))
    return F0, L, Mx, Qd, Qo


def model_at(F0, L, Mx, Qd, Qo, y, x):
    terms = [(1, F0)]
    for i, c in enumerate(y):
        if c:
            terms.append((c, L[i]))
            terms.append((c * c, Qd[i]))
    for j, c in enumerate(x):
        if c:
            terms.append((c, Mx[j]))
    for (i, j), D in Qo.items():
        c = y[i] * y[j]
        if c:
            terms.append((c, D))
    return dcomb(*terms)


def selftest(q, kmats, ds4, F0, L, Mx, Qd, Qo, trials, rng):
    """The model must reproduce the true residual at random integer points."""
    bad = 0
    for _ in range(trials):
        y = [rng.randint(-2, 2) for _ in kmats]
        x = [0] * len(ds4)
        for _ in range(12):
            x[rng.randrange(len(ds4))] = rng.randint(-2, 2)
        qq = qadd(qadd(q, qlin(kmats, y)), qlin(ds4, x))
        true = rf(qq)
        pred = model_at(F0, L, Mx, Qd, Qo, y, x)
        diff = dcomb((1, true), (-1, pred))
        if diff:
            bad += 1
    return bad


# --- the blocked sector -----------------------------------------------------

def blocked_sector(F0, L, Mx):
    cols = list(L) + list(Mx)
    target = {k: -v for k, v in F0.items()}
    if N.sparse_solve(cols, target) is not None:
        return None
    S = set()
    for rows, _ in n4phi.diagnose(cols, target):
        S |= set(rows)
    return S


def restrict(D, S):
    return {k: v for k, v in D.items() if k in S}


# --- deciding a quadratic system --------------------------------------------

def relaxation(const, lin, quad):
    """Treat every monomial as a free unknown: sum lin_i u_i + sum quad_m v_m
    = -const.  Unsolvable => the quadratic system is unsolvable."""
    cols = list(lin.values()) + list(quad.values())
    cols = [c for c in cols if c]
    target = {k: -v for k, v in const.items()}
    if not target:
        return True
    return N.sparse_solve(cols, target) is not None


def affine_rows(const, lin, quad, rows):
    """The rows on which every quadratic coefficient vanishes."""
    out = set(rows)
    for D in quad.values():
        out -= set(D)
    return out


def main():
    t0 = time.time()
    procs = max(1, mp.cpu_count() - 2)
    TR = 5
    if "--trials" in sys.argv:
        TR = int(sys.argv[sys.argv.index("--trials") + 1])
    rng = random.Random(2026)

    ms0 = n4vac.graded_monos(0, 0, -5, 1, mfree=(-1, -1))
    B = n4vac.constrained_basis(ms0)
    qb, kern = n4vac.build_embedded_base(B)
    Aemb = [[qb[i][j].get((0, -1, -1, ()), C()) for j in (0, 1)]
            for i in (0, 1)]

    trials = [("embedded diag(2,2)", Aemb)]
    if "--all" in sys.argv:
        trials += [("identity", [[C(1), C()], [C(), C(1)]]),
                   ("diag(1,-1)", [[C(1), C()], [C(), C(-1)]]),
                   ("diag(1,i)", [[C(1), C()], [C(), C(0, 1)]]),
                   ("antidiag", [[C(), C(1)], [C(1), C()]]),
                   ("upper unitriang", [[C(1), C(1)], [C(), C(1)]])]
        for t in range(3):
            while True:
                A = [[C(rng.randint(-3, 3), rng.randint(-3, 3))
                      for _ in (0, 1)] for _ in (0, 1)]
                if det2(A):
                    break
            trials.append(("random %d" % (t + 1), A))

    log("n4rec6: step 1 of the theta-recursion, its affine unknowns")
    log("eliminated exactly, and the quadratic system that is left decided.")
    log("")

    for name, A in trials:
        log("=" * 68)
        log("level-0 datum: %s   det = %s" % (name, det2(A)))
        run_one(A, procs, TR, rng, t0)
        log("[%.1fs]" % (time.time() - t0))
    log("")
    log("[%.1fs] done" % (time.time() - t0))


def run_one(A, procs, TR, rng, t0):
    base = build_base(A, procs)
    if base is None:
        log("  step 0 inconsistent -- nothing to eliminate")
        return
    q, kmats, ds4 = base
    log("  step 0 solved, kernel %d, level-4 unknowns %d  [%.1fs]"
        % (len(kmats), len(ds4), time.time() - t0))

    F0, L, Mx, Qd, Qo = build_model(q, kmats, ds4)
    nrows = len({k for D in [F0] + L + Mx + Qd + list(Qo.values())
                 for k in D})
    log("  model built: %d rows, %d linear + %d affine + %d quadratic"
        " columns  [%.1fs]"
        % (nrows, len(L), len(Mx), len(Qd) + len(Qo), time.time() - t0))

    bad = selftest(q, kmats, ds4, F0, L, Mx, Qd, Qo, TR, rng)
    log("  self-test at %d random integer points (y, x): %d mismatches"
        % (TR, bad))
    if bad:
        log("  !! the quadratic model is WRONG -- everything below is void")
        return
    log("     => no cubic term in y, no x*x, no y*x cross term: the model is"
        " exact")

    S = blocked_sector(F0, L, Mx)
    if S is None:
        log("  the linear system at y = 0 is solvable -- no blocked sector")
        return
    log("  blocked sector: %d rows  [%.1fs]" % (len(S), time.time() - t0))

    # --- restrict everything to the sector ---------------------------------
    F0s = restrict(F0, S)
    Ls = {i: restrict(D, S) for i, D in enumerate(L)}
    Ls = {i: D for i, D in Ls.items() if D}
    Mxs = {j: restrict(D, S) for j, D in enumerate(Mx)}
    Mxs = {j: D for j, D in Mxs.items() if D}
    Qds = {i: restrict(D, S) for i, D in Qd.items()} if isinstance(Qd, dict) \
        else {i: restrict(D, S) for i, D in enumerate(Qd)}
    Qds = {i: D for i, D in Qds.items() if D}
    Qos = {ij: restrict(D, S) for ij, D in Qo.items()}
    Qos = {ij: D for ij, D in Qos.items() if D}
    yidx = set(Ls) | set(Qds)
    for (i, j) in Qos:
        yidx |= {i, j}
    log("  on the sector: %d of the 56 kernel parameters occur, %d of the"
        " 280 level-4 unknowns" % (len(yidx), len(Mxs)))
    log("  quadratic columns on the sector: %d diagonal, %d mixed"
        % (len(Qds), len(Qos)))

    # --- eliminate x: the left kernel of the x-block -----------------------
    Srows = sorted(S)
    ridx = {k: i for i, k in enumerate(Srows)}
    jlist = sorted(Mxs)
    cols_for_null = []
    for k in Srows:
        c = {}
        for n, j in enumerate(jlist):
            v = Mxs[j].get(k)
            if v:
                c[n] = v
        cols_for_null.append(c)
    mus = N.sparse_null(cols_for_null)
    log("  left kernel of the x-block: %d vectors (%d rows - rank %d)"
        % (len(mus), len(Srows), len(Srows) - len(mus)))
    if not mus:
        log("  the x-block has full row rank: the sector imposes nothing"
            " on y")
        return

    # --- the reduced system, in y alone ------------------------------------
    red_c, red_l, red_q = {}, {}, {}
    for a, mu in enumerate(mus):
        m = {Srows[i]: c for i, c in enumerate(mu) if c}
        v = dot(m, F0s)
        if v:
            red_c[a] = v
        for i, D in Ls.items():
            v = dot(m, D)
            if v:
                red_l.setdefault(i, {})[a] = v
        for i, D in Qds.items():
            v = dot(m, D)
            if v:
                red_q.setdefault((i, i), {})[a] = v
        for ij, D in Qos.items():
            v = dot(m, D)
            if v:
                red_q.setdefault(ij, {})[a] = v
    nq_rows = {a for D in red_q.values() for a in D}
    log("  reduced system: %d equations in y alone; %d of them carry a"
        " quadratic part" % (len(mus), len(nq_rows)))
    log("     unknowns really occurring: %d"
        % len(set(red_l) | {i for ij in red_q for i in ij}))

    # --- 1) the affine rows ------------------------------------------------
    aff = set(range(len(mus))) - nq_rows
    aff_c = {a: v for a, v in red_c.items() if a in aff}
    aff_l = {i: {a: v for a, v in D.items() if a in aff} for i, D in
             red_l.items()}
    aff_l = {i: D for i, D in aff_l.items() if D}
    log("  1) the %d AFFINE reduced equations:" % len(aff))
    if not aff_c:
        log("     homogeneous -- y = 0 satisfies them; no proof here")
    else:
        cols = list(aff_l.values())
        target = {a: -v for a, v in aff_c.items()}
        if N.sparse_solve(cols, target) is None:
            log("     INCONSISTENT  ==>  PROOF: no q of the graded class"
                " with this level-0 datum solves (IK4)")
            return
        log("     solvable -- no proof here")

    # --- 2) the linearization relaxation -----------------------------------
    log("  2) the linearization relaxation of the reduced system:")
    if not relaxation(red_c, red_l, red_q):
        log("     INCONSISTENT  ==>  PROOF: no q of the graded class"
            " with this level-0 datum solves (IK4)")
        return
    log("     solvable -- no proof here (the relaxation is weaker than the"
        " system)")

    # --- 3) hand the reduced system to a Groebner basis --------------------
    log("  3) the reduced quadratic system, for a Groebner basis:")
    ys = sorted(set(red_l) | {i for ij in red_q for i in ij})
    log("     %d equations, %d unknowns, %d distinct monomials"
        % (len(mus), len(ys), len(ys) + len(red_q)))
    save_system(red_c, red_l, red_q, ys, len(mus))
    if "--nogb" not in sys.argv:
        groebner(red_c, red_l, red_q, ys, len(mus))


def save_system(red_c, red_l, red_q, ys, neq):
    """Write the reduced system out, so it can be attacked separately."""
    import json
    obj = {"neq": neq, "ys": ys,
           "const": {str(a): [str(v.re), str(v.im)]
                     for a, v in red_c.items()},
           "lin": {str(i): {str(a): [str(v.re), str(v.im)]
                            for a, v in D.items()}
                   for i, D in red_l.items()},
           "quad": {"%d,%d" % ij: {str(a): [str(v.re), str(v.im)]
                                   for a, v in D.items()}
                    for ij, D in red_q.items()}}
    with open("n4rec6-system.json", "w") as fh:
        json.dump(obj, fh)
    log("     written to n4rec6-system.json")


def groebner(red_c, red_l, red_q, ys, neq):
    try:
        import sympy
    except ImportError:
        log("     sympy not available")
        return
    t1 = time.time()
    yv = {i: sympy.Symbol("y%d" % i) for i in ys}
    polys = {}
    for a, v in red_c.items():
        polys[a] = polys.get(a, 0) + sympy.Rational(v.re) \
            + sympy.I * sympy.Rational(v.im)
    for i, D in red_l.items():
        for a, v in D.items():
            polys[a] = polys.get(a, 0) + (sympy.Rational(v.re)
                                          + sympy.I * sympy.Rational(v.im)) \
                * yv[i]
    for (i, j), D in red_q.items():
        for a, v in D.items():
            polys[a] = polys.get(a, 0) + (sympy.Rational(v.re)
                                          + sympy.I * sympy.Rational(v.im)) \
                * yv[i] * yv[j]
    P = [sympy.expand(p) for p in polys.values() if p != 0]
    log("     %d nonzero polynomials handed to sympy" % len(P))
    try:
        G = sympy.groebner(P, *[yv[i] for i in ys], order="grevlex",
                           domain=sympy.QQ_I)
    except Exception as e:                      # noqa: BLE001
        log("     Groebner failed: %s" % e)
        return
    gl = list(G.exprs)
    log("     Groebner basis: %d elements  [%.1fs]"
        % (len(gl), time.time() - t1))
    if len(gl) == 1 and gl[0].is_number and gl[0] != 0:
        log("     the basis is {1}  ==>  PROOF: the reduced system has NO"
            " solution, so no q of the graded class with this level-0 datum"
            " solves (IK4)")
    else:
        log("     the basis is nontrivial: the system HAS solutions over the"
            " algebraic closure; degrees %s"
            % sorted({sympy.total_degree(g) for g in gl}))


if __name__ == "__main__":
    main()
