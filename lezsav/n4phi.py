#!/usr/bin/env python3
"""n4phi.py -- the N=4 vacuum by prepotential Newton over a DEEP graded window.

THE POINT.  Two facts join here:
  (1) corrections dq = bD_{+g} cD_-^b Phi satisfy both constraints of (IK4)
      identically for EVERY Phi (n4noeth.py section (2)), so Newton over
      Phi-space never leaves the constraint variety -- only E1+E2 remain;
  (2) the exact N=2 vacuum E = 2/(m Omega) is the RECIPROCAL of a polynomial,
      whose theta^{2k}-level sits at w^{-1-k}: the vacuum lives DEEPER in w
      than the polynomial windows n4vac.py's q-space Newton searched.
So: start from the exactly-constrained embedded base (pi(q) = diag(E, Eb),
n4vac.build_embedded_base), and Newton over Phi-monomials z^0 w^b m^{-1}
theta^S with b down to W_LO and S over ALL even subsets, plus the polynomial
kernel directions.  Everything exact over Q(i); the Jacobian is built in
parallel.

Usage:  ../venv/bin/python n4phi.py     (writes n4phi-out.txt, -state.pkl)
"""

import itertools
import multiprocessing as mp
import pickle
import sys
import time

from fractions import Fraction as Q

import n4noeth as N
import n4vac
from n4noeth import (mono, add, sub, smul, mul, mzero, madd, msub, msmul,
                     mid, mmul, mapply, minv, meq, cDm, bDm, cDp, bDp,
                     C, MI, PL, M_MONO,
                     lhs1, lhs2, rhs2, var_lhs1, var_lhs2, var_rhs2,
                     sparse_solve, q_of_phi)
from n4emb import I

W_LO, W_HI = -5, 1
Z_LO, Z_HI = 0, 0          # override:  python3 n4phi.py --z -1 1

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4phi-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


# --- residuals (E1 + E2 only; constraints are asserted) ---------------------

def stack(scalars):
    out = {}
    for slot, f in enumerate(scalars):
        for k, v in f.items():
            out[(slot, k)] = v
    return out


def flat(Ms):
    L = []
    for M in Ms:
        for i in (0, 1):
            for j in (0, 1):
                L.append(M[i][j])
    return L


def resid_vec(q, qi):
    R1 = lhs1(q, qi)
    R2 = msub(lhs2(q, qi), rhs2(q))
    return stack(flat([R1, R2]))


def var_resid_vec(q, qi, dq):
    dR1 = var_lhs1(q, qi, dq)
    dR2 = msub(var_lhs2(q, qi, dq), var_rhs2(dq))
    return stack(flat([dR1, dR2]))


# --- the direction set ------------------------------------------------------

def phi_monomials():
    out = []
    for r in (0, 2, 4, 6, 8):
        for S in itertools.combinations(MI + PL, r):
            for a in range(Z_LO, Z_HI + 1):
                for b in range(W_LO, W_HI + 1):
                    out.append(mono(a, b, -1, S))
    return out


LEAD_KEYS = {(0, -1, -1, ())}      # the one theta-free key q may carry


def safe_direction(dq):
    """Keep the theta-free part of det q a single monomial: allow theta-free
    terms only on the diagonal and only on the w^{-1} m^{-1} lead."""
    for i in (0, 1):
        for j in (0, 1):
            for k in dq[i][j]:
                if k[3] == ():
                    if i != j or k not in LEAD_KEYS:
                        return False
    return True


# --- parallel Jacobian ------------------------------------------------------

_G = {}


def _init(qp, qip, dirsp):
    _G["q"] = pickle.loads(qp)
    _G["qi"] = pickle.loads(qip)
    _G["dirs"] = pickle.loads(dirsp)


def _col(j):
    return j, var_resid_vec(_G["q"], _G["qi"], _G["dirs"][j])


def jacobian(q, qi, dirs, procs):
    qp, qip, dirsp = (pickle.dumps(x) for x in (q, qi, dirs))
    cols = [None] * len(dirs)
    with mp.Pool(procs, _init, (qp, qip, dirsp)) as pool:
        for j, col in pool.imap_unordered(_col, range(len(dirs)),
                                          chunksize=8):
            cols[j] = col
    return cols


# --- exact Gauss-Newton on an inconsistent component ------------------------

def cconj(z):
    return C(z.re, -z.im)


def gn_solve(cols, target):
    """Solve consistent components exactly; on an inconsistent one solve the
    normal equations A^H A x = A^H t over Q(i) (always consistent): the step
    that kills the reachable part of the residual."""
    comp = N._components(cols, target.keys())
    keycomp = {}
    for root, members in comp.items():
        for mm in members:
            if mm[0] == "k":
                keycomp[mm[1]] = root
    groups = {}
    for j, col in enumerate(cols):
        if col:
            root = keycomp[next(iter(col.keys()))]
            groups.setdefault(root, []).append(j)
    tgroups = {}
    for k, v in target.items():
        tgroups.setdefault(keycomp[k], {})[k] = v
    x = [C() for _ in cols]
    n_gn = 0
    for root, tg in tgroups.items():
        js = groups.get(root, [])
        if not js:
            n_gn += 1          # untouchable sector: no step can move it
            continue
        sol = N.sparse_solve([cols[j] for j in js], tg)
        if sol is None:
            n_gn += 1
            keys = sorted({k for j in js for k in cols[j]} | set(tg))
            kidx = {k: i for i, k in enumerate(keys)}
            A = []             # columns as dense C-vectors
            for j in js:
                v = [C() for _ in keys]
                for k, c in cols[j].items():
                    v[kidx[k]] = c
                A.append(v)
            t = [tg.get(k, C()) for k in keys]
            nc = len(js)
            M = [[sum((cconj(A[p][r]) * A[q][r] for r in range(len(keys))),
                      C()) for q in range(nc)] for p in range(nc)]
            b = [sum((cconj(A[p][r]) * t[r] for r in range(len(keys))), C())
                 for p in range(nc)]
            sol = N.csolve([[M[p][q] for p in range(nc)]
                            for q in range(nc)], b)
            if sol is None:
                continue       # cannot happen for true normal equations
        for j, c in zip(js, sol):
            x[j] = x[j] + c
    return x, n_gn


def fnorm(R):
    s = 0.0
    for v in R.values():
        s += float(v.re) ** 2 + float(v.im) ** 2
    return s ** 0.5


# --- diagnostics on an inconsistent step ------------------------------------

def diagnose(cols, target):
    comp = N._components(cols, target.keys())
    keycomp = {}
    for root, members in comp.items():
        for mm in members:
            if mm[0] == "k":
                keycomp[mm[1]] = root
    groups = {}
    for j, col in enumerate(cols):
        if col:
            root = keycomp[next(iter(col.keys()))]
            groups.setdefault(root, []).append(j)
    tgroups = {}
    for k, v in target.items():
        tgroups.setdefault(keycomp[k], {})[k] = v
    bad = []
    for root, tg in tgroups.items():
        js = groups.get(root, [])
        sub_target = tg
        if not js:
            bad.append((sorted(sub_target), "no columns touch this sector"))
            continue
        x = N.sparse_solve([cols[j] for j in js], sub_target)
        if x is None:
            bad.append((sorted(sub_target), "%d columns, no solution"
                        % len(js)))
    return bad


# --- main -------------------------------------------------------------------

def main():
    global Z_LO, Z_HI
    if "--z" in sys.argv:
        i = sys.argv.index("--z")
        Z_LO, Z_HI = int(sys.argv[i + 1]), int(sys.argv[i + 2])
    t0 = time.time()
    procs = max(1, mp.cpu_count() - 2)
    log("n4phi: prepotential Newton, z-window [%d, %d], w-window [%d, %d],"
        " %d procs" % (Z_LO, Z_HI, W_LO, W_HI, procs))

    ms = n4vac.graded_monos(Z_LO, Z_HI, W_LO, W_HI, mfree=(-1, -1))
    log("entry monomials: %d" % len(ms))
    B = n4vac.constrained_basis(ms)
    log("constrained basis: %d matrices  [%.1fs]" % (len(B), time.time() - t0))
    q0, kern = n4vac.build_embedded_base(B)
    log("embedded base built, kernel %d  [%.1fs]" % (len(kern),
                                                     time.time() - t0))
    nc = N.check_constraints(q0)
    log("base constraint violations: %d (must be 0)" % nc)
    if nc:
        return 1

    dirs = []
    for Phi in phi_monomials():
        dq = q_of_phi(Phi)
        if any(dq[i][j] for i in (0, 1) for j in (0, 1)):
            dirs.append(dq)
    dirs += B          # the FULL constrained space over the window: the
    # pi-part (diagonal couplings included) must be free, not only the
    # pi-null kernel -- kern is a subset of span(B) and is not re-added
    dirs = [d for d in dirs if safe_direction(d)]
    log("directions: %d (prepotential + constrained basis, theta-free-safe)"
        % len(dirs))

    q = q0
    hist = []
    for it in range(1, 16):
        assert N.check_constraints(q) == 0, "left the constraint variety"
        qi = minv(q)
        R = resid_vec(q, qi)
        nrm = fnorm(R)
        hist.append(nrm)
        log("iter %d: %d residual keys, |R| = %.6g  [%.1fs]"
            % (it, len(R), nrm, time.time() - t0))
        with open("n4phi-state.pkl", "wb") as fh:
            pickle.dump({"q": q, "iter": it, "resid_keys": len(R)}, fh)
        if not R:
            log("EXACT N=4 VACUUM FOUND at iteration %d" % it)
            with open("n4phi-state.pkl", "wb") as fh:
                pickle.dump({"q": q, "iter": it, "resid_keys": 0,
                             "solved": True}, fh)
            return 0
        if len(hist) >= 4 and all(h >= hist[-4] * 0.999 for h in hist[-3:]):
            log("   |R| stagnant over 3 iterations -- stopping; the blocked"
                " sector is an invariant obstruction in this window")
            for k, v in sorted(R.items(), key=str)[:20]:
                log("     slot %s key %s: %s" % (k[0], k[1], v))
            with open("n4phi-state.pkl", "wb") as fh:
                pickle.dump({"q": q, "iter": it, "stagnant": True,
                             "resid": R}, fh)
            return 1
        cols = jacobian(q, qi, dirs, procs)
        log("   jacobian built  [%.1fs]" % (time.time() - t0))
        target = {k: -v for k, v in R.items()}
        x, n_gn = gn_solve(cols, target)
        nnz = sum(1 for c in x if c)
        log("   step: %d nonzero updates, %d Gauss-Newton sectors  [%.1fs]"
            % (nnz, n_gn, time.time() - t0))
        if nnz == 0:
            log("   zero step with nonzero residual -- stalled; dump:")
            for k, v in sorted(R.items(), key=str)[:20]:
                log("     slot %s key %s: %s" % (k[0], k[1], v))
            return 1
        for c, d in zip(x, dirs):
            if c:
                q = madd(q, msmul(c, d))
    log("no convergence in 15 iterations")
    return 1


if __name__ == "__main__":
    sys.exit(main())
