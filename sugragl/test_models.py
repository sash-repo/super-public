#!/usr/bin/env python3
"""
test_models.py -- self-tests for the engine of models.py / cohom.py.

Nothing here is calibrated against the paper: these are internal checks that
the algebra is an algebra and the differential is a differential.  The external
calibration -- reproducing the six columns of the Table that Grozman computed
with the SuperLie package -- is in run.py.
"""

import sys
from fractions import Fraction
from itertools import product

from models import Prolong, Model, blocks_of, backend
from cohom import Complex, c1_items, c2_items

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


# --------------------------------------------------------------- the algebra


def test_blocks():
    check("blocks (00100)", blocks_of("00100") == [(2, 0), (1, 1), (2, 0)])
    check("blocks (111100001111)",
          blocks_of("111100001111") == [(2, 1), (2, 1), (2, 0), (2, 0),
                                        (2, 1), (2, 1)])
    m = Model("00100")
    check("sl(2) factors of (00100)", m.sl2 == [(0, 1), (3, 4)])
    check("depth of (00100)", max(m.b) == 2)


def test_jacobi(par):
    """super Jacobi for g_- and for [g_0, g_-]."""
    P = Prolong(par, "conf")
    n = P.ngm
    bad = 0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                acc = {}
                for (x, y, z, s) in ((a, b, c, 1),):
                    pass
                # [[a,b],c] - (-1)^{p_a p_b}[[b,a],c] is automatic; test
                # [a,[b,c]] = [[a,b],c] + (-1)^{p_a p_b}[b,[a,c]]
                lhs = {}
                for u, l1 in P.f[b][c].items():
                    for v, l2 in P.f[a][u].items():
                        lhs[v] = lhs.get(v, 0) + l1 * l2
                rhs = {}
                for u, l1 in P.f[a][b].items():
                    for v, l2 in P.f[u][c].items():
                        rhs[v] = rhs.get(v, 0) + l1 * l2
                s = (-1) ** (P.gm_par[a] * P.gm_par[b])
                for u, l1 in P.f[a][c].items():
                    for v, l2 in P.f[b][u].items():
                        rhs[v] = rhs.get(v, 0) + s * l1 * l2
                keys = set(lhs) | set(rhs)
                if any(lhs.get(k, 0) != rhs.get(k, 0) for k in keys):
                    bad += 1
    check(f"Jacobi in g_- for {par}", bad == 0, f"({bad} triples)")


def test_antisym(par):
    P = Prolong(par, "conf")
    bad = 0
    for a in range(P.ngm):
        for b in range(P.ngm):
            s = -(-1) ** (P.gm_par[a] * P.gm_par[b])
            ba = {k: s * v for k, v in P.f[b][a].items()}
            if {k: v for k, v in P.f[a][b].items() if v} != \
               {k: v for k, v in ba.items() if v}:
                bad += 1
    check(f"super-antisymmetry of the bracket for {par}", bad == 0)


def test_prolong_derivation(par, kind):
    """Re-check, independently of how it was found, that every basis vector of
    g_k is a derivation."""
    P = Prolong(par, kind)
    bad = 0
    for k in sorted(P.basis):
        if k <= 0:
            continue
        for idx in range(P.dim(k)):
            act = P.basis[k][idx]
            for a in range(P.ngm):
                for b in range(a, P.ngm):
                    lhs = {}
                    for c, lam in P.f[a][b].items():
                        for t, v in act[c].items():
                            lhs[t] = lhs.get(t, 0) + lam * v
                    rhs = {}
                    sgn = (-1) ** (P.gm_par[a] * P.gm_par[b])
                    for (x, y, s) in ((a, b, 1), (b, a, -sgn)):
                        m = k + P.gm_deg[x]
                        for t0, cf in act[x].items():
                            for t, v in P.rb[m][t0].get(y, {}).items():
                                rhs[t] = rhs.get(t, 0) + s * cf * v
                    keys = set(lhs) | set(rhs)
                    if any(lhs.get(t, 0) != rhs.get(t, 0) for t in keys):
                        bad += 1
    check(f"prolong of {par} ({kind}) consists of derivations", bad == 0,
          f"({bad} failures)")


def test_module(par, kind):
    """[x,[y,m]] - (-1)^{p_x p_y}[y,[x,m]] = [[x,y],m] on all of P."""
    P = Prolong(par, kind)
    bad = 0
    for m in sorted(P.basis):
        for idx in range(P.dim(m)):
            for a in range(P.ngm):
                for b in range(P.ngm):
                    lhs = {}
                    for (x, y, s) in ((a, b, 1),
                                      (b, a, -(-1) ** (P.gm_par[a]
                                                       * P.gm_par[b]))):
                        for t, v in P.left_bracket(y, m, idx).items():
                            mm = m + P.gm_deg[y]
                            for t2, v2 in P.left_bracket(x, mm, t).items():
                                lhs[t2] = lhs.get(t2, 0) + s * v * v2
                    rhs = {}
                    for c, lam in P.f[a][b].items():
                        for t, v in P.left_bracket(c, m, idx).items():
                            rhs[t] = rhs.get(t, 0) + lam * v
                    keys = set(lhs) | set(rhs)
                    if any(lhs.get(t, 0) != rhs.get(t, 0) for t in keys):
                        bad += 1
    check(f"P is a g_- module for {par} ({kind})", bad == 0, f"({bad})")


def test_dd(par, kind):
    """d2 o d1 = 0 on every C^1 basis vector."""
    P = Prolong(par, kind)
    C = Complex(P)
    bad = 0
    for (a, m, idx, deg, w) in c1_items(P):
        col = C.d1_column(a, m, idx)
        if not col:
            continue
        pc = (P.pars[m][idx] + P.gm_par[a]) % 2
        vals = {k: dict(v) for k, v in col.items()}
        # the parity of d1(c) is that of c, shifted by 1 in the Koszul sense;
        # in the multilinear-map convention p(dc) = p(c)
        cand = set()
        for x in range(P.ngm):
            for (u, v) in vals:
                for t in (tuple(sorted((x, u, v))),):
                    if C.ok3(*t):
                        cand.add(t)
        for x in range(P.ngm):
            for y in range(P.ngm):
                for c in P.f[x][y]:
                    for (u, v) in vals:
                        for z in (u, v):
                            if c in (u, v):
                                t = tuple(sorted((x, y, z)))
                                if C.ok3(*t):
                                    cand.add(t)
        for tri in cand:
            r = C.eval_d2(vals, pc, *tri)
            if r:
                bad += 1
                if bad < 4:
                    print(f"       d2 d1 != 0 at {tri}: {r}")
    check(f"d2 o d1 = 0 for {par} ({kind})", bad == 0, f"({bad} triples)")


if __name__ == "__main__":
    print(f"exact linear algebra backend: {backend()}")
    print("--- combinatorics ---")
    test_blocks()
    print("--- the algebra g_- ---")
    for par in ("00100", "00001", "001100", "100001"):
        test_antisym(par)
        test_jacobi(par)
    print("--- the prolong ---")
    for par in ("0000", "00100", "00001", "001100"):
        for kind in ("riem", "conf"):
            test_prolong_derivation(par, kind)
    print("--- the module structure ---")
    for par in ("0000", "00100"):
        for kind in ("riem", "conf"):
            test_module(par, kind)
    print("--- the differential ---")
    for par in ("0000", "00100", "00001"):
        for kind in ("riem", "conf"):
            test_dd(par, kind)
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
