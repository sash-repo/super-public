"""Calibration of super.py.  Nothing downstream is trusted before this passes."""

import sys, random
from fractions import Fraction as Fr
from itertools import combinations
from super import *

random.seed(5)
FAIL, NCHECK = [], 0


def chk(name, cond):
    global NCHECK
    NCHECK += 1
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        FAIL.append(name)


def rand_fun(n, par=None, maxdeg=3, terms=6):
    """A random function of definite parity par (None = any)."""
    out = {}
    for _ in range(terms):
        a = random.randint(0, maxdeg)
        k = random.randint(0, n)
        if par is not None and k % 2 != par:
            k = k - 1 if k >= 1 else k + 1
            if k < 0 or k > n or (par is not None and k % 2 != par):
                continue
        S = tuple(sorted(random.sample(range(1, n + 1), k)))
        out = add(out, mono(a, S, random.randint(-4, 4)))
    return out


for n in (1, 2, 3, 4):
    print("== n = %d ==" % n)
    ev = [rand_fun(n, 0) for _ in range(3)] + [mono(1), mono(0, (1, 2)) if n >= 2 else mono(0)]
    od = [rand_fun(n, 1) for _ in range(3)] + [mono(0, (1,)), mono(1, (1,))]
    tests = [f for f in ev + od if f]
    hs = [rand_fun(n) for _ in range(3)]

    # [K_f,K_g] = K_{{f,g}} in the super sense
    ok = True
    for f in tests:
        for g in tests:
            pf, pg = parity(f), parity(g)
            for h in hs:
                lhs = add(Kf(n, f, Kf(n, g, h)),
                          scal(-(-1) ** (pf * pg), Kf(n, g, Kf(n, f, h))))
                rhs = Kf(n, kbracket(n, f, g), h)
                if add(lhs, scal(-1, rhs)):
                    ok = False
    chk("[K_f,K_g] = K_{{f,g}}", ok)

    # the frame commutes with k_{-2} + k_{-1}
    gens = [mono(0)] + [mono(0, (i,)) for i in range(1, n + 1)]
    ok = True
    for f in gens:
        pf = parity(f)
        for i in range(1, n + 1):
            for h in hs:
                lhs = add(Kf(n, f, D(i, h)), scal(-(-1) ** pf, D(i, Kf(n, f, h))))
                if lhs:
                    ok = False
        for h in hs:
            if add(Kf(n, f, R(h)), scal(-1, R(Kf(n, f, h)))):
                ok = False
    chk("the frame D_i, R commutes with k_{-2}+k_{-1}", ok)

    # D_i D_j + D_j D_i = -2 delta_ij R   (the Heisenberg relation of the frame)
    ok = True
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for h in hs:
                anti = add(D(i, D(j, h)), D(j, D(i, h)))
                want = scal(-2, R(h)) if i == j else {}
                if add(anti, scal(-1, want)):
                    ok = False
    chk("D_i D_j + D_j D_i = -2 delta_ij R", ok)

    # F_lam is a module
    for lam in (Fr(0), Fr(1), Fr(-1), Fr(2, 3)):
        M = Dens(n, lam)
        ok = True
        for f in tests:
            for g in tests:
                pf, pg = parity(f), parity(g)
                for F in hs:
                    lhs = add(M.act(f, M.act(g, F)),
                              scal(-(-1) ** (pf * pg), M.act(g, M.act(f, F))))
                    if add(lhs, scal(-1, M.act(kbracket(n, f, g), F))):
                        ok = False
        chk("F_%s is a k(1|%d)-module" % (lam, n), ok)

    # the adjoint module is F_{-1}
    M = Dens(n, -1)
    ok = all(not add(M.act(f, g), scal(-1, kbracket(n, f, g)))
             for f in tests for g in tests)
    chk("F_{-1} = the adjoint module", ok)

    # the second weight: is nu a module parameter?  (should be: n = 2 only)
    if n >= 2:
        M = Dens(n, Fr(1, 3), Fr(1, 2))
        ok = True
        for f in tests:
            for g in tests:
                pf, pg = parity(f), parity(g)
                for F in hs:
                    lhs = add(M.act(f, M.act(g, F)),
                              scal(-(-1) ** (pf * pg), M.act(g, M.act(f, F))))
                    if add(lhs, scal(-1, M.act(kbracket(n, f, g), F))):
                        ok = False
        chk("the second weight nu gives a module iff n = 2  (n = %d: %s)"
            % (n, "yes" if ok else "no"), ok == (n == 2))

    # the grading element
    z = mono(1, (), 2)
    ok = True
    for i in range(-2, 4):
        for f in k_piece(n, i):
            if add(kbracket(n, z, f), scal(-i, f)):
                ok = False
    chk("K_{2t} is the grading element (deg t = 2, deg xi = 1)", ok)

    # dimensions of the graded pieces
    dims = [len(k_piece(n, i)) for i in (-2, -1, 0, 1)]
    want = [1, n, 1 + n * (n - 1) // 2, n + n * (n - 1) * (n - 2) // 6]
    chk("dim k_{-2},k_{-1},k_0,k_1 = %s" % (want,), dims == want)

    # div K_f = (2-n)/2 f_t, i.e. Vol = F_{(2-n)/2}
    ok = True
    for f in tests:
        A, B = Kf_coeffs(n, f)
        div = d_t(A)
        pf = parity(f)
        for i in range(1, n + 1):
            div = add(div, scal((-1) ** (pf + 1), d_xi(i, B[i - 1])))
        if add(div, scal(-Fr(2 - n, 2), d_t(f))):
            ok = False
    chk("div K_f = ((2-n)/2) f_t, i.e. Vol = F_{(2-n)/2}", ok)

print("\n%d checks, %d failed" % (NCHECK, len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
