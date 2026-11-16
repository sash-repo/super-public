#!/usr/bin/env python3
"""An INDEPENDENT check of the operators found by gls.py.

gls.py finds them by solving a linear system; here we take the operator
written out by hand and verify

        P(L^{V1,lam1}_{K_f} phi)  =  (-1)^{p(f)p(P)} L^{V2,lam2}_{K_f}(P phi)

directly, for every generating function f of the algebra and for every
section phi = (monomial) . e_a, using only the module action Lf of kasV.py.
Nothing of the solver is reused.

Usage: python3 glscheck.py
"""
from fractions import Fraction as Fr

from superN import add, smul, mono, parity, par_mon
from kas6 import ALG, NAMES
from kasV import TYPES, Lf, ALGS
from kasbin import testmons
import gls                                     # noqa: F401  (registers k16)

TRIV, L1, SP, SM = TYPES['F'], TYPES['L1'], TYPES['Sp'], TYPES['Sm']
XI, ETA = [0, 2, 4], [1, 3, 5]
D = [ALG.Dword((i,), 0) for i in range(6)]
DT = ALG.Dword((), 1)

# C = B^{-1}: the pairing xi_i <-> eta_i
PARTNER = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}


def op_grad(phi):
    """F --> T(Lambda^1): (grad f)_i = sum_j C_{ij} D_j(f)."""
    return [D[PARTNER[i]](phi[0]) for i in range(6)]


def op_div(phi):
    """T(Lambda^1) --> F: div(phi) = sum_i D_i(phi_i)."""
    s = {}
    for i in range(6):
        s = add(s, D[i](phi[i]))
    return [s]


def op_dirac(V1, V2, coeffs):
    """the operator sum c_{i,a,b} D_i(phi_a) e_b, from a coefficient table."""
    def P(phi):
        out = [{} for _ in range(V2.dim)]
        for (i, a, b), c in coeffs.items():
            if phi[a]:
                out[b] = add(out[b], smul(c, D[i](phi[a])))
        return out
    return P


# the order-1 operator T(S^+) --> T(S^-) found by gls.py (the Dirac operator)
DIRAC = {(0, 1, 1): Fr(1, 2), (0, 2, 2): Fr(1, 2),
         (1, 0, 0): Fr(1), (1, 3, 3): Fr(1),
         (2, 1, 0): Fr(-1, 2), (2, 3, 2): Fr(1, 2),
         (3, 0, 1): Fr(1), (3, 2, 3): Fr(-1),
         (4, 2, 0): Fr(-1, 2), (4, 3, 1): Fr(-1, 2),
         (5, 0, 2): Fr(1), (5, 1, 3): Fr(1)}


def check(name, which, V1, V2, lam1, lam2, P, pP, tmax=2, ymax=6):
    F = ALGS[which]()
    MS = testmons(tmax, ymax)
    bad = []
    for fname, f in sorted(F.items()):
        pf = parity(f)
        s = -1 if (pf and pP) else 1
        for a0 in range(V1.dim):
            for m in MS:
                phi = [{} for _ in range(V1.dim)]
                phi[a0] = mono(*m)
                lhs = P(Lf(V1, f, lam1, phi))
                rhs = [smul(s, x) for x in Lf(V2, f, lam2, P(phi))]
                if any(add(x, smul(-1, y)) for x, y in zip(lhs, rhs)):
                    bad.append((fname, a0, m))
                    break
    print(f"{name}: {which}-invariance of  {V1.name} (lam={lam1}) -> "
          f"{V2.name} (lam={lam2}):  "
          f"{'OK' if not bad else 'FAILS at ' + str(bad[:3])}"
          f"   [{len(F)} generators, {len(MS)} test monomials, "
          f"{V1.dim} fibre directions]", flush=True)
    return not bad


if __name__ == '__main__':
    print("== the operators [GLS] Sec.10 predicts (as singular vectors "
          "of degree 1) ==")
    check("grad ", 'kas', TRIV, L1, Fr(0), Fr(2), op_grad, 1)
    check("Dirac", 'kas', SP, SM, Fr(-5, 2), Fr(-3, 2),
          op_dirac(SP, SM, DIRAC), 1)
    print("== and the ones it does not list ==")
    check("div  ", 'kas', L1, TRIV, Fr(-4), Fr(-4), op_div, 1)
    print("== controls: the same operators at a neighbouring weight "
          "(must FAIL) ==")
    check("grad ", 'kas', TRIV, L1, Fr(1), Fr(3), op_grad, 1)
    check("div  ", 'kas', L1, TRIV, Fr(-3), Fr(-3), op_div, 1)
    check("Dirac", 'kas', SP, SM, Fr(-3, 2), Fr(-1, 2),
          op_dirac(SP, SM, DIRAC), 1)
    print("== and all three under as, where they hold for every weight ==")
    for lam in (Fr(0), Fr(3, 7)):
        check("grad ", 'as', TRIV, L1, lam, lam, op_grad, 1)
        check("div  ", 'as', L1, TRIV, lam, lam, op_div, 1)
        check("Dirac", 'as', SP, SM, lam, lam, op_dirac(SP, SM, DIRAC), 1)
    print("== is the Dirac operator k(1|6)-invariant? ==")
    check("Dirac", 'k16', SP, SM, Fr(-5, 2), Fr(-3, 2),
          op_dirac(SP, SM, DIRAC), 1)
    check("grad ", 'k16', TRIV, L1, Fr(0), Fr(2), op_grad, 1)
    check("div  ", 'k16', L1, TRIV, Fr(-4), Fr(-4), op_div, 1)
