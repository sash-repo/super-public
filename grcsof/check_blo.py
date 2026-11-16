#!/usr/bin/env python3
"""Validation of blo.py: (a) F |-> X_F is a homomorphism of Lie superalgebras
in [BLO]'s conventions, (b) the eta's Clifford relations, (c) [BLO]'s own N=1
formulas (3.2)-(3.3) (= [GO]'s supertransvectants) ARE invariant -- which
validates our reading of their conventions, signs and normalizations."""
from fractions import Fraction as Fr
from blo import BLO, check_invariance, J_N1, d_x
from superN import add, smul, mul, mono, parity, par_mon, clean


def commL(alg, f, g, lam, u):
    """[L_f, L_g](u) - L_{{f,g}}(u), super-commutator."""
    pf, pg = parity(f), parity(g)
    Lf, Lg = alg.L(f, lam), alg.L(g, lam)
    s = -1 if (pf and pg) else 1
    lhs = add(Lf(Lg(u)), smul(-s, Lg(Lf(u))))
    rhs = alg.L(alg.kb(f, g), lam)(u)
    return clean(add(lhs, smul(-1, rhs)))


def test_hom(N, lam=Fr(3, 7), tmax=3):
    alg = BLO(N)
    MS = alg.monos(tmax)
    fs = [mono(a, m) for a in range(3) for m in range(1 << N)]
    bad = 0
    for f in fs:
        for g in fs:
            for m in MS:
                if commL(alg, f, g, lam, mono(*m)):
                    bad += 1
    print(f"N={N}: homomorphism [L_f,L_g] = L_(f,g), lam={lam}: "
          f"{'OK' if not bad else str(bad) + ' FAILURES'}")
    return bad


def test_clifford(N, tmax=3):
    alg = BLO(N)
    MS = alg.monos(tmax)
    bad = 0
    for i in range(N):
        for j in range(N):
            ei, ej = alg.eta(i), alg.eta(j)
            for m in MS:
                u = mono(*m)
                lhs = add(ei(ej(u)), ej(ei(u)))
                rhs = smul(-2 if i == j else 0, d_x(u))
                if clean(add(lhs, smul(-1, rhs))):
                    bad += 1
    print(f"N={N}: eta_i eta_j + eta_j eta_i = -2 delta_ij d_x: "
          f"{'OK' if not bad else str(bad) + ' FAILURES'}")
    return bad


def test_osp_closed(N):
    """osp(N|2) = <1, x, x^2, theta_i, x theta_i, theta_i theta_j> is a
    subalgebra: check that all brackets stay in the span."""
    alg = BLO(N)
    gens = alg.osp_gens()
    span = {k for g in gens for k in g}
    bad = [(f, g) for f in gens for g in gens
           if any(k not in span for k in alg.kb(f, g))]
    print(f"N={N}: osp({N}|2) closed under the contact bracket: "
          f"{'OK' if not bad else str(len(bad)) + ' FAILURES'}")
    return len(bad)


def test_J1(ks=(1, 2, 3, Fr(3, 2), Fr(5, 2), Fr(7, 2)),
            tau=Fr(3, 7), lam=Fr(-5, 11), tmax=4):
    """[BLO]'s (3.2)-(3.3) for N=1 must be osp(1|2)-invariant."""
    alg = BLO(1)
    for k in ks:
        T = J_N1(alg, tau, lam, k)
        bad = check_invariance(alg, T, int(2 * k), tau, lam, tmax=tmax)
        print(f"  N=1, k={k} (order {2*k}): "
              f"{'invariant' if not bad else str(len(bad)) + ' failures'}")


if __name__ == '__main__':
    for N in (1, 2, 3):
        test_clifford(N)
        test_hom(N)
        test_osp_closed(N)
    print("\n[BLO]'s N=1 supertransvectants (3.2)-(3.3), generic weights:")
    test_J1()
