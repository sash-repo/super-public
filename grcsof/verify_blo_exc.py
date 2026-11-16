#!/usr/bin/env python3
"""Are the dimension jumps of check_blo3.py real?

The classifier solves a linear system built from test monomials up to degree
tmax; too small a tmax can only INFLATE the answer.  So:

  * where we find dim 1 the answer is certain (dim is upper semicontinuous in
    (tau,lam), hence >= the generic 1, and the computation gives <= 1);
  * where we find dim 2 it must be confirmed -- by recomputing with larger
    tmax and by feeding each basis vector back into the independent
    invariance test check_invariance().

The points tested are the four exceptional pairs we find at k = 3/2 and the
four that [BLO]'s Theorem 4.1(v) predicts instead.
"""
import sys
from fractions import Fraction as Fr
from blo import (BLO, Classifier, check_invariance, terms_of_order, label)
from superN import add, smul, mul, parity


def build(alg, ts, coeffs):
    """The operator T = sum_i c_i (-1)^{|B_i| p(u)} P_i(u) Q_i(v)."""
    Ps = [alg.etaword(A, a) for (a, A, b, B) in ts]
    Qs = [alg.etaword(B, b) for (a, A, b, B) in ts]
    pQ = [len(B) % 2 for (a, A, b, B) in ts]

    def T(u, v):
        pu = parity(u)
        out = {}
        for i, c in enumerate(coeffs):
            if c == 0:
                continue
            sgn = -1 if (pQ[i] and pu) else 1
            out = add(out, smul(c * sgn, mul(Ps[i](u), Qs[i](v))))
        return out
    return T


OURS = [(Fr(-1), Fr(-1)), (Fr(-1), Fr(-1, 2)), (Fr(-1), Fr(0)),
        (Fr(-1, 2), Fr(-1, 2))]
THEIRS = [(Fr(0), Fr(0)), (Fr(0), Fr(-1, 2)), (Fr(0), Fr(-1)),
          (Fr(-1, 2), Fr(-1, 2))]
GENERIC = [(Fr(3, 7), Fr(-5, 11)), (Fr(-1), Fr(3, 7)), (Fr(0), Fr(3, 7))]


def run(ordr=3):
    alg = BLO(3)
    pts = []
    for p in OURS + THEIRS + GENERIC:
        if p not in pts:
            pts.append(p)
    print(f"order {ordr} (k = {Fr(ordr,2)}), osp(3|2), [BLO]'s conventions")
    print(f"{'(tau,lam)':>16} | dim at tmax=3,4,5 | basis re-verified?")
    print("-" * 66)
    for (tau, lam) in pts:
        dims, ok = [], []
        for tmax in (3, 4, 5):
            ts, ns = Classifier(alg, tau, lam).solve(ordr, tmax=tmax)
            dims.append(len(ns))
            if tmax == 5:
                for v in ns:
                    T = build(alg, ts, v)
                    bad = check_invariance(alg, T, ordr, tau, lam, tmax=5)
                    nz = any(T({m: 1}, {m2: 1}) for m in alg.monos(3)
                             for m2 in alg.monos(3))
                    ok.append("inv" if (not bad and nz)
                              else ("ZERO" if not nz else "FAILS"))
        tag = ""
        if (tau, lam) in OURS:
            tag += " [we say 2]"
        if (tau, lam) in THEIRS:
            tag += " [BLO say 2]"
        print(f"{str((str(tau), str(lam))):>16} | {dims} | "
              f"{', '.join(ok)}{tag}")
        sys.stdout.flush()


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
