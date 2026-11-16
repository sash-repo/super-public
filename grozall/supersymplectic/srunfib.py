"""The dimension of the space of invariant bilinear operators of order N in a
configuration of three 2-tensor fibers of prescribed symmetry types.

Usage:  python3 srunfib.py <form> <m> <n> <sA><sB><sC> <N> [nsamp] [fdeg] [nterms]

    form  = even | odd            (odd = periplectic, then m = n = k)
    sX    = 0 for S^2(W^*), 1 for Lambda^2(W^*)

e.g.  python3 srunfib.py odd 2 2 001 2      -- S^2 x S^2 -> Lambda^2, k = 2.
"""

import os
import pickle
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank, backend             # noqa: E402

import sfib                                                        # noqa: E402
from sfib import MSetup, enumerate_mixed, evaluate, scheme_parity   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TYPE = {0: "S^2", 1: "Lam^2"}


def defect(S, sch, invs, f, Atil, Btil, pT):
    """L_f(T(A,B)) - (-1)^{pT p(X_f)} [ T(L_f A, B) + T(A, L_f B) ]."""
    A = S.A
    pX = (A.parity(f) + (1 if S.form == "odd" else 0)) % 2
    e = -1 if (pT * pX) % 2 else 1
    lhs = S.lie(f, evaluate(S, sch, Atil, Btil, invs), sfib.OUT)
    r1 = evaluate(S, sch, S.lie(f, Atil, sfib.AFIB), Btil, invs)
    r2 = evaluate(S, sch, Atil, S.lie(f, Btil, sfib.BFIB), invs)
    return A.add(lhs, A.scale(A.add(r1, r2), -e))


def operator(S, coeffs, schemes, invs, Atil, Btil):
    A = S.A
    return A.summ([A.scale(evaluate(S, sch, Atil, Btil, invs), c)
                   for c, sch in zip(coeffs, schemes) if c])


def analyse(S, N, nsamp=2, fdeg=None, nterms=3, verbose=True, seed=None):
    """Returns (schemes, invs, kernel, dim)."""
    A = S.A
    invs = S.invariants(sfib.OUT)
    schemes = enumerate_mixed(N, ninv=len(invs))
    if not schemes:
        if verbose:
            print("  no scheme at all")
        return [], invs, [], 0
    pars = {scheme_parity(S, s) for s in schemes}
    assert len(pars) == 1, f"schemes of mixed parity: {pars}"
    pT = pars.pop()
    if fdeg is None:
        fdeg = N + 2
    if seed is None:
        seed = 20260908 + 1000 * S.m + 100 * S.n + 10 * N
    rng = random.Random(seed)
    hams = [p for d in (1, 2, 3) for p in S.monomials(d)]

    if verbose:
        sA, sB, sC = S.shifts[:3]
        print(f"  fibers {TYPE[sA]} x {TYPE[sB]} -> {TYPE[sC]}, order {N}: "
              f"{len(schemes)} schemes ({len(invs)} invariant element(s) in the "
              f"target), operator parity {'ODD' if pT else 'even'}")

    rows = []
    measured = set()
    for _ in range(nsamp):
        Atil = S.random_field(sfib.AFIB, fdeg, rng, nterms=nterms)
        Btil = S.random_field(sfib.BFIB, fdeg, rng, nterms=nterms)
        for f in hams:
            ds = []
            for sch in schemes:
                val = evaluate(S, sch, Atil, Btil, invs)
                if val:
                    measured.add(A.parity(val))
                ds.append(defect(S, sch, invs, f, Atil, Btil, pT))
            keys = set()
            for d in ds:
                keys |= set(d)
            for kk in sorted(keys):
                rows.append([d.get(kk, Fraction(0)) for d in ds])
    if measured:
        assert measured == {pT}, f"parity measured {measured}, predicted {pT}"

    ns = (nullspace(rows) if rows else
          [[Fraction(1) if i == j else Fraction(0) for j in range(len(schemes))]
           for i in range(len(schemes))])

    # how many of those are non-zero AS OPERATORS
    rng2 = random.Random(4242)
    probes = [(S.random_field(sfib.AFIB, max(3, N + 1), rng2, nterms=3),
               S.random_field(sfib.BFIB, max(3, N + 1), rng2, nterms=3))
              for _ in range(2)]
    fps = [[operator(S, v, schemes, invs, P, Q) for (P, Q) in probes] for v in ns]
    keys = [sorted(set().union(*[set(fp[i]) for fp in fps])) if fps else []
            for i in range(len(probes))]
    M = [[fp[i].get(k, Fraction(0)) for i in range(len(probes)) for k in keys[i]]
         for fp in fps]
    dim = rank(M) if M and M[0] else 0
    if verbose:
        print(f"      kernel {len(ns)}/{len(schemes)}, "
              f"zero-operator relations {len(ns) - dim},  DIMENSION {dim}")
    return schemes, invs, ns, dim


def main():
    form = sys.argv[1] if len(sys.argv) > 1 else "odd"
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    sh = sys.argv[4] if len(sys.argv) > 4 else "000"
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    nsamp = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    fdeg = int(sys.argv[7]) if len(sys.argv) > 7 else None
    nterms = int(sys.argv[8]) if len(sys.argv) > 8 else 3

    shifts = tuple(int(c) for c in sh)
    S = MSetup(m, n, shifts=shifts, form=form)
    head = (f"PERIPLECTIC k = {m}, sdim ({m}|{m}), le({m})" if form == "odd"
            else f"SYMPLECTIC sdim ({2*m}|{n}), h({2*m}|{n})")
    print(f"{head};  backend = {backend()}")
    for f in (sfib.AFIB, sfib.BFIB, sfib.OUT):
        ev, od = S.sdim_fiber(f)
        print(f"  family {f} ({TYPE[S.shifts[f]]}): sdim = ({ev}|{od})")
    schemes, invs, ns, dim = analyse(S, N, nsamp=nsamp, fdeg=fdeg, nterms=nterms)


if __name__ == "__main__":
    main()
