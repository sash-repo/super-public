"""Classify the invariant bilinear operators

        T(2,0,...,0) x T(2,0,...,0) --> T(2,0,...,0)

of order N on a symplectic SUPERmanifold of superdimension (2m|n), i.e. on the
super-symmetric covariant 2-tensor fields ("metrics") over h(2m|n).  For n = 0
this is Grozman's configuration and the answer at N = 2 is his operator Gz.

Usage:  python3 srun.py [m] [n] [N] [nsamples] [fielddeg] [nterms]
"""

import os
import pickle
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank, backend            # noqa: E402

from ssuper import SSetup                                 # noqa: E402
from sgen import enumerate_schemes, evaluate              # noqa: E402


def op_parity(S, N):
    """The parity of every scheme of order N.

    Each omega-contraction is even for the even symplectic form (omega^{kl} = 0
    unless p(k) = p(l)) and ODD for the odd one (unless p(k)+p(l) = 1); the
    operator U leaving a derivative index free is even in both cases.  All the
    schemes of one order have the same number (N+2)/2 of contractions, hence one
    and the same parity.
    """
    return (((N + 2) // 2) % 2) if S.form == "odd" else 0


def defect(S, sch, f, Atil, Btil, pT):
    """L_f(T(A,B)) - (-1)^{pT p(X_f)} [ T(L_f A, B) + T(A, L_f B) ].

    Both fields are even, so the only sign is the one an ODD operator picks up
    when the vector field is moved past it.  X_f has parity p(f) for the even
    form and p(f)+1 for the odd one.
    """
    A = S.A
    pX = (A.parity(f) + (1 if S.form == "odd" else 0)) % 2
    e = -1 if (pT * pX) % 2 else 1
    lhs = S.lie(f, evaluate(S, sch, Atil, Btil))
    r1 = evaluate(S, sch, S.lie(f, Atil), Btil)
    r2 = evaluate(S, sch, Atil, S.lie(f, Btil))
    return A.add(lhs, A.scale(A.add(r1, r2), -e))


def operator(S, coeffs, schemes, Atil, Btil):
    A = S.A
    out = []
    for c, sch in zip(coeffs, schemes):
        if c:
            out.append(A.scale(evaluate(S, sch, Atil, Btil), c))
    return A.summ(out)


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    nsamp = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    fdeg = int(sys.argv[5]) if len(sys.argv) > 5 else N + 2
    nterms = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    form = sys.argv[7] if len(sys.argv) > 7 else "even"

    S = SSetup(m, n, form=form)
    schemes = enumerate_schemes(N)
    ev, od = S.sdim_fiber()
    if form == "odd":
        print(f"PERIPLECTIC k = {m}: sdim of the manifold = ({m}|{m}), "
              f"order N = {N}; le({m}), aut(omega) = pe({m})")
        print(f"fiber S^2(W^*): sdim = ({ev}|{od})")
    else:
        print(f"(m,n) = ({m},{n}), sdim of the manifold = ({2*m}|{n}), order N = {N}")
        print(f"fiber S^2(W^*): sdim = ({ev}|{od})   [= sdim osp({n}|{2*m})]")
    print(f"backend = {backend()};  ansatz: {len(schemes)} schemes")

    rng = random.Random(20260904 + 100 * m + 10 * n + N)
    hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
    pT = op_parity(S, N)
    print(f"Hamiltonians tested: {len(hams)} (all monomials of degree 1,2,3); "
          f"every scheme of this order is {'ODD' if pT else 'even'}")

    rows = []
    for s in range(nsamp):
        Atil = S.random_field(fdeg, rng, nterms=nterms)
        Btil = S.random_field(fdeg, rng, nterms=nterms)
        for f in hams:
            ds = [defect(S, sch, f, Atil, Btil, pT) for sch in schemes]
            keys = set()
            for d in ds:
                keys |= set(d)
            for k in sorted(keys):
                rows.append([d.get(k, Fraction(0)) for d in ds])
        print(f"  sample {s+1}: {len(rows)} equations")

    if not rows:
        print("no equations -- every scheme is trivially invariant?")
        ns = [[Fraction(1) if i == j else Fraction(0) for j in range(len(schemes))]
              for i in range(len(schemes))]
    else:
        ns = nullspace(rows)
    print(f"kernel in coefficient space: {len(ns)} of {len(schemes)}")

    # how many of those are non-zero AS OPERATORS
    rng2 = random.Random(4242)
    probes = [(S.random_field(max(3, N + 1), rng2, nterms=3),
               S.random_field(max(3, N + 1), rng2, nterms=3)) for _ in range(2)]
    fps = []
    for v in ns:
        row = []
        for (Atil, Btil) in probes:
            row.append(operator(S, v, schemes, Atil, Btil))
        fps.append(row)
    keys = [sorted(set().union(*[set(fp[i]) for fp in fps])) if fps else []
            for i in range(len(probes))]
    M = []
    for fp in fps:
        r = []
        for i in range(len(probes)):
            for k in keys[i]:
                r.append(fp[i].get(k, Fraction(0)))
        M.append(r)
    dim = rank(M) if M and M[0] else 0
    print(f"DIMENSION of the space of invariant operators of order {N}: {dim}")
    print(f"(zero-operator relations inside the kernel: {len(ns) - dim})")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"sgen_{form}_m{m}_n{n}_N{N}.pkl"), "wb") as fh:
        pickle.dump({"m": m, "n": n, "N": N, "dim": dim, "form": form,
                     "schemes": [(s.FA, s.FB, s.a, s.b, s.alpha, s.beta, s.kinds)
                                 for s in schemes],
                     "kernel": [[str(c) for c in v] for v in ns]}, fh)
    print("saved")


if __name__ == "__main__":
    main()
