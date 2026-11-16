"""The UNARY invariants in the mixed-fiber setting: what a composition is built
of when the value is allowed a symmetry type of its own.

Statement 11.4 of the text lists them for
a value in the metrics, the 1-forms and the functions.  The value may also be a
2-form, and that is what this adds.

A unary scheme is a perfect matching of

        a derivative slots | 2 fiber slots | 2 output slots,

so a is even; the machinery is that of sfib.py with the second factor absent,
which removes the one sign made by hand there.

Usage:  python3 sunfib.py [k] [form]
"""

import os
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank                        # noqa: E402

import sfib                                                 # noqa: E402
from sfib import MSetup, _matchings                         # noqa: E402

TYPE = {0: "S^2", 1: "Lam^2"}


class UScheme:
    def __init__(self, a, kinds, inv=None):
        self.a, self.kinds, self.inv = a, kinds, inv

    def __repr__(self):
        s = "|".join(x + "-" + y for x, y in self.kinds)
        return f"U(d={self.a},{s}" + (f",inv{self.inv}" if self.inv is not None
                                      else "") + ")"


def enumerate_unary(N, ninv=1):
    if N % 2:
        return []
    out, seen = [], set()
    slots = ["Az"] * N + ["Au"] * 2 + ["O"] * 2
    for mm in _matchings(slots):
        key = tuple(sorted(tuple(sorted(p)) for p in mm))
        if key in seen:
            continue
        seen.add(key)
        noo = sum(1 for p in mm if p[0] == "O" and p[1] == "O")
        if noo:
            for r in range(ninv):
                out.append(UScheme(N, mm, inv=r))
        else:
            out.append(UScheme(N, mm))
    return out


def uparity(S, sch):
    sA, _, sC = S.shifts[:3]
    sh = {"Az": 0, "Au": sA, "O": sC}
    odd = 1 if S.form == "odd" else 0
    p = 0
    for (k1, k2) in sch.kinds:
        if k1 == "O" and k2 == "O":
            p += odd
        elif k1 == "O" or k2 == "O":
            p += sC + sh[k2 if k1 == "O" else k1]
        else:
            p += odd + sh[k1] + sh[k2]
    return p % 2


def evaluate_unary(S, sch, Atil, invs):
    A = S.A
    state = [(Fraction(1), Atil)]
    for (k1, k2) in sch.kinds:
        if k1 == "O" or k2 == "O":
            continue
        new = []
        for (c, P) in state:
            for (i, j, w) in S.Omnz:
                cc, PP = c * w, P
                for (kind, idx) in ((k1, i), (k2, j)):
                    PP = (S.dz(PP, idx) if kind == "Az"
                          else S.dy(PP, sfib.AFIB, idx))
                    if not PP:
                        break
                else:
                    new.append((cc, PP))
        state = new
        if not state:
            return {}
    na = sum(1 for (k1, k2) in sch.kinds if {k1, k2} == {"Az", "O"})
    out = []
    for (c, P) in state:
        for _ in range(na):
            P = S.U(P, sfib.ADER)
        if P:
            out.append(A.scale(P, c))
    V = A.summ(out)
    if not V:
        return {}
    if sch.inv is not None:
        return A.mul(V, invs[sch.inv])
    fams = [sfib.FREESLOT[k2 if k1 == "O" else k1] for (k1, k2) in sch.kinds
            if (k1 == "O") != (k2 == "O")]
    return S.Psi(V, fams)


def analyse_unary(S, N, nsamp=2, verbose=True):
    A = S.A
    invs = S.invariants(sfib.OUT)
    schemes = enumerate_unary(N, ninv=len(invs))
    if not schemes:
        return [], invs, [], 0
    pars = {uparity(S, s) for s in schemes}
    assert len(pars) == 1, f"schemes of mixed parity: {pars}"
    pT = pars.pop()
    rng = random.Random(777 + 10 * S.m + N)
    hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
    rows = []
    for _ in range(nsamp):
        P = S.random_field(sfib.AFIB, N + 2, rng, nterms=3)
        for f in hams:
            pX = (A.parity(f) + (1 if S.form == "odd" else 0)) % 2
            e = -1 if (pT * pX) % 2 else 1
            ds = [A.add(S.lie(f, evaluate_unary(S, s, P, invs), sfib.OUT),
                        A.scale(evaluate_unary(S, s, S.lie(f, P, sfib.AFIB), invs),
                                -e))
                  for s in schemes]
            keys = set()
            for d in ds:
                keys |= set(d)
            for kk in sorted(keys):
                rows.append([d.get(kk, Fraction(0)) for d in ds])
    ns = (nullspace(rows) if rows else
          [[Fraction(1) if i == j else Fraction(0) for j in range(len(schemes))]
           for i in range(len(schemes))])
    rng2 = random.Random(31)
    probes = [S.random_field(sfib.AFIB, max(3, N + 1), rng2, nterms=3)
              for _ in range(2)]
    fps = [[A.summ([A.scale(evaluate_unary(S, s, X, invs), c)
                    for c, s in zip(v, schemes) if c]) for X in probes]
           for v in ns]
    keys = [sorted(set().union(*[set(fp[i]) for fp in fps])) if fps else []
            for i in range(len(probes))]
    M = [[fp[i].get(k, Fraction(0)) for i in range(len(probes)) for k in keys[i]]
         for fp in fps]
    dim = rank(M) if M and M[0] else 0
    if verbose:
        sA, _, sC = S.shifts[:3]
        print(f"  {TYPE[sA]:5s} --> {TYPE[sC]:5s}, order {N}: {len(schemes)} schemes, "
              f"kernel {len(ns)}, DIMENSION {dim}  "
              f"(parity {'odd' if pT else 'even'})")
    return schemes, invs, ns, dim


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    form = sys.argv[2] if len(sys.argv) > 2 else "odd"
    n = k if form == "odd" else 0
    print(f"UNARY invariants, {form} form, k = {k}")
    for sA in (0, 1):
        for sC in (0, 1):
            S = MSetup(k, n, shifts=(sA, sA, sC), form=form)
            for N in (0, 2):
                analyse_unary(S, N)


if __name__ == "__main__":
    main()
