"""The periplectic Grozman operators, named -- and the two things the naming needs.

Three questions have to be answered before the six operators over a periplectic
form (three even of order 2, three odd of order 0) can be named, and this script
answers all three in exact arithmetic.

  1) SYMMETRY.  Which of the six operators is symmetric under A <-> A'?  The even
     ones (order 2) are known to split 1 + 2; the odd ones (order 0) were never
     tested.  A bare name should go to a distinguished member, and "the symmetric
     one" is what distinguishes hGz over an even form (it is commutative).

  2) THE SIGN.  The naive transcription of the third odd operator is
         sym_{ij} sum_{a,b} omega^{ab} A_{ia} A'_{jb}
     with no Koszul sign.  The claim to be established is that the sign the
     operator needs is (-1)^{p(b) p(A)} -- one factor, depending on the parity of
     the contracted index of the SECOND factor and on the parity of the FIRST
     field -- and hence that it is +1 on metrics, which are even, so that the
     sign-free expression is right on metrics and incomplete only off them.

  3) THE TRACE.  The periplectic trace of each of the
     three EVEN operators is the antibracket {pi, pi'} of the two traces.  What is
     it for the three ODD ones?  If it is the product pi pi', the two orders answer
     the same way with the two operations the antibracket algebra has.

Usage:  python3 soddnames.py [k ...]        (default 2 3)
"""

import os
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace                             # noqa: E402

from ssuper import SSetup                                  # noqa: E402
from sodd0 import ptr                                      # noqa: E402
from sodd00 import T3, defect_named                        # noqa: E402
from sodd00 import ptr as ptr_unhalved                     # noqa: E402
from srun import op_parity                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# NB.  sodd00.py carries its own periplectic trace, which is TWICE the trace
# ptr A = sum omega^{ij} A_{ij} of sodd0.ptr, the one every other script here
# uses.  For sodd00's own purposes (spans, independence, invariance) the
# normalization cancels; for a constant in front of an answer it does not.  So T1
# and T2 are rebuilt below against sodd0.ptr, and the two traces are compared.

def T1(S, P, Q):
    """oGz_1 = (ptr A) A', with ptr A = sum omega^{ij} A_{ij}."""
    return S.A.mul(ptr(S, P), Q)


def T2(S, P, Q):
    """oGz_2 = (ptr A') A."""
    return S.A.mul(ptr(S, Q), P)


# ---------------------------------------------------------------- the sign rule

def T3_nosign(S, P, Q):
    """The naive transcription -- no Koszul sign at all."""
    A = S.A
    out = []
    for k, l, w in S.Omnz:
        PP = S.du(P, k)
        if not PP:
            continue
        QQ = S.du(Q, l)
        if not QQ:
            continue
        out.append(A.scale(A.mul(PP, QQ), w))
    return A.summ(out)


def T3_rule(S, P, Q):
    """The closed rule:  sign = (-1)^{p(l) p(P)}.

    Written with the parity of the FIELD, not of the differentiated factor.  It
    agrees with T3 of sodd00.py exactly when omega^{kl} = 0 unless p(k)+p(l) = 1,
    which is what "odd form" means -- so this is a theorem about periplectic
    forms, not a rewriting, and the script checks it as such.
    """
    A = S.A
    pP = A.parity(P)
    out = []
    for k, l, w in S.Omnz:
        PP = S.du(P, k)
        if not PP:
            continue
        QQ = S.du(Q, l)
        if not QQ:
            continue
        if S.upar[l] and pP:
            w = -w
        out.append(A.scale(A.mul(PP, QQ), w))
    return A.summ(out)


def same(S, F, G, fields):
    """Do two operators agree on every ordered pair from `fields`?"""
    A = S.A
    for P in fields:
        for Q in fields:
            if A.add(F(S, P, Q), A.scale(G(S, P, Q), -1)):
                return False
    return True


# ------------------------------------------------------------------- symmetry

def symmetry(S, F, fields, pT):
    """+1 / -1 / None:  is F(A,A') = eps * F(A',A) on the given fields?

    The fields are even, so no Koszul sign enters the interchange itself.
    """
    A = S.A
    for eps in (1, -1):
        ok = True
        for i in range(len(fields)):
            for j in range(len(fields)):
                d = A.add(F(S, fields[i], fields[j]),
                          A.scale(F(S, fields[j], fields[i]), -eps))
                if d:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return eps
    return None


# ---------------------------------------------------------------------- run

def run(k):
    S = SSetup(k, k, form="odd")
    A = S.A
    pT = op_parity(S, 0)
    print(f"\n=== periplectic k = {k};  sdim M = ({k}|{k});  "
          f"the order 0 is {'ODD' if pT else 'even'}")

    rng = random.Random(9001 + 13 * k)
    hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
    even_fields = [S.random_field(3, rng, nterms=3) for _ in range(4)]
    even_fields = [T for T in even_fields if T]
    print(f"  {len(even_fields)} even metric fields, {len(hams)} Hamiltonians")

    # ---- 1) the sign rule -------------------------------------------------
    print("\n  1) THE SIGN")
    ok_rule = same(S, T3, T3_rule, even_fields)
    print(f"     the closed rule (-1)^{{p(b) p(A)}} reproduces the operator "
          f"on even fields: {'YES' if ok_rule else 'NO'}")

    # on ODD fields, where the sign is supposed to become visible
    odd_fields = []
    for f in hams[:12]:
        for P in even_fields[:2]:
            L = S.lie(f, P)
            if L and A.parity(L) == 1:
                odd_fields.append(L)
    odd_fields = odd_fields[:4]
    print(f"     {len(odd_fields)} odd fields built as L_X A")
    if odd_fields:
        allf = even_fields + odd_fields
        print(f"     rule = scheme evaluation on all {len(allf)} fields "
              f"(both parities): {'YES' if same(S, T3, T3_rule, allf) else 'NO'}")
        print(f"     sign-free formula = the operator on even fields only: "
              f"{'YES' if same(S, T3_nosign, T3, even_fields) else 'NO'}")
        print(f"     sign-free formula = the operator on all fields:       "
              f"{'YES' if same(S, T3_nosign, T3, allf) else 'NO'}")

    # and the invariance, which is what the sign is for
    for name, F in (("with the sign", T3), ("sign-free", T3_nosign)):
        bad = 0
        for P in even_fields[:2]:
            for Q in even_fields[:2]:
                for f in hams:
                    if defect_named(S, F, f, P, Q, pT):
                        bad += 1
        tot = len(hams) * min(2, len(even_fields)) ** 2
        print(f"     invariant, {name:30s}: "
              f"{'YES' if not bad else f'NO ({bad}/{tot})'}")

    # ---- 2) symmetry ------------------------------------------------------
    print("\n  2) SYMMETRY under A <-> A'  (on even metrics)")
    named = [("oGz_1 = (ptr A) A'", T1), ("oGz_2 = (ptr A') A", T2),
             ("oGz_3 = om^{ab} A_ia A'_jb", T3)]
    for nm, F in named:
        e = symmetry(S, F, even_fields, pT)
        tag = {1: "SYMMETRIC", -1: "ANTIsymmetric", None: "neither"}[e]
        print(f"     {nm:30s} {tag}")
    # the two trace-built ones swap, so their sum and difference are pure
    for nm, F in (("oGz_1 + oGz_2",
                   lambda S, P, Q: S.A.add(T1(S, P, Q), T2(S, P, Q))),
                  ("oGz_1 - oGz_2",
                   lambda S, P, Q: S.A.add(T1(S, P, Q),
                                           S.A.scale(T2(S, P, Q), -1)))):
        e = symmetry(S, F, even_fields, pT)
        tag = {1: "SYMMETRIC", -1: "ANTIsymmetric", None: "neither"}[e]
        print(f"     {nm:30s} {tag}")

    # ---- 3) the trace of the odd ones -------------------------------------
    print("\n  3) THE TRACE of the three odd operators")
    fac = [c for c in (1, 2, Fraction(1, 2))
           if all(not A.add(ptr_unhalved(S, T), A.scale(ptr(S, T), -c))
                  for T in even_fields)]
    print(f"     control: sodd00's trace = {fac} x sodd0's trace")
    for nm, F in named:
        rows = []
        for P in even_fields:
            for Q in even_fields:
                t = ptr(S, F(S, P, Q))
                prod = A.mul(ptr(S, P), ptr(S, Q))
                brk = S.bracket(ptr(S, P), ptr(S, Q))
                keys = sorted(set(t) | set(prod) | set(brk))
                for kk in keys:
                    rows.append([t.get(kk, Fraction(0)),
                                 prod.get(kk, Fraction(0)),
                                 brk.get(kk, Fraction(0))])
        ns = nullspace(rows) if rows else []
        # look for  t = c * prod  and  t = c * brk
        got = []
        for idx, lab in ((1, "pi pi'"), (2, "{pi,pi'}")):
            sol = [v for v in ns if v[0] and not any(
                v[j] for j in (1, 2) if j != idx)]
            if sol:
                v = sol[0]
                got.append(f"= ({-v[idx] / v[0]}) {lab}")
        allzero = all(not ptr(S, F(S, P, Q))
                      for P in even_fields for Q in even_fields)
        print(f"     ptr {nm:30s} "
              + ("identically 0" if allzero else
                 (", ".join(got) if got else
                  f"neither (nullspace dim {len(ns)})")))

    # is pi pi' itself zero?  (pi is odd, so pi^2 = 0, but pi pi' need not be)
    nz = any(A.mul(ptr(S, P), ptr(S, Q))
             for P in even_fields for Q in even_fields)
    print(f"     control: pi pi' is non-zero somewhere: {'YES' if nz else 'NO'}")


def main():
    ks = [int(x) for x in sys.argv[1:]] or [2, 3]
    for k in ks:
        run(k)


if __name__ == "__main__":
    main()
