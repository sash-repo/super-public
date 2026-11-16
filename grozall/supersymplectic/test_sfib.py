"""Calibration of the mixed-fiber engine sfib.py.

Nothing here is assumed.  The sign convention of the scheme evaluation -- the
order of the four blocks and the place of the output variable inside an
(in,out) pair -- is SCANNED over all eight possibilities and settled by
operators whose invariance is known before any computation:

    (ptr A) B,   (ptr B) A          -- order 0, S^2 x S^2 --> S^2,
    d(ptr A) ^ d(ptr B)             -- order 2, S^2 x S^2 --> Lambda^2,

each written out by hand from the periplectic trace, the de Rham differential
and the product of forms, and each required (i) to be invariant and (ii) to be
what the corresponding scheme of the ansatz evaluates to, up to a factor.
Then the engine is required to return, in the configuration S^2 x S^2 --> S^2,
the numbers the pooled engine of sgen.py/srun.py returns.

Usage:  python3 test_sfib.py
"""

import itertools
import random
import sys
from fractions import Fraction

import sfib
from sfib import MSetup, enumerate_mixed, evaluate, scheme_parity


def proportional(A, p, q):
    """Is p a non-zero constant multiple of q?"""
    if not p or not q:
        return None
    k0 = next(iter(q))
    if k0 not in p:
        return None
    r = p[k0] / q[k0]
    return r if A.scale(q, r) == p else None


def invariant(S, val_of, pT):
    """val_of(P, Q) invariant of parity pT, on two random fields?"""
    A = S.A
    rng = random.Random(17)
    hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
    P = S.random_field(0, 2, rng, nterms=2)
    Q = S.random_field(1, 2, rng, nterms=2)
    for f in hams:
        pX = (A.parity(f) + (1 if S.form == "odd" else 0)) % 2
        e = -1 if (pT * pX) % 2 else 1
        d = A.add(S.lie(f, val_of(P, Q), 2),
                  A.scale(A.add(val_of(S.lie(f, P, 0), Q),
                                val_of(P, S.lie(f, Q, 1))), -e))
        if d:
            return False
    return True


# ------------------------------------------------- the hand-written operators


def make_T1(S):                        # (ptr A) . B
    return lambda P, Q: S.A.mul(S.ptr(P, 0), S.reencode(Q, (1, 2)))


def make_T2(S):                        # (ptr B) . A
    return lambda P, Q: S.A.mul(S.ptr(Q, 1), S.reencode(P, (0, 2)))


def make_G3(S):                        # d(ptr A) ^ d(ptr B)
    return lambda P, Q: S.A.mul(S.dform(S.ptr(P, 0)), S.dform(S.ptr(Q, 1)))


def find_scheme(schemes, want):
    want = tuple(sorted(tuple(sorted(p)) for p in want))
    for s in schemes:
        if tuple(sorted(tuple(sorted(p)) for p in s.kinds)) == want:
            return s
    return None


GROUND = [
    # (name, shifts, order, maker, the scheme it must be)
    ("(ptr A) B", (0, 0, 0), 0, make_T1,
     (("Au", "Au"), ("Bw", "O"), ("Bw", "O"))),
    ("(ptr B) A", (0, 0, 0), 0, make_T2,
     (("Au", "O"), ("Au", "O"), ("Bw", "Bw"))),
    ("d(ptr A)^d(ptr B)", (0, 0, 1), 2, make_G3,
     (("Au", "Au"), ("Az", "O"), ("Bw", "Bw"), ("Bz", "O"))),
]


def ground_truths(k=2):
    print("=== the hand-written operators, and the schemes that must give them")
    bad = 0
    for (nm, shifts, N, maker, want) in GROUND:
        S = MSetup(k, k, shifts=shifts, form="odd")
        invs = S.invariants(sfib.OUT)
        fun = maker(S)
        pT = ((N + 2) // 2) % 2
        inv = invariant(S, fun, pT)
        sch = find_scheme(enumerate_mixed(N, ninv=len(invs)), want)
        rng = random.Random(99)
        P = S.random_field(0, 2, rng, nterms=2)
        Q = S.random_field(1, 2, rng, nterms=2)
        r = proportional(S.A, evaluate(S, sch, P, Q, invs), fun(P, Q))
        ok = inv and r is not None
        bad += 0 if ok else 1
        print(f"  {nm:20s} {str(shifts)} order {N}: invariant={'yes' if inv else 'NO'}, "
              f"its scheme = {('x' + str(r)) if r is not None else 'NOT PROPORTIONAL'}")
    return bad


def transfer(k=2):
    """Psi is twice the identity re-encoding when the two types agree."""
    print("\n=== reading the value off: Psi against the identity")
    bad = 0
    for form, m, n in [("odd", k, k), ("even", 1, 2)]:
        for s in (0, 1):
            S = MSetup(m, n, shifts=(s, s, s), form=form)
            rng = random.Random(8)
            P = S.random_field(sfib.AFIB, 2, rng, nterms=2)
            r = proportional(S.A, S.Psi(P, (sfib.AFIB, sfib.AFIB)),
                             S.reencode(P, (sfib.AFIB, sfib.OUT)))
            if r is None:
                bad += 1
            print(f"  {form:4s} ({m},{n}) shift {s}: Psi = "
                  + (f"x{r} the identity" if r is not None else "NOT the identity"))
    return bad


def lie_scan(k=2):
    """All 256 sign patterns of the fiber term of the Lie derivative."""
    print("\n=== the sign in the Lie derivative: scanned, not assumed")
    from fractions import Fraction as F

    def lie_eps(S, f, T, fam, bits):
        A = S.A
        X = S.ham(f)
        pX = (A.parity(f) + (1 if S.form == "odd" else 0)) % 2
        terms = [A.mul(X[kk], S.dz(T, kk)) for kk in range(S.nc)]
        c, ei, ek, eX, eik, eiX, ekX, eikX = bits
        for i in range(S.nc):
            for kk in range(S.nc):
                dXk = S.dz(X[kk], i)
                if not dXk:
                    continue
                dT = S.dy(T, fam, kk)
                if not dT:
                    continue
                pi, pk = S.par[i], S.par[kk]
                s = (c + ei * pi + ek * pk + eX * pX + eik * pi * pk
                     + eiX * pi * pX + ekX * pk * pX + eikX * pi * pk * pX) % 2
                t = A.mul(S.y(fam, i), A.mul(dXk, dT))
                terms.append(A.scale(t, -1) if s else t)
        return A.summ(terms)

    for form, m, n in [("odd", k, k), ("even", 1, 2)]:
        for s in (0, 1):
            S = MSetup(m, n, shifts=(s, s, s), form=form)
            A = S.A
            hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
            sel = random.Random(5).sample(hams, min(7, len(hams)))
            rng = random.Random(4)
            Ts = [A.summ([A.mul(S.y(sfib.OUT, i), S.z(i)) for i in range(S.nc)]),
                  S.random_field(sfib.OUT, 2, rng, nterms=2)]
            good = []
            for bits in itertools.product((0, 1), repeat=8):
                ok = True
                for T in Ts:
                    for f in sel:
                        for g in sel:
                            pf = (A.parity(f) + (1 if form == "odd" else 0)) % 2
                            pg = (A.parity(g) + (1 if form == "odd" else 0)) % 2
                            lhs = A.add(
                                lie_eps(S, f, lie_eps(S, g, T, sfib.OUT, bits),
                                        sfib.OUT, bits),
                                A.scale(lie_eps(S, g, lie_eps(S, f, T, sfib.OUT, bits),
                                                sfib.OUT, bits),
                                        -1 if not (pf * pg) % 2 else 1))
                            if lhs != lie_eps(S, S.bracket(f, g), T, sfib.OUT, bits):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    good.append(bits)
            eX = {b[3] for b in good}
            print(f"  {form:4s} ({m},{n}) shift {s}: {len(good)} of 256 patterns "
                  f"are a representation; the p(X) bit is {sorted(eX)} "
                  f"(the engine uses {s})")


def main():
    bad = ground_truths(2)
    bad += transfer(2)
    lie_scan(2)

    print("\n=== the fibers, and their invariant constant tensors")
    for form, m, n in [("odd", 2, 2), ("odd", 3, 3), ("even", 1, 0),
                       ("even", 1, 2), ("even", 2, 2)]:
        for s in (0, 1):
            S = MSetup(m, n, shifts=(s, s, s), form=form)
            ev, od = S.sdim_fiber(sfib.OUT)
            iv = S.invariants(sfib.OUT)
            nm = "S^2(W^*)" if s == 0 else "Lambda^2(W^*)"
            print(f"  {form:4s} ({m},{n}) {nm:14s}: sdim = ({ev}|{od}), "
                  f"{len(iv)} invariant element(s)"
                  + (f", of parity {S.A.parity(iv[0])}" if iv else ""))

    print("\n=== the Lie derivative is a representation, in every family")
    for form, m, n in [("odd", 2, 2), ("even", 1, 2)]:
        for sh in [(0, 0, 0), (1, 1, 1), (0, 1, 1)]:
            S = MSetup(m, n, shifts=sh, form=form)
            A = S.A
            rng = random.Random(3)
            ok = True
            mons = [p for d in (1, 2, 3) for p in S.monomials(d)]
            sel = rng.sample(mons, min(8, len(mons)))
            for fam in range(5):
                T = S.random_field(fam, 2, rng, nterms=2)
                for f in sel:
                    for g in sel:
                        pf = (A.parity(f) + (1 if form == "odd" else 0)) % 2
                        pg = (A.parity(g) + (1 if form == "odd" else 0)) % 2
                        lhs = A.add(S.lie(f, S.lie(g, T, fam), fam),
                                    A.scale(S.lie(g, S.lie(f, T, fam), fam),
                                            -1 if not (pf * pg) % 2 else 1))
                        rhs = S.lie(S.bracket(f, g), T, fam)
                        if lhs != rhs:
                            ok = False
            print(f"  {form:4s} ({m},{n}) shifts {sh}: {ok}")

    print("\n=== calibration against the pooled engine (S^2 x S^2 -> S^2)")
    from srunfib import analyse
    expect = {("odd", 2, 0): 3, ("odd", 2, 2): 3, ("odd", 2, 4): 0,
              ("odd", 3, 0): 3, ("odd", 3, 2): 3,
              ("even", 1, 0): 1, ("even", 1, 2): 1, ("even", 1, 4): 0}
    for (form, m, N), want in sorted(expect.items()):
        n = m if form == "odd" else 0
        S = MSetup(m, n, shifts=(0, 0, 0), form=form)
        _, _, _, dim = analyse(S, N, verbose=False)
        flag = "ok" if dim == want else "MISMATCH"
        if dim != want:
            bad += 1
        print(f"  {form:4s} m={m} N={N}: dim {dim}, expected {want}   {flag}")
    print("\nALL GOOD" if bad == 0 else f"\n{bad} MISMATCHES")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
