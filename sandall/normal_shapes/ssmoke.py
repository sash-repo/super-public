#!/usr/bin/env python3
"""Controls for sform.py: the conventions, and the two engines it must agree with.

    (A) associativity, supercommutativity, d^2 = 0, Cartan's formula;
    (B) against peri.py  (the periplectic case: odd forms over O(n|n;u), p > 2);
    (C) against nalt2.py (the non-alternating case: p = 2, no odd indeterminates).

Nothing here is new mathematics: it is the reason to believe the rest.
"""
import sys, os, itertools, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sform as S

random.seed(20260821)
LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ssmoke.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def rnd(alg, deg, k=4):
    B = alg.basis(deg)
    out = {}
    for _ in range(k):
        mm = random.choice(B)
        out = alg.add(out, {mm: random.randrange(1, alg.p)})
    return out


# ------------------------------------------------------------------ (A)
say("(A) conventions")
bad = 0
for (m, n, p, u) in [(1, 1, 3, (1,)), (2, 1, 3, (1, 1)), (1, 2, 5, (1,)),
                     (2, 0, 2, (1, 1)), (1, 1, 2, (1,)), (2, 2, 2, (1, 1)),
                     (1, 1, 2, (2,))]:
    A = S.Om(m, n, p, u)
    d = A.d()
    # associativity + supercommutativity on random elements
    for _ in range(25):
        X, Y, Z = (rnd(A, random.randrange(0, 3)) for _ in range(3))
        if A.mul(A.mul(X, Y), Z) != A.mul(X, A.mul(Y, Z)):
            bad += 1
    for mm in A.basis(0) + A.basis(1) + A.basis(2):
        for nn in A.basis(1):
            X, Y = {mm: 1}, {nn: 1}
            s = (-1) ** (A.parity(mm) * A.parity(nn))
            if A.mul(X, Y) != A.smul(s, A.mul(Y, X)):
                bad += 1
    # d^2 = 0
    for deg in (0, 1, 2):
        for mm in A.basis(deg):
            if d(d({mm: 1})):
                bad += 1
    # Cartan: L_D = iota_D d + d iota_D  on Omega^0..Omega^2
    Bv = S.vfield_basis(A)
    for D in random.sample(Bv, min(6, len(Bv))):
        LD, iD = A.lie(D), A.iota(D)
        s = (-1) ** D[2]
        for deg in (0, 1, 2):
            for mm in random.sample(A.basis(deg), min(8, len(A.basis(deg)))):
                X = {mm: 1}
                lhs = LD(X)
                rhs = A.add(iD(d(X)), A.smul(s, d(iD(X))))
                if lhs != rhs:
                    bad += 1
    say(f"    O({m}|{n};{u}) p={p}: sdim O = {A.sdim_funs()}, "
        f"dim Omega^2 = {len(A.basis(2))}")
say("  failures:", bad, " (must be 0)")

# ------------------------------------------------------------------ (B)
say("")
say("(B) against peri.py  (the periplectic case)")
import peri as P                                                   # noqa: E402
bad2 = 0
for (n, p, u) in [(1, 3, (1,)), (1, 3, (2,)), (2, 3, (1, 1)), (1, 5, (1,)),
                  (2, 5, (1, 1)), (1, 7, (1,))]:
    A = S.Om(n, n, p, u)
    Q = P.Omega(n, p, u)
    row = []
    for k in (0, 1, 2):
        a_all = S.cohomology(A, k)[0]
        a_ev = S.cohomology(A, k, 0)[0]
        a_od = S.cohomology(A, k, 1)[0]
        q_all = P.cohomology(Q, k)[0]
        q_ev = P.cohomology(Q, k, 0)[0]
        q_od = P.cohomology(Q, k, 1)[0]
        if (a_all, a_ev, a_od) != (q_all, q_ev, q_od):
            bad2 += 1
        row.append(f"H^{k}=({a_all};{a_ev}|{a_od})")
    # dimensions of the modules themselves
    for k in (0, 1, 2, 3):
        if len(A.basis(k)) != len(Q.basis(k)):
            bad2 += 1
    say(f"    n={n} p={p} u={u}: " + "  ".join(row))
say("  mismatches:", bad2, " (must be 0)")

# the exceptional p=3 class, and omega_0, computed in the new engine
say("")
say("    the p=3 class xi_1(dxi_1)^(2) and omega_0 = sum dq_i dxi_i, new engine:")
for (n, p, u) in [(1, 3, (1,)), (2, 3, (1, 1)), (1, 5, (1,)), (2, 5, (1, 1))]:
    A = S.Om(n, n, p, u)
    cls = A.mul(A.xi(0), A.dxi(0, 2))
    om0 = {}
    for i in range(n):
        om0 = A.add(om0, A.mul(A.dx(i), A.dxi(i)))
    nd, C = S.nondegenerate(A, om0)
    say(f"      n={n} p={p}: omega_0 closed={S.is_closed(A, om0)} "
        f"nondeg={nd} parity={A.parity(next(iter(om0)))}; "
        f"class closed={S.is_closed(A, cls)} exact={S.is_exact(A, cls)}")

# ------------------------------------------------------------------ (C)
say("")
say("(C) against nalt2.py  (the non-alternating case, p = 2, n = 0)")
import nalt2 as N                                                  # noqa: E402
bad3 = 0
for m in (1, 2, 3, 4):
    A = S.Om(m, 0, 2, (1,) * m)
    alg = N.Alg(m)
    dimO = len(A.basis(0))
    dO2 = len(A.basis(2))
    h2, tot, r1, r0 = S.cohomology(A, 2)
    h1 = S.cohomology(A, 1)[0]
    # nalt2's own numbers
    Z = N.closed_space(alg)
    nz = len(Z)
    # dim H^2 = dim Z^2 - dim B^2
    B2 = S.coboundaries(A, 2, 0)[0]
    dB2 = S.rank_p(B2, 2)
    if dO2 != m * (m + 1) // 2 * dimO:
        bad3 += 1
    if nz != dO2 - r1:
        bad3 += 1
    if h2 != nz - dB2:
        bad3 += 1
    say(f"    m={m}: dim O={dimO}, dim Omega^2={dO2} (nalt2: "
        f"{m*(m+1)//2*dimO}), dim Z^2={nz} (new: {dO2-r1}), "
        f"dim B^2={dB2}, dim H^2={h2}, dim H^1={h1}")
say("  mismatches:", bad3, " (must be 0)")

# the standard shape and its preserving algebra, both engines
say("")
say("    the standard p=2 shape (dx_1)^(2) + dx_2dx_3 for m=3:")
A = S.Om(3, 0, 2, (1, 1, 1))
om = A.add(A.dx(0, 2), A.mul(A.dx(1), A.dx(2)))
ev, od, bas = S.preserving(A, om)
nd, C = S.nondegenerate(A, om)
say(f"      closed={S.is_closed(A, om)} nondeg={nd} "
    f"dim h = {ev + od} (nalt2 says 10)")
if ev + od != 10:
    bad3 += 1

say("")
say("TOTAL failures:", bad + bad2 + bad3, " (must be 0)")
LOG.close()
