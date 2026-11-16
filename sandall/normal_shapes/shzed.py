#!/usr/bin/env python3
"""The one statement that holds in all three cases of the union:

    omega non-degenerate  ==>  D |--> iota_D omega  is an isomorphism of
    O-modules vect(m|n;u) --> Omega^1 of parity p(omega)+1, and it carries
    h(omega) = {D | L_D omega = 0} onto the CLOSED 1-forms.  Hence

        h(omega) = Pi^{p(omega)+1} Z^1,     h(omega)/{H_f} = Pi^{p(omega)+1} H^1,

    and the Hamiltonian fields {H_f} are Pi^{p(omega)} (O/K).

Verified here for every case the text discusses, every p, both parities.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sform as S

LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "shzed.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def Z1(A, par):
    """dim of the closed 1-forms of the given parity."""
    d = A.d()
    B1 = [mm for mm in A.basis(1) if A.parity(mm) == par]
    B2 = [mm for mm in A.basis(2) if A.parity(mm) == (par + 1) % 2]
    i2 = {mm: i for i, mm in enumerate(B2)}
    M = S.matrix_of(A, d, B1, i2)
    return len(B1) - S.rank_p(M, A.p)


def omega_even(A):
    k = A.m // 2
    om = {}
    for i in range(k):
        om = A.add(om, A.mul(A.dx(i), A.dx(k + i)))
    for j in range(A.n):
        om = A.add(om, A.dxi(j, 2))
    return om


def omega_odd(A):
    om = {}
    for i in range(A.m):
        om = A.add(om, A.mul(A.dx(i), A.dxi(i)))
    return om


def omega_I(A):
    """p=2: omega_I = sum_a (dg_a)^{(2)}, the analog of Lebedev's One_M.

    Even, closed, non-degenerate for EVERY m|n -- the Gram matrix is the
    identity.  This is the shape that has no analog for p > 2.
    """
    om = {}
    for i in range(A.m):
        om = A.add(om, A.dx(i, 2))
    for j in range(A.n):
        om = A.add(om, A.dxi(j, 2))
    return om


def omega_Pi(A):
    """p=2: the alternating shape, pairing LIKE parities -- whence even.

    It is non-degenerate only for m and n both even; pairing x with xi gives
    the ODD shape omega_odd instead, and a mixed pairing gives a form that is
    not homogeneous at all.
    """
    om = {}
    for i in range(0, A.m - 1, 2):
        om = A.add(om, A.mul(A.dx(i), A.dx(i + 1)))
    for j in range(0, A.n - 1, 2):
        om = A.add(om, A.mul(A.dxi(j), A.dxi(j + 1)))
    return om


say("h(omega) = Pi^{p(omega)+1} Z^1  --  checked case by case")
say("  columns: sdim O | sdim h (machine) | Pi^{p+1} sdim Z^1 (predicted) | "
    "dim H^1 | sdim {H_f}")
bad = 0
CASES = []
for (m, n, p, u) in [(2, 1, 3, (1, 1)), (2, 2, 3, (1, 1)), (2, 1, 5, (1, 1)),
                     (2, 0, 3, (1, 1)), (2, 3, 3, (1, 1)), (4, 1, 3, (1, 1, 1, 1))]:
    CASES.append((m, n, p, u, "even"))
for (n, p, u) in [(1, 3, (1,)), (2, 3, (1, 1)), (1, 5, (1,)), (1, 3, (2,)),
                  (3, 3, (1, 1, 1))]:
    CASES.append((n, n, p, u, "odd"))
for (m, n, u) in [(2, 0, (1, 1)), (1, 1, (1,)), (2, 1, (1, 1)), (0, 2, ()),
                  (1, 2, (1,)), (3, 0, (1, 1, 1)), (0, 3, ()), (2, 2, (1, 1)),
                  (1, 1, (2,)), (0, 1, ()), (1, 0, (1,))]:
    CASES.append((m, n, 2, u, "I"))
for (m, n, u) in [(2, 0, (1, 1)), (1, 1, (1,)), (0, 2, ()), (2, 2, (1, 1)),
                  (1, 1, (2,)), (3, 1, (1, 1, 1))]:
    CASES.append((m, n, 2, u, "Pi"))
for (n, u) in [(1, (1,)), (2, (1, 1)), (1, (2,))]:
    CASES.append((n, n, 2, u, "odd"))

for (m, n, p, u, kind) in CASES:
    A = S.Om(m, n, p, u, dh=2, dxh=2)
    om = {"even": omega_even, "odd": omega_odd,
          "I": omega_I, "Pi": omega_Pi}[kind](A)
    nd, C = S.nondegenerate(A, om)
    if not nd or not S.is_closed(A, om):
        say(f"    O({m}|{n};{u}) p={p} {kind:4s}: SKIPPED "
            f"(closed={S.is_closed(A, om)}, nondeg={nd})")
        continue
    pars = set(A.parity(mm) for mm in om)
    assert len(pars) == 1, "inhomogeneous standard shape"
    pw = pars.pop()
    ev, od, _ = S.preserving(A, om)
    z0, z1 = Z1(A, 0), Z1(A, 1)
    # h of parity r  <->  Z^1 of parity r + p(omega) + 1
    pred = (z1 if (pw + 1) % 2 == 1 else z0,
            z0 if (pw + 1) % 2 == 1 else z1)
    h1 = S.cohomology(A, 1)[0]
    sdO = A.sdim_funs()
    hf = (sdO[0] - 1, sdO[1]) if pw == 0 else (sdO[1], sdO[0] - 1)
    ok = (ev, od) == pred and (ev + od) - (hf[0] + hf[1]) == h1
    if not ok:
        bad += 1
    say(f"    O({m}|{n};{u}) p={p} {kind:4s} omega {'even' if pw==0 else 'odd '}: "
        f"({sdO[0]}|{sdO[1]}) | ({ev}|{od}) | ({pred[0]}|{pred[1]}) | "
        f"{h1} | ({hf[0]}|{hf[1]})   {'ok' if ok else 'MISMATCH'}")

say("")
say("failures:", bad, " (must be 0)")
LOG.close()
