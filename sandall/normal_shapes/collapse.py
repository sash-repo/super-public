#!/usr/bin/env python3
"""The collapse at p = 2, verified literally.

Claim.  For p = 2 the assignment

    x_i |--> x_i,   xi_j |--> x_{m+j},   dx_i |--> dx_i,   dxi_j |--> dx_{m+j}

is an isomorphism of differential algebras

    Omega^*(m|n; u)  -->  Omega^*(m+n; (u, 1, ..., 1))

which preserves the form degree and forgets exactly one thing: the Z/2-grading.
So at p = 2 a superspace and a space have the same de Rham complex, and the
"union" of the two cases is literal: the super case is the non-alternating case plus a grading.

The reason it is true: at p = 2 an odd indeterminate (xi^2 = 0, no divided
powers) IS an even indeterminate of height 2, and the differential of an even
indeterminate is odd but carries divided powers all the same, since
(dx)^2 = 2(dx)^{(2)} = 0 holds identically.  Both facts fail for p > 2.

Checked here: the multiplication table, the differential, the form degree and
the parity, on a full basis of each side.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sform as S

LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "collapse.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def transport(A, B, mon):
    """A = Om(m,n), B = Om(m+n,0): move a monomial across."""
    e = [0] * B.N
    for i in range(A.m):
        e[i] = mon[A.ix + i]
        e[B.idx + i] = mon[A.idx + i]
    for j in range(A.n):
        e[A.m + j] = mon[A.ixi + j]
        e[B.idx + A.m + j] = mon[A.idxi + j]
    return tuple(e)


def move(A, B, X):
    return {transport(A, B, mm): c for mm, c in X.items()}


def check(m, n, u, degs=(0, 1, 2, 3)):
    M = m + n
    U = tuple(u) + (1,) * n
    A = S.Om(m, n, 2, u, dh=2, dxh=2)
    B = S.Om(M, 0, 2, U, dh=2, dxh=2)
    bad = 0
    bases = {k: A.basis(k) for k in degs}
    # 1) the transport is a bijection of bases, degree by degree
    for k in degs:
        left = sorted(transport(A, B, mm) for mm in bases[k])
        right = sorted(B.basis(k))
        if left != right:
            bad += 1
    # 2) multiplication
    dA, dB = A.d(), B.d()
    for k in degs[:3]:
        for l in degs[:3]:
            if k + l > max(degs):
                continue
            for a in bases[k]:
                for b in bases[l]:
                    lhs = move(A, B, A.mul({a: 1}, {b: 1}))
                    rhs = B.mul({transport(A, B, a): 1}, {transport(A, B, b): 1})
                    if lhs != rhs:
                        bad += 1
    # 3) the differential and the form degree
    for k in degs:
        for a in bases[k]:
            if move(A, B, dA({a: 1})) != dB({transport(A, B, a): 1}):
                bad += 1
            if A.degree(a) != B.degree(transport(A, B, a)):
                bad += 1
    # 4) the parity is NOT transported -- report how far off it is
    diff = sum(1 for k in degs for a in bases[k]
               if A.parity(a) != B.parity(transport(A, B, a)))
    tot = sum(len(bases[k]) for k in degs)
    say(f"    O({m}|{n};{u}) -> O({M};{U}): "
        f"{tot} basis monomials in degrees {degs}, failures {bad}; "
        f"parity differs on {diff} of them")
    return bad


say("The p = 2 collapse: Omega^*(m|n;u) = Omega^*(m+n;(u,1^n)) as differential algebras")
bad = 0
for (m, n, u) in [(1, 1, (1,)), (2, 1, (1, 1)), (1, 2, (1,)), (2, 2, (1, 1)),
                  (0, 2, ()), (3, 0, (1, 1, 1)), (0, 3, ()), (1, 1, (2,)),
                  (2, 1, (2, 1)), (1, 3, (1,))]:
    bad += check(m, n, u)
say("")
say("failures:", bad, " (must be 0)")

say("")
say("For p > 2 the same transport is NOT an isomorphism, and the two reasons")
say("show up separately.  With p = 3, m = n = 1:")
A = S.Om(1, 1, 3, (1,), dh=2, dxh=2)
say(f"    dim Omega^k(1|1) for k = 0..3: {[len(A.basis(k)) for k in range(4)]}")
B = S.Om(2, 0, 3, (1, 1), dh=2, dxh=2)
say(f"    dim Omega^k(2)   for k = 0..3: {[len(B.basis(k)) for k in range(4)]}")
say("    xi has no divided powers and dxi does, dx has none:")
say(f"      (dxi)^(2) is a basis monomial of Omega^2(1|1): "
    f"{tuple(A.dxi(0, 2))[0] in A.basis(2) if A.dxi(0,2) else False}")
say(f"      (dx)^(2) in Omega^2(2): {list(B.dx(0, 2))[0] in B.basis(2)}")
LOG.close()
