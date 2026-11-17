#!/usr/bin/env python3
"""
The queer superalgebra Q(n) and the anti-automorphisms of Subsec. 1.4 of
the text: center vs supercenter, the order of -q, the anti-symmetric
elements, and Montgomery's condition (Cond).

Exact arithmetic over Q(i) (sympy matrices with the symbol I).
"""
import sympy as sp

I = sp.I


def Qn_basis(n):
    """Basis of Q(n) = {(A B / B A)} inside Mat(n|n), even part first."""
    ev, od = [], []
    for i in range(n):
        for j in range(n):
            E = sp.zeros(2 * n, 2 * n)
            E[i, j] = 1
            E[n + i, n + j] = 1                      # (A 0 / 0 A)
            ev.append(E)
            F = sp.zeros(2 * n, 2 * n)
            F[i, n + j] = 1
            F[n + i, j] = 1                          # (0 B / B 0)
            od.append(F)
    return ev, od


def parity_of(M, n):
    return 0 if all(M[i, j] == 0 for i in range(n) for j in range(n, 2 * n)) else 1


def center(n, super_=False):
    """Dimension of the (super)center of Q(n)."""
    ev, od = Qn_basis(n)
    basis = ev + od
    pars = [0] * len(ev) + [1] * len(od)
    coeffs = sp.symbols('c0:%d' % len(basis))
    X = sum((c * B for c, B in zip(coeffs, basis)), sp.zeros(2 * n, 2 * n))
    eqs = []
    for B, pb in zip(basis, pars):
        for pc in (0, 1):
            # split X into its even and odd halves for the supercommutator
            Xp = sum((c * Bb for c, Bb, pp in zip(coeffs, basis, pars) if pp == pc),
                     sp.zeros(2 * n, 2 * n))
            sign = -1 if (super_ and pc == 1 and pb == 1) else 1
            C = Xp * B - sign * B * Xp
            eqs.extend(list(C))
    sol = sp.linsolve(eqs, coeffs)
    free = len(coeffs) - len(sp.Matrix([[sp.diff(e, c) for c in coeffs]
                                        for e in eqs]).rref()[1])
    return free


def mq_map(M, n):
    """The map named -q in eq. (5):  (A B / B A) -> (A^t  -i B^t / -i B^t  A^t).
    NB: the display defines -q, so this IS -q, and q = -(this)."""
    A = M[:n, :n]
    B = M[:n, n:]
    out = sp.zeros(2 * n, 2 * n)
    At, Bt = A.T, B.T
    out[:n, :n] = At
    out[:n, n:] = -I * Bt
    out[n:, :n] = -I * Bt
    out[n:, n:] = At
    return out                                        # = (-q)(M)


def main():
    print("Q(n): center vs supercenter, and the anti-automorphism -q\n")
    for n in (2, 3, 4):
        ev, od = Qn_basis(n)
        dim = len(ev) + len(od)
        zc = center(n, super_=False)
        zs = center(n, super_=True)
        print("  n=%d:  sdim Q(n) = (%d|%d), dim Q(n) = %d" % (n, len(ev), len(od), dim))
        print("        dim(ordinary center) = %d, dim(supercenter) = %d" % (zc, zs))
        print("        dim_Z R = %d (ordinary)   %d (super)"
              % (dim // zc, dim // zs))

    print("\n  the hypothesis dim_Z R > 16 of Theorem GS:")
    for n in range(2, 7):
        print("    n=%d: ordinary center -> dim_Z R = %-4d %s ;  supercenter -> %-4d %s"
              % (n, n * n, "PASS" if n * n > 16 else "fail",
                 2 * n * n, "PASS" if 2 * n * n > 16 else "fail"))

    print("\n  the order of q (eq. (5)):")
    for n in (2, 3):
        ev, od = Qn_basis(n)
        ok2 = ok4 = True
        delta = True
        for B in ev + od:
            q2 = mq_map(mq_map(B, n), n)      # (-q)^2 = q^2
            q4 = mq_map(mq_map(q2, n), n)
            p = parity_of(B, n)
            if sp.simplify(sp.Matrix(q2 - B)) != sp.zeros(2 * n, 2 * n):
                ok2 = False
            if sp.simplify(sp.Matrix(q4 - B)) != sp.zeros(2 * n, 2 * n):
                ok4 = False
            # delta_{-1}: X -> (-1)^{p(X)} X
            if sp.simplify(sp.Matrix(q2 - (-1)**p * B)) != sp.zeros(2 * n, 2 * n):
                delta = False
        print("    n=%d:  q^2 = id ? %s    q^4 = id ? %s    q^2 = delta_{-1} ? %s"
              % (n, ok2, ok4, delta))

    print("\n  the anti-symmetric elements of -q, i.e. {X | (-q)(X) = -X}:")
    for n in (2, 3, 4):
        ev, od = Qn_basis(n)
        basis = ev + od
        coeffs = sp.symbols('d0:%d' % len(basis))
        X = sum((c * B for c, B in zip(coeffs, basis)), sp.zeros(2 * n, 2 * n))
        C = mq_map(X, n) + X
        eqs = [sp.expand(e) for e in list(C)]
        M = sp.Matrix([[sp.diff(e, c) for c in coeffs] for e in eqs])
        d = len(coeffs) - len(M.rref()[1])
        print("    n=%d:  dim = %d;  dim o(n) = n(n-1)/2 = %d  -> %s"
              % (n, d, n * (n - 1) // 2, "o(n)" if d == n * (n - 1) // 2 else "?"))

    print("\n  Montgomery's condition (Cond), in the reading u^2 in Z\\{0}:")
    print("    Mat(a|b), a != b: an odd u has u^2 = (BC 0 / 0 CB) with BC = t*1_a,")
    print("      CB = t*1_b; taking traces, a*t = tr(BC) = tr(CB) = b*t, so t=0")
    print("      -> u^2 in Z\\{0} is impossible, (Cond) holds vacuously.")
    for (a, b) in ((2, 2), (3, 3)):
        u = sp.zeros(a + b, a + b)
        for i in range(a):
            u[i, a + i] = 1
            u[a + i, i] = 1
        print("    Mat(%d|%d): u = (0 1 / 1 0) is odd, u^2 = 1 in Z, u not in Z"
              " -> (Cond) FAILS" % (a, b))
    for n in (2, 3):
        J = sp.zeros(2 * n, 2 * n)
        for i in range(n):
            J[i, n + i] = 1
            J[n + i, i] = -1
        print("    Q(%d):     u = J is odd, J^2 = %s, J not in Z -> (Cond) FAILS"
              % (n, "-1" if sp.simplify(J * J + sp.eye(2 * n)) == sp.zeros(2 * n, 2 * n) else "?"))


if __name__ == '__main__':
    main()
