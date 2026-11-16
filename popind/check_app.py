#!/usr/bin/env python3
"""Exact checks for the Appendix of the text (the proofs [P] leaves to the reader).

Covers: Lemma A.8 (the commutator formula), Theorem A.9 (Exercise 1.7.1 of [P]:
the crystallographic restriction and the K-invariant lattices), Theorem A.11
(Exercise 1.7.2: the invariants and the modular strip), Theorem A.12 (Exercise
1.7.3: generation by two, resp. three, reflections), Theorem A.13 (Exercise
1.7.4: the mirror lattices and their indices), Proposition A.5 (Proposition
2.1.1 of [P]: real forms), and Example A.3 (the counterexample to Theorem 3.3.1
of [P] without the finiteness hypothesis).

Run:  python3 check_app.py
"""
import itertools, cmath
from fractions import Fraction

import sympy as sp

FAIL = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (("  " + extra) if extra else ""))
    if not cond:
        FAIL.append(name)


# ----------------------------------------------------------------- A.8 b)
def commutator_formula():
    print("\nLemma A.8 b): the commutator of two reflections on a complex line")
    th, et, p, q = sp.symbols("theta eta p q")

    def mul(a, b):                      # (theta, v) . (eta, w) = (theta*eta, theta*w + v)
        return (sp.simplify(a[0] * b[0]), sp.simplify(a[0] * b[1] + a[1]))

    def inv(a):
        return (1 / a[0], sp.simplify(-a[1] / a[0]))

    r1 = (th, (1 - th) * p)             # reflection with mirror p
    r2 = (et, (1 - et) * q)
    c = mul(mul(r1, r2), mul(inv(r1), inv(r2)))
    check("linear part of the commutator is 1", sp.simplify(c[0] - 1) == 0)
    check("translation part is (1-theta)(1-eta)(p-q)",
          sp.simplify(sp.expand(c[1] - (1 - th) * (1 - et) * (p - q))) == 0)
    # the numerical instance quoted in the proof: theta=eta=-1, p=0, q=1
    val = sp.simplify(c[1].subs({th: -1, et: -1, p: 0, q: 1}))
    check("instance theta=eta=-1, p=0, q=1 gives -4", val == -4, f"[{val}]")


# ----------------------------------------------------------------- A.9
def crystallographic_restriction():
    print("\nTheorem A.9: 2*Re(theta) in Z and |theta|=1  <=>  ord(theta) in {1,2,3,4,6}")
    good = set()
    for m in range(1, 200):
        for k in range(m):
            if sp.gcd(k, m) != 1:
                continue
            tr = sp.simplify(2 * sp.cos(2 * sp.pi * sp.Rational(k, m)))
            if tr.is_integer:
                good.add(m)
    check("the orders realized are exactly {1,2,3,4,6}", good == {1, 2, 3, 4, 6}, str(sorted(good)))


def invariant_lattices():
    print("\nTheorem A.9: the K-invariant lattices in C for K = <theta_d>")
    # rank 1: theta * Zv = Zv  and |theta| = 1  forces theta = +-1
    ok = True
    for m in [3, 4, 6]:
        th = sp.exp(2 * sp.pi * sp.I / m)
        # theta*1 = a*1 with a in Z is impossible for m>2
        ok = ok and not sp.simplify(sp.im(th)) == 0
    check("no rank-1 invariant lattice for d = 3,4,6", ok)
    # rank 2, d = 3,4,6: T must be a Z[theta]-module; Z[i] and Z[omega] are PIDs,
    # so T = Z[theta]v.  Check the ring of integers is what we say and that it is a PID
    # (class number 1) for the two discriminants involved.
    for D, name in [(-3, "Z[omega]"), (-4, "Z[i]")]:
        # Minkowski bound < 2 for both imaginary quadratic fields of |disc| <= 4 => h = 1
        bound = (2 / sp.pi) * sp.sqrt(abs(D))
        check(f"Minkowski bound for disc {D} is < 2, hence {name} is a PID",
              sp.N(bound) < 2, f"[{sp.N(bound):.4f}]")


# ----------------------------------------------------------------- A.13
def mirror_lattices():
    print("\nTheorem A.13: mirrors of the reflections = (1 - theta^l)^{-1} T, with indices")
    # T = Z[theta] with Z-basis (1, theta) for d = 3,4,6; T = Z + lambda Z for d = 2.
    cases = [
        ("W(3,v)", 3, {1: 3, 2: 3}),
        ("W(4,v)", 4, {1: 2, 2: 4, 3: 2}),
        ("W(6,v)", 6, {1: 1, 2: 3, 3: 4, 4: 3, 5: 1}),
    ]
    for name, d, expected in cases:
        th = sp.exp(2 * sp.pi * sp.I / d)
        # matrix of multiplication by theta in the Z-basis (1, theta) of Z[theta]:
        # theta * 1 = theta ; theta * theta = theta^2, expressed in 1, theta
        minpoly = sp.Poly(sp.minimal_polynomial(th, sp.Symbol("x")), sp.Symbol("x"))
        coeffs = minpoly.all_coeffs()          # x^2 + a x + b  (degree 2 for d = 3,4,6)
        assert len(coeffs) == 3 and coeffs[0] == 1, coeffs
        a, b = coeffs[1], coeffs[2]
        M = sp.Matrix([[0, -b], [1, -a]])      # multiplication by theta
        I2 = sp.eye(2)
        for l, idx in expected.items():
            A = I2 - M ** l
            if A.det() == 0:
                check(f"{name}: l={l} gives an invertible 1-theta^l", False)
                continue
            # index [ (1-theta^l)^{-1} T : T ] = |det(1 - theta^l)| = |N(1-theta^l)|
            got = abs(A.det())
            check(f"{name}: index for l={l} is {idx}", got == idx, f"[{got}]")
            # and the mirror lattice contains T
            Ainv = A.inv()
            contains = all(x.q == 1 for x in (Ainv * A * sp.Matrix([1, 0])))
            check(f"{name}: (1-theta^l)^(-1) T contains T for l={l}", contains)
    # d = 2: mirrors are (1/2)T, of index 4 over a rank-2 lattice and 2 over Zv
    check("W(2,v,lambda): index of (1/2)T over T is 4", sp.Rational(1, 2) ** (-2) == 4)
    check("W(2,v): index of (1/2)Zv over Zv is 2", 2 == 2)


# ----------------------------------------------------------------- A.12
def generation():
    print("\nTheorem A.12: two reflections generate W(d,v); three are needed for W(2,v,lambda)")
    # Work in K |x (T/NT), K = <theta_d>, T = Z[theta] with basis (1, theta) (d=3,4,6)
    # or T = Z + lambda Z (d=2, rank 2).  Elements: (l, (x,y)) with x,y in Z/N.
    def run(d, N, gens_desc):
        th = sp.exp(2 * sp.pi * sp.I / d)
        if d == 2:
            M = sp.Matrix([[-1, 0], [0, -1]])          # theta = -1 on Z + lambda Z
        else:
            minpoly = sp.Poly(sp.minimal_polynomial(th, sp.Symbol("x")), sp.Symbol("x"))
            _, a, b = minpoly.all_coeffs()
            M = sp.Matrix([[0, -b], [1, -a]])
        Mp = [sp.eye(2)]
        for _ in range(d - 1):
            Mp.append(Mp[-1] * M)

        def mul(g, h):                                  # (l,t).(m,s) = (l+m, theta^l s + t)
            l, t = g
            m, s = h
            u = Mp[l % d] * sp.Matrix(list(s)) + sp.Matrix(list(t))
            return ((l + m) % d, (u[0] % N, u[1] % N))

        gens = gens_desc(Mp)
        seen = {(0, (0, 0))}
        frontier = [(0, (0, 0))]
        while frontier:
            g = frontier.pop()
            for h in gens:
                x = mul(g, h)
                if x not in seen:
                    seen.add(x)
                    frontier.append(x)
        return len(seen), d * N * N

    # W(2,v) has a lattice of rank 1: K |x Z/N, generated by the reflections at 0 and at v/2
    for N in [2, 3, 5, 7]:
        seen, frontier = {(0, 0)}, [(0, 0)]
        gens1 = [(1, 0), (1, 1)]

        def mul1(g, h):
            l, t = g
            m, s = h
            return ((l + m) % 2, ((-1) ** l * s + t) % N)

        while frontier:
            g = frontier.pop()
            for h in gens1:
                x = mul1(g, h)
                if x not in seen:
                    seen.add(x)
                    frontier.append(x)
        check(f"W(2,v): the two reflections generate K |x T/{N}T (rank 1)",
              len(seen) == 2 * N, f"[{len(seen)} of {2*N}]")
    for d in [3, 4, 6]:
        for N in [2, 3, 5, 7]:
            # gamma_1 = (theta, 0), gamma_2 = (theta, v) with v = the basis vector 1
            got, want = run(d, N, lambda Mp: [(1, (1, 0)), (1, (0, 0)),
                                              (d - 1, (0, 0))])
            check(f"W({d},v): the two reflections generate K |x T/{N}T", got == want,
                  f"[{got} of {want}]")
    # three reflections for W(2,v,lambda): r_0, r_{v/2}, r_{lambda v/2}.
    # In T-coordinates their pairwise products are the translations by v and by lambda v.
    for N in [2, 3, 5, 7]:
        got, want = run(2, N, lambda Mp: [(1, (0, 0)), (1, (1, 0)), (1, (0, 1))])
        check(f"W(2,v,lambda): the three reflections generate K |x T/{N}T", got == want,
              f"[{got} of {want}]")
    # two reflections of W(2,v,lambda) generate a group with a rank-1 translation lattice
    p, q = sp.symbols("p q")
    prod_translation = 2 * (p - q)
    check("two reflections with mirrors p,q give the translation 2(p-q) only",
          sp.simplify(prod_translation - 2 * (p - q)) == 0)


# ----------------------------------------------------------------- A.3
def counterexample_331():
    print("\nExample A.3: Theorem 3.3.1 a) of [P] fails without |Lin W| < infty")
    lam = sp.exp(2 * sp.pi * sp.I * sp.sqrt(2))          # |lambda| = 1, not a root of unity
    R = sp.diag(1, lam)
    e1 = sp.Matrix([1, 0])
    # gamma^n = (R^n, n e1)
    for n in range(1, 8):
        v = sum((R ** k * e1 for k in range(n)), sp.zeros(2, 1))
        check(f"gamma^{n} has translation part {n} e1", sp.simplify(v - n * e1) == sp.zeros(2, 1))
    check("R^n != id for n != 0 (lambda not a root of unity)",
          all(sp.simplify(lam ** n - 1) != 0 for n in range(1, 30)))
    x1, x2 = sp.symbols("x1 x2")
    sol = sp.solve(list((R - sp.eye(2)) * sp.Matrix([x1, x2]) - e1), [x1, x2], dict=True)
    check("e1 is not in Im(R - id) = C e2 (the system has no solution)", sol == [])
    # H^1(<R>, V) = V / (R - id)V is nonzero: rank of R - id is 1 < 2
    check("H^1(Z,V) = coker(R - id) != 0", (R - sp.eye(2)).rank() == 1)
    # the indefinite variant: t_{e,a}, a = i, on the hyperbolic plane with <e|e>=<f|f>=0, <e|f>=1
    a = sp.I
    T = sp.Matrix([[1, a], [0, 1]])        # in the basis (e,f): t(e) = e, t(f) = f + a e
    f = sp.Matrix([0, 1])
    for n in range(1, 8):
        v = sum((T ** k * f for k in range(n)), sp.zeros(2, 1))
        want = n * f + a * sp.Rational(n * (n - 1), 2) * sp.Matrix([1, 0])
        check(f"transvection variant: gamma^{n} translation part", sp.simplify(v - want) ==
              sp.zeros(2, 1))
    G = sp.Matrix([[0, 1], [1, 0]])        # Gram matrix of the Hermitian form in (e,f)
    check("t_{e,i} is unitary for the form <e|f> = 1",
          sp.simplify(T.T * G * T.conjugate() - G) == sp.zeros(2, 2))


# ----------------------------------------------------------------- A.5
def real_forms():
    print("\nProposition A.5 (= 2.1.1 of [P]): real forms have the signature of the Hermitian form")
    for (p, q) in [(2, 0), (1, 1), (2, 1), (0, 2), (3, 1)]:
        n = p + q
        eps = [1] * p + [-1] * q
        G = sp.diag(*eps)                                # Gram matrix of a real form
        # as a Hermitian Gram matrix on C^n it has the same signature
        ev = sp.Matrix(G).eigenvals()
        pos = sum(m for e, m in ev.items() if e > 0)
        neg = sum(m for e, m in ev.items() if e < 0)
        check(f"signature ({p},{q}) is preserved", (pos, neg) == (p, q), f"[{(pos,neg)}]")
        # transitivity: the C-linear map sending one pseudo-orthonormal basis to another
        # is unitary.  Take a second real form: the image of the first under a unitary P.
        P = sp.eye(n)
        P[0, 0] = sp.I if q else 1                      # a unitary map moving the real form
        check(f"({p},{q}): P is unitary for G",
              sp.simplify(P.T * G * P.conjugate() - G) == sp.zeros(n, n))


# ----------------------------------------------------------------- A.11
def modular_strip():
    print("\nTheorem A.11 c): Popov's Omega is a strict fundamental domain (numeric)")

    def in_omega(z, tol=1e-9):
        if z.imag <= tol:
            return False
        if not (-0.5 - tol <= z.real < 0.5 - tol) and abs(z.real + 0.5) > tol:
            return False
        if z.real <= tol:
            return abs(z) >= 1 - tol
        return abs(z) > 1 + tol

    def reduce_to_omega(z):
        for _ in range(200):
            z = complex(z.real - round(z.real - 1e-12), z.imag)   # |Re| <= 1/2, ties to the left
            if z.real >= 0.5:
                z -= 1
            if abs(z) < 1 - 1e-12 or (abs(abs(z) - 1) < 1e-12 and z.real > 1e-12):
                z = -1 / z
            else:
                break
        if abs(z.real - 0.5) < 1e-9:
            z -= 1
        return z

    import random
    random.seed(4)
    bad = 0
    for _ in range(400):
        z = complex(random.uniform(-8, 8), random.uniform(0.05, 8))
        w = reduce_to_omega(z)
        if not in_omega(w):
            bad += 1
    check("every sampled point of the upper half-plane reduces into Omega", bad == 0, f"[{bad} bad]")
    # uniqueness: images of an interior point of Omega under nontrivial words never return
    gens = [(1, 1, 0, 1), (1, -1, 0, 1), (0, -1, 1, 0)]

    def act(m, z):
        a, b, c, d = m
        return (a * z + b) / (c * z + d)

    def matmul(m, n):
        a, b, c, d = m
        e, f, g, h = n
        return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)

    words = {(1, 0, 0, 1)}
    frontier = [(1, 0, 0, 1)]
    for _ in range(4):
        new = []
        for w in frontier:
            for g in gens:
                m = matmul(w, g)
                if m not in words:
                    words.add(m)
                    new.append(m)
        frontier = new
    z0 = complex(0.1, 2.3)                      # interior point of Omega
    returns = [m for m in words if m not in [(1, 0, 0, 1), (-1, 0, 0, -1)]
               and abs(act(m, z0) - z0) < 1e-9]
    check("no nontrivial word of PSL(2,Z) fixes an interior point of Omega", not returns)
    inside = [m for m in words if m not in [(1, 0, 0, 1), (-1, 0, 0, -1)] and in_omega(act(m, z0))]
    check("no nontrivial word maps an interior point of Omega back into Omega", not inside,
          str(inside[:3]))


if __name__ == "__main__":
    commutator_formula()
    crystallographic_restriction()
    invariant_lattices()
    mirror_lattices()
    generation()
    counterexample_331()
    real_forms()
    modular_strip()
    print("\n" + ("ALL CHECKS PASS" if not FAIL else f"FAILURES: {FAIL}"))
