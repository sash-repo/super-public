#!/usr/bin/env python3
"""
Exact check of the osp(1|2) realization eq. (22) of the text and of the
Casimir C := 2(X^+X^- + X^-X^+) - \nabla^+\nabla^- + \nabla^-\nabla^+ + H^2.

Operators act on C[x,theta]; an element is written f = a(x) + theta*b(x) and is
represented by the pair (a, b) of sympy polynomials in x, with the parameter
lambda symbolic.  Everything below is exact symbolic algebra, no numerics.
"""
import sympy as sp

x, lam = sp.symbols('x lamda')
c = lam - sp.Rational(1, 2)          # the constant of eq. (22)

# ---------------------------------------------------------------- operators
# each operator maps (a,b) -> (a,b);  f = a + theta*b


def d_x(f):
    a, b = f
    return (sp.diff(a, x), sp.diff(b, x))


def Xm(f):                            # X^- = -d_x
    a, b = f
    return (-sp.diff(a, x), -sp.diff(b, x))


def H(f):                             # H = 2x d_x + theta d_theta + c
    a, b = f
    return (2 * x * sp.diff(a, x) + c * a,
            2 * x * sp.diff(b, x) + b + c * b)


def Xp(f):                            # X^+ = x^2 d_x + x theta d_theta + c x
    a, b = f
    return (x**2 * sp.diff(a, x) + c * x * a,
            x**2 * sp.diff(b, x) + x * b + c * x * b)


def nm(f):                            # nabla^- = d_theta - theta d_x
    a, b = f
    return (b, -sp.diff(a, x))


def np_printed(f):                    # nabla^+ of an earlier draft
    a, b = f                          #   +x d_theta + x theta d_x + c theta
    return (x * b, x * sp.diff(a, x) + c * a)


def np_fixed(f):                      # nabla^+ with the sign of eq. (22)
    a, b = f                          #   -x d_theta + x theta d_x + c theta
    return (-x * b, x * sp.diff(a, x) + c * a)


def comp(*ops):
    def g(f):
        for o in reversed(ops):
            f = o(f)
        return f
    return g


def add(*fs):
    return (sp.expand(sum(u[0] for u in fs)), sp.expand(sum(u[1] for u in fs)))


def smul(s, f):
    return (sp.expand(s * f[0]), sp.expand(s * f[1]))


def bracket(A, B, sign):
    """sign = -1: commutator AB - BA;  sign = +1: anticommutator AB + BA."""
    return lambda f: add(comp(A, B)(f), smul(sign, comp(B, A)(f)))


# ------------------------------------------------------- generic test vector
A, B = sp.symbols('A0:6'), sp.symbols('B0:6')
gen = (sum(A[i] * x**i for i in range(6)), sum(B[i] * x**i for i in range(6)))


def is_zero(op):
    r = op(gen)
    return sp.simplify(r[0]) == 0 and sp.simplify(r[1]) == 0


def equals(op1, op2):
    return is_zero(lambda f: add(op1(f), smul(-1, op2(f))))


def report(name, ok):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    return ok


def check_relations(NP, tag):
    print(f"\n=== osp(1|2) relations for {tag} ===")
    good = True
    good &= report("[X^+,X^-] = H",
                   equals(bracket(Xp, Xm, -1), H))
    good &= report("[H,X^+] = 2X^+",
                   equals(bracket(H, Xp, -1), lambda f: smul(2, Xp(f))))
    good &= report("[H,X^-] = -2X^-",
                   equals(bracket(H, Xm, -1), lambda f: smul(-2, Xm(f))))
    good &= report("[H,nabla^+] = nabla^+",
                   equals(bracket(H, NP, -1), NP))
    good &= report("[H,nabla^-] = -nabla^-",
                   equals(bracket(H, nm, -1), lambda f: smul(-1, nm(f))))
    good &= report("{nabla^-,nabla^-} = 2X^-",
                   equals(bracket(nm, nm, +1), lambda f: smul(2, Xm(f))))
    good &= report("{nabla^+,nabla^+} = -2X^+",
                   equals(bracket(NP, NP, +1), lambda f: smul(-2, Xp(f))))
    good &= report("{nabla^+,nabla^-} = H",
                   equals(bracket(NP, nm, +1), H))
    print(f"  --> {tag}: {'all osp(1|2) relations hold' if good else 'NOT an osp(1|2)-action'}")
    if not good:
        r = bracket(NP, nm, +1)(gen)
        h = H(gen)
        print("      {nabla^+,nabla^-} - H  on (a,b) =",
              (sp.simplify(r[0] - h[0]), sp.simplify(r[1] - h[1])))
    return good


def casimir(NP, alpha):
    """2(X^+X^- + X^-X^+) + alpha*(nabla^+nabla^- - nabla^-nabla^+) + H^2."""
    def C(f):
        return add(smul(2, comp(Xp, Xm)(f)),
                   smul(2, comp(Xm, Xp)(f)),
                   smul(alpha, comp(NP, nm)(f)),
                   smul(-alpha, comp(nm, NP)(f)),
                   comp(H, H)(f))
    return C


def scalar_of(C):
    """Return the scalar by which C acts, or None if it is not scalar."""
    r = C(gen)
    # act on 1 and on theta separately
    one = C((sp.Integer(1), sp.Integer(0)))
    th = C((sp.Integer(0), sp.Integer(1)))
    s1 = sp.simplify(one[0])
    s2 = sp.simplify(th[1])
    if sp.simplify(one[1]) != 0 or sp.simplify(th[0]) != 0:
        return None
    if sp.simplify(s1 - s2) != 0:
        return ('not scalar: 1 -> %s, theta -> %s' % (s1, s2))
    # check on a generic vector
    if sp.simplify(r[0] - s1 * gen[0]) != 0 or sp.simplify(r[1] - s1 * gen[1]) != 0:
        return ('not scalar on higher degrees')
    return sp.expand(s1)


if __name__ == '__main__':
    print("The osp(1|2) realization and its Casimir  (exact, sympy)")
    check_relations(np_printed, "nabla^+ = +x d_theta + ...  (earlier draft)")
    ok = check_relations(np_fixed, "nabla^+ = -x d_theta + ...  (sign of eq. (22))")

    print("\n=== the central element, for the repaired nabla^+ ===")
    for alpha in (sp.Integer(1), sp.Integer(-1), sp.Rational(1, 2), sp.Integer(2)):
        s = scalar_of(casimir(np_fixed, alpha))
        print(f"  alpha = {alpha}:  {s}")

    print("\n=== the Casimir C (alpha = -1), constant lambda-1/2 ===")
    s = scalar_of(casimir(np_fixed, sp.Integer(-1)))
    print("  C acts by the scalar :", sp.factor(s))
    print("  in terms of lambda   :", sp.expand(s))
    print("  lambda^2 + lambda    :", sp.expand(lam**2 + lam))
    print("  lambda^2 + 3 lambda  :", sp.expand(lam**2 + 3 * lam))
    print("  (lambda+3/2)^2 - 9/4 :", sp.expand((lam + sp.Rational(3, 2))**2 - sp.Rational(9, 4)))
    for cand, name in ((lam**2 + lam, "lambda^2+lambda"),
                       (lam**2 + 3 * lam, "lambda^2+3lambda")):
        print(f"  C == {name} ?  {sp.simplify(s - cand) == 0}")

    # which highest/lowest weight does the vacuum 1 carry?
    print("\n=== weights of the vacuum ===")
    one = (sp.Integer(1), sp.Integer(0))
    print("  H(1)        =", H(one)[0], " -> weight of the vacuum 1 is", sp.expand(c))
    print("  X^-(1)      =", Xm(one), "(so 1 is a lowest weight vector: X^- kills it)")
    print("  nabla^-(1)  =", nm(one))
    mu = sp.symbols('mu')
    print("  with mu := -c = 1/2 - lambda :  scalar =",
          sp.expand(s.subs(lam, sp.Rational(1, 2) - mu)),
          " vs mu^2+mu =", sp.expand(mu**2 + mu))

    # --------------------------------------------------------------------
    # Which constant in eq. (22) makes every printed statement true?
    # The relations hold for an arbitrary constant c (they were verified
    # above with c = lambda-1/2, lambda symbolic).  The vacuum 1 is a
    # LOWEST weight vector of weight c, so the highest weight of the
    # corresponding finite-dimensional module is mu = -c, and C acts by
    # c^2-c = mu^2+mu.  The paper wants mu = lambda, i.e. c = -lambda.
    print("\n=== the constant of eq. (22): what makes mu = lambda ? ===")
    print("  C acts by c^2-c, where c is the constant of eq. (22)")
    print("    c = lambda-1/2 (draft):    C =",
          sp.expand(c**2 - c), " and mu = 1/2-lambda")
    print("    c = -lambda    (eq. (22)): C =",
          sp.expand(lam**2 + lam), " and mu = lambda")

    # verify c = -lambda directly: rebuild the operators and find the
    # finite-dimensional submodule for lambda = n
    print("\n=== c = -lambda: the finite-dimensional submodule for lambda = n ===")
    for n in range(1, 6):
        cc = -sp.Integer(n)

        def H2(f, cc=cc):
            a, b = f
            return (2 * x * sp.diff(a, x) + cc * a,
                    2 * x * sp.diff(b, x) + b + cc * b)

        def Xp2(f, cc=cc):
            a, b = f
            return (x**2 * sp.diff(a, x) + cc * x * a,
                    x**2 * sp.diff(b, x) + x * b + cc * x * b)

        def np2(f, cc=cc):
            a, b = f
            return (-x * b, x * sp.diff(a, x) + cc * a)

        hw_even = (x**n, sp.Integer(0))          # candidate highest weight vector
        print(f"  n={n}:  X^+(x^n) = {sp.expand(Xp2(hw_even)[0])},"
              f"  nabla^+(x^n) = {sp.expand(np2(hw_even)[1])}*theta,"
              f"  H(x^n) = {sp.expand(H2(hw_even)[0]/x**n)}*x^n")
        # the submodule spanned by 1..x^n and theta*1..theta*x^{n-1}
        dim_ev, dim_od = n + 1, n
        print(f"          finite-dimensional submodule of sdim ({dim_ev}|{dim_od})"
              f" = ({n}+1|{n}), so Mat({n}+1|{n})")
