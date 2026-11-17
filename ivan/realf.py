#!/usr/bin/env python3
r"""
Section "On versions over R": the two anti-linear involutions of gl(lambda),
and what to call the real forms they cut out (Proposition 4.1.1 of the text).

[LSe, 6.3] gives one anti-linear map omega, on the generators of the principal
sl(2):   omega(X) = Y,  omega(Y) = X,  omega(H) = H.
One checks that omega is an anti-AUTOMORPHISM, so the
automorphism is -omega.  This engine verifies, in the finite-dimensional model
gl(lambda)|_{lambda=n} = gl(n), that

  1) the map omega_0(A) := conj(A)^t realizes [LSe, 6.3]: it is anti-linear and
     acts on the principal sl(2) exactly by X <-> Y, H -> H;
  2) omega_0 is an anti-automorphism of the Lie algebra and NOT an
     automorphism, whereas -omega_0 is an involutive anti-linear automorphism;
  3) the fixed points of -omega_0 are u(n)         -- the compact form;
  4) the outer automorphism phi = -t of [LSe, 6.1] carries -omega_0 to the
     second involution, (-omega_0) o phi : A -> conj(A), whose fixed points are
     gl(n;R);
  5) so the two involutions differ by the generator of Out gl(lambda) = Z/2,
     and the pair of real forms is (u(n), gl(n;R)).

Both involutions exist for every lambda with lambda^2 real -- the relations of
[LSe, 6.2] constrain the scalar c by c^2 = 1 only -- hence BOTH families of
lambda, the real one and the imaginary one, carry the same two forms.  That is
the argument for naming them u(lambda) and gl(lambda;R) in both cases, with no
decoration recording which family lambda is in.

Exact symbolic arithmetic (sympy); no numerics.
"""
import itertools
import sympy as sp

LOG = None


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    if LOG:
        LOG.write(msg + "\n")
        LOG.flush()


def principal_sl2(n):
    """H, X, Y spanning the principal sl(2) of gl(n), normalized so that
    Y = X^t (both real), [H,X] = 2X, [H,Y] = -2Y, [X,Y] = H."""
    H = sp.zeros(n, n)
    for i in range(n):
        H[i, i] = sp.Integer(n - 1 - 2 * i)
    X = sp.zeros(n, n)
    Y = sp.zeros(n, n)
    for i in range(n - 1):
        c = sp.sqrt(sp.Integer((i + 1) * (n - 1 - i)))
        X[i, i + 1] = c
        Y[i + 1, i] = c
    return H, X, Y


def br(A, B):
    return sp.expand(A * B - B * A)


def zero(M, n):
    return sp.simplify(sp.Matrix(M)) == sp.zeros(n, n)


def conj_t(A):
    """omega_0 : A -> conj(A)^t, anti-linear."""
    return A.conjugate().T


def basis_gl(n):
    out = []
    for i in range(n):
        for j in range(n):
            E = sp.zeros(n, n)
            E[i, j] = 1
            out.append(E)
    return out


def check(n):
    say("")
    say(f"  gl({n}):")
    H, X, Y = principal_sl2(n)
    # the principal sl(2) relations
    ok = (zero(br(H, X) - 2 * X, n) and zero(br(H, Y) + 2 * Y, n)
          and zero(br(X, Y) - H, n))
    say(f"    principal sl(2) relations [H,X]=2X, [H,Y]=-2Y, [X,Y]=H : {ok}")

    # 1) omega_0 realizes [LSe, 6.3] on the generators
    r = (zero(conj_t(X) - Y, n) and zero(conj_t(Y) - X, n)
         and zero(conj_t(H) - H, n))
    say(f"    omega_0(X)=Y, omega_0(Y)=X, omega_0(H)=H            : {r}")

    B = basis_gl(n)
    # anti-linearity on a sample: omega_0(i*A) = -i*omega_0(A)
    al = all(zero(conj_t(sp.I * A) + sp.I * conj_t(A), n) for A in B)
    say(f"    omega_0 is anti-linear                              : {al}")

    # 2) automorphism?  anti-automorphism?
    isauto = all(zero(conj_t(br(A, C)) - br(conj_t(A), conj_t(C)), n)
                 for A, C in itertools.product(B, repeat=2))
    isanti = all(zero(conj_t(br(A, C)) + br(conj_t(A), conj_t(C)), n)
                 for A, C in itertools.product(B, repeat=2))
    say(f"    omega_0 is an automorphism / anti-automorphism      : "
        f"{isauto} / {isanti}")

    def m_omega(A):
        return -conj_t(A)

    isauto2 = all(zero(m_omega(br(A, C)) - br(m_omega(A), m_omega(C)), n)
                  for A, C in itertools.product(B, repeat=2))
    inv2 = all(zero(m_omega(m_omega(A)) - A, n) for A in B)
    say(f"    -omega_0 is an automorphism, and involutive         : "
        f"{isauto2}, {inv2}")

    # 3) fixed points of -omega_0 : anti-Hermitian = u(n)
    #    (a real form: real dimension n^2 inside the real 2n^2 of gl(n;C))
    def fixed_real_dim(f):
        """real dimension of {A | f(A) = A}, computed over R by splitting
        A = P + iQ with P,Q real."""
        rows = []
        syms = []
        P = sp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                a = sp.Symbol(f"a_{i}_{j}", real=True)
                b = sp.Symbol(f"b_{i}_{j}", real=True)
                syms += [a, b]
                P[i, j] = a + sp.I * b
        D = sp.expand(f(P) - P)
        eqs = []
        for i in range(n):
            for j in range(n):
                e = sp.expand(D[i, j])
                eqs.append(sp.re(e))
                eqs.append(sp.im(e))
        A, _ = sp.linear_eq_to_matrix(eqs, syms)
        return len(syms) - A.rank()

    d1 = fixed_real_dim(m_omega)
    herm = all(zero(m_omega(A) - A, n) ==
               zero(conj_t(A) + A, n) for A in B)
    say(f"    dim_R Fix(-omega_0) = {d1}  (= n^2 = {n * n}); "
        f"Fix = anti-Hermitian = u({n}): {d1 == n * n}")

    # 4) the outer automorphism phi = -t, and the second involution
    def phi(A):
        return -A.T

    def second(A):
        return m_omega(phi(A))

    same_as_conj = all(zero(second(A) - A.conjugate(), n) for A in B)
    isauto3 = all(zero(second(br(A, C)) - br(second(A), second(C)), n)
                  for A, C in itertools.product(B, repeat=2))
    inv3 = all(zero(second(second(A)) - A, n) for A in B)
    d2 = fixed_real_dim(second)
    say(f"    (-omega_0) o phi is A -> conj(A)                    : "
        f"{same_as_conj}")
    say(f"    it is an involutive automorphism                    : "
        f"{isauto3}, {inv3}")
    say(f"    dim_R Fix = {d2} (= n^2); Fix = real matrices = gl({n};R): "
        f"{d2 == n * n}")
    return ok and r and al and (not isauto) and isanti and isauto2


def main():
    global LOG
    LOG = open("realf.log", "w")
    say("=" * 78)
    say("The two anti-linear involutions of gl(lambda) and the names of the")
    say("real forms they cut out.  realf.py, exact arithmetic.")
    say("=" * 78)
    for n in (2, 3, 4):
        check(n)
    say("")
    say("=" * 78)
    say("CONCLUSION")
    say("=" * 78)
    say("omega of [LSe, 6.3] is an anti-automorphism; the automorphism is")
    say("-omega, whose fixed points are the COMPACT form u(n) -- not a form")
    say("special to imaginary lambda.  The second involution is -omega o phi,")
    say("phi = -t being the generator of Out gl(lambda) = Z/2 of [LSe, 6.1],")
    say("and it cuts out gl(n;R).  Neither construction uses the value of")
    say("lambda: by the relations of [LSe, 6.2] the scalar c obeys c^2 = 1 for")
    say("EVERY lambda with lambda^2 real.  Hence the same two names serve both")
    say("families, and the natural notation is")
    say("")
    say("     u(lambda)  and  gl(lambda;R)   for every lambda with lambda^2 in R,")
    say("")
    say("the condition lambda^2 in R being a property of the parameter, not")
    say("part of the name.  No decoration -- and no division by sqrt(-1) -- is")
    say("needed; when the imaginary family has to be visible on the page, write")
    say("the parameter as lambda = sqrt(-1) t with t real.")
    LOG.close()


if __name__ == "__main__":
    main()
