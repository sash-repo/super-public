"""
Statement 6.2.6: the contact condition, the frame matrix and the integrality of the exponent
of Radul's Bott cocycle (190) on the group of contact diffeomorphisms of S^{1|2}; [Ra] =
Radul's paper, Section 4, "Bott cocycles for K(2) and K(2)^+" (here eta_i, al_i, be_i are the
rho_i, sigma_i, beta_j of the Statement, and xi_i its phi_i).

What is checked, each from [Ra]'s own definitions (nn. 1.4 and 4.1-4.3):

 1) the matrix logarithm of  (A B; -B A):  his displayed formula on p. 48 prints
    ln(A^2+B^2) where the matrix logarithm gives  1/2 ln(A^2+B^2);  and what the
    slip does and does not touch.

 2) his eq. (10) on p. 49: for
        G:  x = f(y,eta),
            xi_1 = al_1 + a eta_1 + b eta_2 + be_1 eta_1 eta_2,
            xi_2 = al_2 + c eta_1 + d eta_2 + be_2 eta_1 eta_2
    to be contact, i.e. to satisfy his own criterion of n. 1.4,
        theta_{eta_i}(f) + sum_j phi_j theta_{eta_i}(phi_j) = 0,
    he states  a = d,  c = -b,  be_1 = -al_2',  be_2 = -al_1'.

 3) the multiplier m = f' - sum phi_j phi_j' of his n. 1.4, and the frame matrix
    (b_ij) = (theta_{eta_i} phi_j) of his n. 1.4 B), against the M(G) of p. 49.

 4) the integral of p. 49:  int < (0 1; -1 0), ln M(G) > Ber(y,eta)
    = int_{S^1} (b a' - a b')/(a^2+b^2) dy = int psi' dy, for psi with
    cos psi = a/sqrt(a^2+b^2), sin psi = -b/sqrt(a^2+b^2);  his pairing being
    < (a b; -b a), (c d; -d c) > = ad - bc.

The odd coefficient functions al_i, be_i are carried honestly: they take values
in an auxiliary Grassmann algebra Lambda(eps_1, eps_2), so that al_1 al_2 =
- al_2 al_1 and al_i al_i = 0 hold as they must.  Odd labels: 1,2 = eta_1,eta_2;
3,4 = eps_1,eps_2.

Heading 1) of the Statement: the parts "the matrix (a b; c d) is conformal" and "the solution is
unique and is be_1 = -al_2', be_2 = +al_1'"; heading 2): parts 3)-4) above; heading 3): part 1).
Run:  python3 bott2.py
"""

import itertools
import sympy as sp

LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


CHECKS = []


def check(name, ok, extra=""):
    CHECKS.append((name, bool(ok)))
    say("  [%s] %s%s" % ("ok" if ok else "FAIL", name, ("  -- " + extra) if extra else ""))


y = sp.Symbol('y')
ETA, EPS = (1, 2), (3, 4)
ALL = ETA + EPS
KEYS = [S for r in range(5) for S in itertools.combinations(ALL, r)]


# ------------------------------- Lambda(eta_1,eta_2,eps_1,eps_2) over C^infty(S^1)

def gadd(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = sp.expand(out.get(k, 0) + v)
    return {k: v for k, v in out.items() if v != 0}


def gsmul(c, f):
    return {k: sp.expand(c * v) for k, v in f.items() if sp.expand(c * v) != 0}


def _sort(S):
    S = list(S)
    sgn = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] == S[j + 1]:
                return (), 0
            if S[j] > S[j + 1]:
                S[j], S[j + 1] = S[j + 1], S[j]
                sgn = -sgn
    if len(set(S)) != len(S):
        return (), 0
    return tuple(S), sgn


def gmul(f, g):
    out = {}
    for S1, c1 in f.items():
        for S2, c2 in g.items():
            S, sgn = _sort(S1 + S2)
            if sgn == 0:
                continue
            out[S] = sp.expand(out.get(S, 0) + sgn * c1 * c2)
    return {k: v for k, v in out.items() if v != 0}


def gd_y(f):
    return {k: sp.diff(v, y) for k, v in f.items() if sp.diff(v, y) != 0}


def gd_eta(i, f):
    out = {}
    for S, c in f.items():
        if i in S:
            p = S.index(i)
            T = S[:p] + S[p + 1:]
            out[T] = sp.expand(out.get(T, 0) + (-1) ** p * c)
    return {k: v for k, v in out.items() if v != 0}


def gen(i):
    return {(i,): sp.Integer(1)}


def theta(i, f):
    """[Ra], n. 1.4: theta_{eta_i} = d_{eta_i} - eta_i d_y  (i = 1, 2)."""
    return gadd(gd_eta(i, f), gsmul(-1, gmul(gen(i), gd_y(f))))


def ber(f):
    """the Berezin integral over eta_1, eta_2 -- the coefficient of eta_1 eta_2,
    the eps being spectators."""
    out = {}
    for S, c in f.items():
        if 1 in S and 2 in S:
            T = tuple(k for k in S if k not in ETA)
            # bring eta_1 eta_2 to the front
            p = S.index(1) + (S.index(2) - 1)
            out[T] = sp.expand(out.get(T, 0) + (-1) ** p * c)
    return {k: v for k, v in out.items() if v != 0}


# ==================================================== 1) the matrix logarithm

def check_log():
    say("")
    say("--- 1) the matrix logarithm of (A B; -B A), p. 48 " + "-" * 23)
    r, phi = sp.symbols('r varphi', positive=True)
    A, B = r * sp.cos(phi), r * sp.sin(phi)
    J = sp.Matrix([[0, 1], [-1, 0]])
    I2 = sp.eye(2)
    M = sp.Matrix([[A, B], [-B, A]])
    exp_ours = sp.simplify(sp.sqrt(A ** 2 + B ** 2) * (sp.cos(phi) * I2 + sp.sin(phi) * J))
    exp_his = sp.simplify((A ** 2 + B ** 2) * (sp.cos(phi) * I2 + sp.sin(phi) * J))
    check("exp( 1/2 ln(A^2+B^2) I + phi J ) = (A B; -B A)",
          sp.simplify(exp_ours - M) == sp.zeros(2, 2))
    check("exp( ln(A^2+B^2) I + phi J ) is NOT (A B; -B A): it is sqrt(A^2+B^2)"
          " times too big", sp.simplify(exp_his - M) != sp.zeros(2, 2))
    say("     so the display of p. 48 wants 1/2 ln(A^2+B^2), not ln(A^2+B^2).")

    def pair(P, Q):
        """his < (a b; -b a), (c d; -d c) > = ad - bc, i.e. the determinant form
        in the coordinates (a,b) and (c,d)."""
        return P[0, 0] * Q[0, 1] - P[0, 1] * Q[0, 0]
    check("< J, I > = -1 and < J, J > = 0, so the scalar part of ln M does NOT"
          " drop out of the pairing", pair(J, I2) == -1 and pair(J, J) == 0)
    say("     -- it drops out of the integral of p. 49 for another reason: the")
    say("        Berezin integral keeps the eta_1 eta_2-component alone, and the")
    say("        scalar summand of ln M(G) carries no eta at all.  In Bott_2")
    say("        itself, however, the nilpotent part of one factor meets the")
    say("        scalar part of the other, so there the 1/2 does tell.")


# ============================================== 2) the contact conditions (10)

def setup():
    a, b, c, d = [sp.Function(nm)(y) for nm in ('a', 'b', 'c', 'd')]
    A1, A2 = [sp.Function(nm)(y) for nm in ('A_1', 'A_2')]
    B = sp.symbols('B11 B12 B21 B22', cls=sp.Function)
    B11, B12, B21, B22 = [f(y) for f in B]
    # al_1 = A_1 eps_1,  al_2 = A_2 eps_2,  be_j = B_{j1} eps_1 + B_{j2} eps_2
    al1 = {(3,): A1}
    al2 = {(4,): A2}
    be1 = {(3,): B11, (4,): B12}
    be2 = {(3,): B21, (4,): B22}
    phi1 = gadd(al1, gsmul(a, gen(1)), gsmul(b, gen(2)),
                gmul(be1, gmul(gen(1), gen(2))))
    phi2 = gadd(al2, gsmul(c, gen(1)), gsmul(d, gen(2)),
                gmul(be2, gmul(gen(1), gen(2))))
    return (a, b, c, d, A1, A2, B11, B12, B21, B22), phi1, phi2


def general_f():
    """the most general even f on Lambda(eta,eps): even coefficient functions on
    the monomials of even total degree, odd ones (i.e. eps-linear) on the rest."""
    unk, f = [], {}
    for S in KEYS:
        if len(S) % 2 == 0:
            g = sp.Function('f%s' % ("".join(map(str, S)) or "0"))(y)
            unk.append(g)
            f[S] = g
    return f, unk


def contact():
    """f enters the criterion only through theta_i(f), so eliminate it:  with
    W_i := sum_j phi_j theta_i(phi_j) the criterion reads theta_i(f) = -W_i, and
    since theta_i theta_i = -d_y and theta_1 theta_2 + theta_2 theta_1 = 0, such an
    f exists if and only if
        (C1)  theta_1(W_1) = theta_2(W_2),      (C2)  theta_1(W_2) + theta_2(W_1) = 0.
    These are conditions on phi alone; they are what his eq. (10) records."""
    say("")
    say("--- 2) his eq. (10): when is G contact?  p. 49 " + "-" * 26)
    (a, b, c, d, A1, A2, B11, B12, B21, B22), phi1, phi2 = setup()
    check("theta_i theta_i = -d_y on every monomial", _sq_ok())
    W = []
    for i in (1, 2):
        W.append(gadd(gmul(phi1, theta(i, phi1)), gmul(phi2, theta(i, phi2))))
    C = gadd(theta(1, W[0]), gsmul(-1, theta(2, W[1])))
    D = gadd(theta(1, W[1]), theta(2, W[0]))
    eqs = [sp.expand(v) for v in list(C.values()) + list(D.values())]
    eqs = [e for e in eqs if e != 0]
    say("     the f-free conditions (C1), (C2) have %d non-zero components:" % len(eqs))
    for e in eqs:
        say("        %s = 0" % sp.factor(e))
    # his (10)
    his = {d: a, c: -b, B11: sp.Integer(0), B12: -sp.diff(A2, y),
           B21: -sp.diff(A1, y), B22: sp.Integer(0)}
    left = [sp.simplify(e.subs(his).doit()) for e in eqs]
    check("[Ra]'s (10) as printed does NOT solve them: the last condition is"
          " off by a sign", any(v != 0 for v in left),
          "residues all proportional to al_1\': %s" % [sp.factor(v) for v in left if v != 0])
    fixed = dict(his)
    fixed[B21] = sp.diff(A1, y)
    left2 = [sp.simplify(e.subs(fixed).doit()) for e in eqs]
    check("with be_2 = +al_1\' they are solved identically", all(v == 0 for v in left2))
    # and it is the only solution on the branch a = d, c = -b: solve for the B's
    rot = {d: a, c: -b}
    lin = [sp.expand(e.subs(rot).doit()) for e in eqs]
    lin = [e for e in lin if e != 0]
    unk = [B11, B12, B21, B22]
    sol = sp.solve(lin, unk, dict=True)
    say("     solving those for (B11, B12, B21, B22) on the branch a=d, c=-b:")
    for s0 in sol:
        for k, v in s0.items():
            say("        %s = %s" % (k, sp.simplify(v)))
    ok = (len(sol) == 1 and
          sp.simplify(sol[0].get(B11, B11)) == 0 and
          sp.simplify(sol[0].get(B22, B22)) == 0 and
          sp.simplify(sol[0].get(B12, B12) + sp.diff(A2, y)) == 0 and
          sp.simplify(sol[0].get(B21, B21) - sp.diff(A1, y)) == 0)
    check("the solution is unique and is be_1 = -al_2\', be_2 = +al_1\'", ok)
    say("     The RELATIVE sign is convention-free: moving the odd coefficient to")
    say("     the other side of eta_1 eta_2, or flipping the sign of theta_i, or")
    say("     interchanging eta_1 and eta_2, multiplies both conditions by the same")
    say("     factor.  [Ra] prints them with the SAME sign, and they are opposite.")
    return eqs


def _sq_ok():
    ok = True
    for S in KEYS:
        f = {S: sp.Function('h')(y)}
        for i in (1, 2):
            if gadd(theta(i, theta(i, f)), gd_y(f)):
                ok = False
    return ok


def conformal():
    """with the odd coefficients switched off, the criterion says exactly that the
    matrix (a b; c d) is conformal -- his condition A) of n. 1.4 -- whose
    orientation-preserving half is a = d, c = -b."""
    say("")
    say("     with al = be = 0, the conditions on (a,b,c,d) alone:")
    a, b, c, d = [sp.Function(nm)(y) for nm in ('a', 'b', 'c', 'd')]
    phi1 = gadd(gsmul(a, gen(1)), gsmul(b, gen(2)))
    phi2 = gadd(gsmul(c, gen(1)), gsmul(d, gen(2)))
    f, unk = general_f()
    eqs = []
    for i in (1, 2):
        E = gadd(theta(i, f), gmul(phi1, theta(i, phi1)), gmul(phi2, theta(i, phi2)))
        for S, v in E.items():
            eqs.append(sp.expand(v))
    # eliminate f: the conditions free of f are the differences/sums below
    red = []
    for e in eqs:
        if not any(e.has(u) for u in unk) and not any(e.has(sp.diff(u, y)) for u in unk):
            red.append(e)
    # pairs that cancel the f-terms
    combos = [sp.expand(eqs[i] - eqs[j]) for i in range(len(eqs)) for j in range(len(eqs))]
    combos += [sp.expand(eqs[i] + eqs[j]) for i in range(len(eqs)) for j in range(len(eqs))]
    free = set()
    for e in combos:
        if e == 0:
            continue
        if not any(e.has(u) or e.has(sp.diff(u, y)) for u in unk):
            free.add(sp.factor(e))
    say("        %s" % sorted({str(e) for e in free}))
    want1 = sp.expand(a ** 2 + c ** 2 - b ** 2 - d ** 2)
    want2 = sp.expand(a * b + c * d)
    ok1 = any(sp.simplify(sp.expand(e) - want1) == 0 or
              sp.simplify(sp.expand(e) + want1) == 0 for e in free)
    ok2 = any(sp.simplify(sp.expand(e) - 2 * want2) == 0 or
              sp.simplify(sp.expand(e) + 2 * want2) == 0 or
              sp.simplify(sp.expand(e) - want2) == 0 or
              sp.simplify(sp.expand(e) + want2) == 0 for e in free)
    check("the criterion is exactly a^2+c^2 = b^2+d^2 and ab+cd = 0, i.e."
          " (a b; c d) conformal -- his condition A) of n. 1.4", ok1 and ok2)
    check("its orientation-preserving branch is a = d, c = -b -- his (10)",
          sp.simplify(want1.subs({d: a, c: -b})) == 0 and
          sp.simplify(want2.subs({d: a, c: -b})) == 0)


# ============================================= 3)-4) M(G) and the p. 49 integral

def frame_and_integral():
    say("")
    say("--- 3)-4) M(G) and the integral of p. 49 " + "-" * 31)
    (a, b, c, d, A1, A2, B11, B12, B21, B22), phi1, phi2 = setup()
    sub = {c: -b, d: a, B11: 0, B12: -sp.diff(A2, y), B21: -sp.diff(A1, y), B22: 0}

    def ap(f):
        return {k: sp.expand(v.subs(sub).doit()) for k, v in f.items()
                if sp.expand(v.subs(sub).doit()) != 0}
    p1, p2 = ap(phi1), ap(phi2)
    # b_ij = theta_i(phi_j)
    bmat = [[theta(i, p) for p in (p1, p2)] for i in (1, 2)]

    def comp(f, S):
        return sp.expand(f.get(tuple(sorted(S)), 0))
    # the eta-free part of (b_ij)
    B0 = sp.Matrix(2, 2, lambda i, j: comp(bmat[i][j], ()))
    say("     the eta-free part of (theta_i phi_j) = %s" % (B0.tolist(),))
    check("it is (a -b; b a) up to transposition, i.e. conformal with the"
          " rotation sign", sp.simplify(B0[0, 0] - B0[1, 1]) == 0 and
          sp.simplify(B0[0, 1] + B0[1, 0]) == 0)
    # the nilpotent (eta_1 eta_2) part, on the branch al = 0
    sub0 = {A1: sp.Integer(0), A2: sp.Integer(0)}
    N = sp.Matrix(2, 2, lambda i, j: sp.expand(comp(bmat[i][j], (1, 2))))
    say("     the eta_1 eta_2-part of (theta_i phi_j)      = %s" % (N.tolist(),))

    # M(G) = (b_ij)^{-1};  with l := the nilpotent part, M = B0^{-1} - B0^{-1} l B0^{-1}
    Binv = B0.inv()
    say("     (theta_i phi_j)^{-1} eta-free part           = %s"
        % (sp.simplify(Binv).tolist(),))
    check("(b_ij)^{-1} = 1/(a^2+b^2) (a b; -b a) on the eta-free part"
          " -- his inverse of p. 49",
          sp.simplify(Binv - sp.Matrix([[a, b], [-b, a]]) / (a ** 2 + b ** 2))
          == sp.zeros(2, 2))

    # ln M(G) = ln(even part) + (even part)^{-1} * (nilpotent part), l^2 = 0.
    # His M(G) of p. 49 has (a -b; b a) as its eta-free part and, on al = 0, the
    # eta_1 eta_2-part (-b' -a'; a' -b').  We test the integral he computes:
    Msc = sp.Matrix([[a, -b], [b, a]])
    Mnil = sp.Matrix([[-sp.diff(b, y), -sp.diff(a, y)],
                      [sp.diff(a, y), -sp.diff(b, y)]])
    L = sp.simplify(Msc.inv() * Mnil)
    say("     (even part)^{-1} (nilpotent part)            = %s" % (sp.simplify(L).tolist(),))
    check("it is of the shape (p q; -q p)", sp.simplify(L[0, 0] - L[1, 1]) == 0 and
          sp.simplify(L[0, 1] + L[1, 0]) == 0)
    J = sp.Matrix([[0, 1], [-1, 0]])

    def pair(P, Q):
        return P[0, 0] * Q[0, 1] - P[0, 1] * Q[0, 0]
    val = sp.simplify(pair(J, L))
    his = sp.simplify((b * sp.diff(a, y) - a * sp.diff(b, y)) / (a ** 2 + b ** 2))
    say("     < J, that > = %s" % val)
    say("     [Ra] prints  (b a' - a b')/(a^2+b^2) = %s" % his)
    check("the two agree up to the overall sign of his pairing",
          sp.simplify(val - his) == 0 or sp.simplify(val + his) == 0,
          "val %s his" % ("=" if sp.simplify(val - his) == 0 else "= -"))
    # psi with cos psi = a/r, sin psi = -b/r
    psi = sp.atan2(-b, a)
    check("that quantity is psi', for psi with cos psi = a/sqrt(a^2+b^2),"
          " sin psi = -b/sqrt(a^2+b^2) -- so the integral is 2 pi Z",
          sp.simplify(sp.diff(psi, y) - his) == 0)


# ============================================================================ main

if __name__ == "__main__":
    say("=" * 74)
    say("[Ra] pp. 47-50, Section 4: the Bott cocycles for K(2) and K(2)^+")
    say("=" * 74)
    check_log()
    conformal()
    contact()
    frame_and_integral()
    say("")
    say("=" * 74)
    bad = [nm for nm, ok in CHECKS if not ok]
    say("%d checks, %d failures" % (len(CHECKS), len(bad)))
    for nm in bad:
        say("  FAILED: %s" % nm)
    pass
