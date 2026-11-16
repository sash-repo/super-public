r"""
The proof of Theorem 3.5.3 of the text: ber of the Jacobi matrix of a C-point of
SOSp(1|2) equals (Cx+D+delta theta)^{-1}.  The four numerators X, Y, Z, V of the
Jacobi matrix are recomputed from scratch.

Setting.  A C-point of SOSp(1|2) acts by
      tilde x = (Ax + B + gamma th)/(Cx + D + delta th),
      tilde th = (alpha x + beta + E th)/(Cx + D + delta th),
with A, B, C, D, E even, alpha, beta, gamma, delta odd, and (sysol):
      gamma = alpha B - beta A,  delta = alpha D - beta C,
      E = 1 - alpha beta,        AD - BC = 1 + alpha beta.

Set W := (Cx + D + delta th)^2 and X := W*R, Y := W*T, Z := W*S, V := W*U for
      R = d(tilde x)/dx,  T = d(tilde x)/dth,  S = d(tilde th)/dx,  U = d(tilde th)/dth,
then uses  ber = (XV - ZY)/V^2.

We compute X, Y, Z, V, compare them with a candidate set of numerators (whose V
has no theta-term), and check ber = (XV-ZY)/V^2
= (Cx+D+delta th)^{-1}.

Ground ring: the Grassmann algebra on the odd generators alpha, beta and the odd
coordinate theta, over rational functions in A, B, C, x.  D is eliminated by
D = (1 + BC + alpha beta)/A, which solves AD - BC = 1 + alpha beta.
"""

import random
import sympy as sp

x = sp.Symbol('x')
GENS = ('al', 'be', 'th')          # odd generators, in this order

# A, B, C are specialised to random rationals (x stays symbolic).  Every identity below
# is an identity of rational functions in A, B, C, x of bounded degree, so agreement at
# a batch of independent random points is a decisive test; SAMPLES points are used.
SAMPLES = 8
A = B = C = None                   # filled in by run_at_point()


# ------------------------------------------------------------------ Grassmann algebra
# an element is a dict {sorted tuple of generators: even coefficient}

def NORM(e):
    return sp.cancel(sp.together(e))


def _norm(S):
    """sort a tuple of generators, return (sorted tuple, sign); 0 if a repeat occurs"""
    S = list(S)
    s = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if GENS.index(S[j]) > GENS.index(S[j + 1]):
                S[j], S[j + 1] = S[j + 1], S[j]
                s = -s
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            return (), 0
    return tuple(S), s


def el(*terms):
    """el((coeff, ('al','th')), ...) -> element"""
    out = {}
    for c, S in terms:
        T, s = _norm(S)
        if s == 0:
            continue
        out[T] = NORM(out.get(T, 0) + c * s)
    return {k: v for k, v in out.items() if v != 0}


def add(*xs):
    out = {}
    for e in xs:
        for S, c in e.items():
            out[S] = NORM(out.get(S, 0) + c)
    return {k: v for k, v in out.items() if v != 0}


def neg(e):
    return {S: NORM(-c) for S, c in e.items()}


def mul(a, b):
    out = {}
    for S1, c1 in a.items():
        for S2, c2 in b.items():
            T, s = _norm(S1 + S2)
            if s == 0:
                continue
            out[T] = NORM(out.get(T, 0) + s * c1 * c2)
    return {k: v for k, v in out.items() if v != 0}


def simp(e):
    # everything in sight is a rational function of A, B, C, x, so cancel() is both
    # fast and a decision procedure -- simplify() is neither
    out = {}
    for S, c in e.items():
        c = sp.cancel(sp.together(c))
        if c != 0:
            out[S] = c
    return out


def inv(e):
    """inverse of an even invertible element:  e = c0(1 + n),  n nilpotent"""
    c0 = e.get((), 0)
    assert c0 != 0, "not invertible"
    n = {S: NORM(c / c0) for S, c in e.items() if S != ()}
    # (1+n)^{-1} = 1 - n + n^2 - n^3 ... ; only 3 odd generators, so n^4 = 0
    out = el((1, ()))
    term = el((1, ()))
    for k in range(1, 4):
        term = mul(term, n)
        if not term:
            break
        out = add(out, term if k % 2 == 0 else neg(term))
    return {S: NORM(c / c0) for S, c in out.items()}


def eq(a, b):
    d = add(a, neg(b))
    return all(sp.cancel(sp.together(c)) == 0 for c in d.values())


def show(e):
    e = simp(e)
    if not e:
        return "0"
    parts = []
    for S in sorted(e, key=lambda S: (len(S), S)):
        parts.append(f"({e[S]})" + ("" if not S else "*" + "*".join(S)))
    return " + ".join(parts)


# --------------------------------------------------- left partial derivatives
def d_odd(e, g):
    """left derivative d/dg, g an odd generator"""
    out = {}
    for S, c in e.items():
        if g not in S:
            continue
        i = S.index(g)
        sign = (-1) ** i                    # move g to the front
        T = S[:i] + S[i + 1:]
        out[T] = NORM(out.get(T, 0) + sign * c)
    return {k: v for k, v in out.items() if v != 0}


def d_x(e):
    return {S: NORM(sp.diff(c, x)) for S, c in e.items() if sp.diff(c, x) != 0}


# ================================================================== the computation
def run_at_point(Av, Bv, Cv, verbose):
    """all of Theorem 3.5.3 at one specialisation A=Av, B=Bv, C=Cv (x stays symbolic)"""
    al = el((1, ('al',)))
    be = el((1, ('be',)))
    th = el((1, ('th',)))
    one = el((1, ()))

    Ae, Be, Ce = el((Av, ())), el((Bv, ())), el((Cv, ()))
    De = el(((1 + Bv * Cv) / Av, ()), (sp.Rational(1, 1) / Av, ('al', 'be')))  # D=(1+BC+al be)/A
    Ee = add(one, neg(mul(al, be)))                     # E = 1 - al be
    ga = add(mul(al, Be), neg(mul(be, Ae)))             # gamma = al B - be A
    de = add(mul(al, De), neg(mul(be, Ce)))             # delta = al D - be C

    assert eq(add(mul(Ae, De), neg(mul(Be, Ce))), add(one, mul(al, be))), "AD-BC != 1+al be"

    xe = el((x, ()))
    P = add(mul(Ce, xe), De, mul(de, th))               # P = Cx + D + delta th
    W = mul(P, P)
    Pinv = inv(P)
    xt = mul(add(mul(Ae, xe), Be, mul(ga, th)), Pinv)   # tilde x
    tht = mul(add(mul(al, xe), be, mul(Ee, th)), Pinv)  # tilde theta

    R, T, S_, U = d_x(xt), d_odd(xt, 'th'), d_x(tht), d_odd(tht, 'th')
    X, Y, Z, V = mul(W, R), mul(W, T), mul(W, S_), mul(W, U)

    res = {}

    # ---- a candidate set of numerators
    X_paper = add(one, mul(al, be), mul(mul(al, Ee), th))
    Y_paper = add(mul(mul(al, Ee), xe), mul(be, Ee),
                  neg(mul(el((2, ())), mul(mul(ga, de), th))))
    Z_paper = add(mul(al, De), neg(mul(be, Ce)),
                  mul(add(mul(al, de), neg(mul(Ce, Ee))), th))
    V_paper = add(mul(add(mul(Ee, Ce), mul(al, de)), xe), mul(Ee, De), mul(be, de))
    for nm, got, pr in (("X", X, X_paper), ("Y", Y, Y_paper),
                        ("Z", Z, Z_paper), ("V", V, V_paper)):
        res["candidate " + nm + " correct"] = eq(got, pr)

    # ---- the corrected closed forms
    res["X = 1 + al be + al th"] = eq(X, add(one, mul(al, be), mul(al, th)))
    res["V = Cx + D + 2 delta th"] = eq(V, add(mul(Ce, xe), De, mul(el((2, ())), mul(de, th))))
    res["Y = alpha x + beta (theta-free)"] = eq(
        {S: c for S, c in Y.items() if 'th' not in S}, add(mul(al, xe), be))
    res["Z = delta (theta-free)"] = eq({S: c for S, c in Z.items() if 'th' not in S}, de)
    # the candidate V has no theta-term at all, and equals (1-2 al be)(Cx+D):
    res["candidate V = (1-2 al be)(Cx+D)"] = eq(
        V_paper, mul(add(one, neg(mul(el((2, ())), mul(al, be)))), add(mul(Ce, xe), De)))

    # ---- the berezinian, two ways
    ber_formula = mul(add(mul(X, V), neg(mul(Z, Y))), inv(mul(V, V)))     # (XV-ZY)/V^2
    ber_direct = mul(add(R, neg(mul(mul(S_, inv(U)), T))), inv(U))        # (R-SU^-1T)U^-1
    res["ber = (Cx+D+delta th)^-1"] = eq(ber_formula, Pinv)
    res["(XV-ZY)/V^2 = (R-SU^-1T)U^-1"] = eq(ber_formula, ber_direct)
    XVZY = add(mul(X, V), neg(mul(Z, Y)))
    res["XV - ZY = V^2/(Cx+D+delta th)"] = eq(XVZY, mul(mul(V, V), Pinv))
    # the closed form that makes the rewritten proof a one-liner:
    res["XV - ZY = Cx + D + 3 delta th"] = eq(
        XVZY, add(mul(Ce, xe), De, mul(el((3, ())), mul(de, th))))
    res["(Cx+D+3d th)/(Cx+D+2d th)^2 = (Cx+D+d th)^-1"] = eq(
        mul(add(mul(Ce, xe), De, mul(el((3, ())), mul(de, th))),
            inv(mul(add(mul(Ce, xe), De, mul(el((2, ())), mul(de, th))),
                    add(mul(Ce, xe), De, mul(el((2, ())), mul(de, th)))))), Pinv)
    # with the *candidate* V one does not get the right answer:
    ber_paperV = mul(add(mul(X_paper, V_paper), neg(mul(Z_paper, Y_paper))),
                     inv(mul(V_paper, V_paper)))
    res["candidate X,Y,Z,V give the right ber"] = eq(ber_paperV, Pinv)

    # ---- eqs. (81) and (82)
    D_th = lambda e: add(d_odd(e, 'th'), mul(th, d_x(e)))
    res["D_th(tilde th) = (Cx+D+delta th)^-1"] = eq(D_th(tht), Pinv)
    res["superconformality D(x~)-th~D(th~)=0"] = eq(
        add(D_th(xt), neg(mul(tht, D_th(tht)))), {})
    m_G = add(d_x(xt), mul(tht, d_x(tht)))
    res["m_G = (D_th(th~))^2"] = eq(m_G, mul(D_th(tht), D_th(tht)))
    res["m_G = (Cx+D+delta th)^-2"] = eq(m_G, mul(Pinv, Pinv))

    if verbose:
        print("=" * 78)
        print(f"the four numerators at A={Av}, B={Bv}, C={Cv}   (entry = numerator / W)")
        print("=" * 78)
        for nm, val in (("X = W d(x~)/dx  ", X), ("Y = W d(x~)/dth ", Y),
                        ("Z = W d(th~)/dx ", Z), ("V = W d(th~)/dth", V)):
            print(f"  {nm} = {show(val)}")
        print()
        print(f"  gamma = {show(ga)}")
        print(f"  delta = {show(de)}")
        print(f"  candidate V = {show(V_paper)}")
        print(f"  Cx + D + 2 delta theta = "
              f"{show(add(mul(Ce, xe), De, mul(el((2, ())), mul(de, th))))}")
        print()
    return res


# ------------------------------------------------------------------------- driver
random.seed(20260803)
points = [(sp.Rational(3, 2), sp.Rational(-5, 3), sp.Rational(7, 4))]
while len(points) < SAMPLES:
    a, b, c = (sp.Rational(random.randint(-9, 9), random.randint(1, 7)) for _ in range(3))
    if a != 0:
        points.append((a, b, c))

allres = []
for i, (a, b, c) in enumerate(points):
    allres.append(run_at_point(a, b, c, verbose=(i == 0)))

print("=" * 78)
print(f"verdict over {len(points)} independent random specialisations of (A,B,C)")
print("=" * 78)
keys = list(allres[0])
width = max(len(k) for k in keys)
for k in keys:
    vals = [r[k] for r in allres]
    ok = all(vals)
    same = len(set(vals)) == 1
    print(f"  {k:<{width}} : {str(ok):<5}"
          + ("" if same else "   [!! not constant over the points: %r]" % vals))
