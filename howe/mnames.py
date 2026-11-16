"""mnames.py -- the constants of eqs. (44) and (84) in the naming M_f, i.e. with the
elements of m(n)_{-1} called after their own generating functions rather than after the
coordinates they are the hats of.  Answers, for n = 1,2,3,4:

    M_1 as a vector field                 (is it d/dtau, or twice it?)
    [M_{x_i}, M_{xi_j}]                   (the constant in front of M_1)
    [M_{x_i}, M_{x_j}], [M_{xi_i}, M_{xi_j}]
    parities and M-degrees
    M_1 central in m(n)_{<0}, M_1^2 = 0
    the identification U(m(n)_{<0}) = Lambda(M_x) (x) C[M_xi] (x) Lambda(M_1)

Everything exact, over Q, on top of mn.py .
Run:  python3 mnames.py
"""
from fractions import Fraction as F

from mn import mono, one, tau, mbr, Mvf, mdeg, par_m, monos
from le import padd, pscal


def x(n, i):
    return mono(n, [1 if k == i else 0 for k in range(n)], [])


def xi(n, i):
    return mono(n, [0] * n, [i])


def pr(f, n):
    """print a generating function"""
    if not f:
        return "0"
    out = []
    for (a, S), c in sorted(f.items()):
        s = ""
        for i, e in enumerate(a):
            s += "x%d" % (i + 1) + ("^%d" % e if e > 1 else "") if e else ""
        for i in S:
            s += "tau" if i == n else "xi%d" % (i + 1)
        out.append(("%s" % c if s == "" else ("%s*%s" % (c, s) if c != 1 else s)))
    return " + ".join(out)


def vf(f, n):
    """print the vector field 'f"""
    V = Mvf(f, n)
    names = ["d/dx%d" % (i + 1) for i in range(n)] + \
            ["d/dxi%d" % (i + 1) for i in range(n)] + ["d/dtau"]
    out = []
    for g, c in sorted(V.comp.items()):
        if c:
            out.append("(%s)%s" % (pr(c, n), names[g]))
    return " + ".join(out) if out else "0"


ok = 0
bad = 0


def check(claim, cond):
    global ok, bad
    print(("  OK   " if cond else "  FAIL ") + claim)
    if cond:
        ok += 1
    else:
        bad += 1


for n in range(1, 5):
    print("=" * 72)
    print("n =", n)
    o = one(n)

    # 1) M_1 as a vector field
    print("  M_1 = '1 =", vf(o, n), " (parity %d, M-degree %s)" % (par_m(o), mdeg(o, n)))
    V1 = Mvf(o, n)
    check("M_1 = 2 d/dtau", V1.comp.get(2 * n) == {((0,) * n, ()): F(2)}
          and all(not V1.comp.get(g) for g in range(2 * n)))
    check("M_1 is odd", par_m(o) == 1)
    check("deg M_1 = -2", mdeg(o, n) == -2)
    check("[M_1, M_1] = 0", mbr(o, o, n) == {})

    for i in range(n):
        Xi, Ei = x(n, i), xi(n, i)
        if i == 0:
            print("  M_x1  =", vf(Xi, n), " (parity %d, M-degree %s)" % (par_m(Xi), mdeg(Xi, n)))
            print("  M_xi1 =", vf(Ei, n), " (parity %d, M-degree %s)" % (par_m(Ei), mdeg(Ei, n)))
        check("M_x%d odd, M_xi%d even, both of degree -1" % (i + 1, i + 1),
              par_m(Xi) == 1 and par_m(Ei) == 0 and mdeg(Xi, n) == -1 and mdeg(Ei, n) == -1)

    # 2) the brackets in m(n)_{-1}
    for i in range(n):
        for j in range(n):
            b = mbr(x(n, i), xi(n, j), n)
            want = o if i == j else {}
            check("[M_x%d, M_xi%d] = %s M_1" % (i + 1, j + 1, "+" if i == j else "0"),
                  b == want)
    for i in range(n):
        for j in range(n):
            check("[M_x%d, M_x%d] = 0" % (i + 1, j + 1), mbr(x(n, i), x(n, j), n) == {})
            check("[M_xi%d, M_xi%d] = 0" % (i + 1, j + 1), mbr(xi(n, i), xi(n, j), n) == {})

    # 3) M_1 central in m(n)_{<0} and in m(n)_0
    cen = True
    for f in [o] + [x(n, i) for i in range(n)] + [xi(n, i) for i in range(n)]:
        cen = cen and mbr(o, f, n) == {}
    check("M_1 central in m(n)_{<0}", cen)
    cen0 = all(mbr(o, f, n) == {} for f in monos(n, 2, 0))
    check("M_1 central in the pe(n)-part of m(n)_0, but not against M_tau",
          cen0 and mbr(o, tau(n), n) != {})

    # 4) the hatted naming: defining hat xi_i as the field of
    #    -xi_i and printed [hat x_i, hat xi_j] = 2 delta_ij D.  With that definition the
    #    bracket is -2 delta_ij D: the prose and the equation disagreed.
    for i in range(n):
        b = mbr(x(n, i), pscal(xi(n, i), F(-1)), n)
        check("[M_x%d, M_{-xi%d}] = -M_1, not +M_1" % (i + 1, i + 1),
              b == pscal(o, F(-1)))

# ------------------------------------------------------------------------------
# 5) the second-order singular vector of Theorem 9.5.1, in the naming M_f.
#    The engine works in Kochetkov's PBW order 'partial^a 'delta^S D^t, i.e. in
#    (-M_xi)^a (M_x)^S (M_1/2)^t; the Theorem prints the vector in the order
#    M_{x_i} M_{xi_i}, and the two differ by [M_{x_i},M_{xi_i}] = M_1 != 0.
print("=" * 72)
print("the family 2a) vector as the Theorem prints it:")
print("   y = sum_i M_{x_i} M_{xi_i} (x) v + M_1 (x) w,   w = (lam-n)/2 v")

from mimod import Mn, MInduced  # noqa: E402


def _scal(e, c):
    return {k: c * v for k, v in e.items() if c * v}


def _add(*es):
    out = {}
    for e in es:
        for k, v in e.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


for n in (2, 3, 4):
    for lam in (-n - 1, -n, -1, 0, 1, n, n + 1):
        mn = Mn(n)
        I = MInduced(mn, [1] * n, lam + 2)
        sv = I.singular(2, tuple(F(1) for _ in range(n)))
        base = {(((0,) * n, (), 0), tuple(I.lam), 0): F(1)}
        S = {}
        for i in range(n):
            S = _add(S, I.lmul(n + i, _scal(I.lmul(i, base), F(-1))))
        M1 = _scal(I.lmul(2 * n, base), F(2))
        cand = _add(S, _scal(M1, F(lam - n, 2)))
        same = False
        if len(sv) == 1 and cand:
            s = sv[0]
            common_keys = set(cand) & set(s)
            if common_keys:
                k = next(iter(common_keys))
                r = s[k] / cand[k]
                same = (_scal(cand, r) == s)
        check("n=%d, lam=%3d: the printed vector spans the singular space" % (n, lam), same)
        if lam == n:
            check("n=%d: at lam=n the M_1-part is absent" % n,
                  F(lam - n, 2) == 0)

print("=" * 72)
print("%d checks, %d failed" % (ok + bad, bad))
