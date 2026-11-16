"""
k^L(1|4) in the theta-basis: the table (70) of the text.

Realisation: generating functions f(t, theta_1..theta_4), Laurent in t, with

  {f,g} = (2-E)(f) d_t g - d_t f (2-E)(g)
          - (-1)^{p(f)} kappa * sum_{1<=i<=4} d_{theta_i}f d_{theta_i}g,

E = the odd Euler operator.  kappa is the normalisation of the odd part of the
contact bracket relative to the even one; it is a convention (it depends on how
the contact form is written), so we scan over it rather than guess.

Basis of (monBas)/(70):
  L_n = K_{t^{n+1}}                          T^{ij}_n = K_{t^n th_i th_j}
  S_n = K_{t^{n-1} th_1 th_2 th_3 th_4}      G^i_n = K_{t^{n+1} th_i}
  F^i_n = K_{t^n d_{th_i}(th_1 th_2 th_3 th_4)}
"""

from fractions import Fraction as Q
from itertools import combinations, permutations, product

from stringy import mono, add, mul, smul, d_t, d_odd, parity

ODD = (1, 2, 3, 4)

# --------------------------------------------------------------- the bracket

def nabla(h):
    """(2 - E)(h)."""
    return {k: v for k, v in
            (((a, S), (2 - len(S)) * c) for (a, S), c in h.items()) if v}


def bracket(f, g, kappa=Q(1)):
    res = add(mul(nabla(f), d_t(g)), smul(-1, mul(d_t(f), nabla(g))))
    s = {}
    for i in ODD:
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    return add(res, smul(-((-1) ** parity(f)) * kappa, s))


def jacobi_ok(keys, kappa=Q(1)):
    bad = 0
    for kf, kg, kh in product(keys, repeat=3):
        f, g, h = mono(*kf), mono(*kg), mono(*kh)
        lhs = bracket(f, bracket(g, h, kappa), kappa)
        rhs = add(bracket(bracket(f, g, kappa), h, kappa),
                  smul((-1) ** (parity(f) * parity(g)),
                       bracket(g, bracket(f, h, kappa), kappa)))
        if add(lhs, smul(-1, rhs)):
            bad += 1
    return bad


# --------------------------------------------------- basis names <-> keys
# F^i_n = K_{t^n d_{th_i}(th_1234)} = (-1)^{i-1} K_{t^n th_{complement of i}}

def name_of(key):
    """(a,S) -> (name, index, sign) : the monomial equals sign * (named vector)."""
    a, S = key
    if S == ():
        return ("L", a - 1, 1)
    if len(S) == 2:
        return ("T%d%d" % S, a, 1)
    if S == ODD:
        return ("S", a + 1, 1)
    if len(S) == 1:
        return ("G%d" % S[0], a - 1, 1)
    if len(S) == 3:
        i = [j for j in ODD if j not in S][0]
        return ("F%d" % i, a, (-1) ** (i - 1))
    raise ValueError(key)


ODD_NAMES = lambda n: n[0][0] in "GF"   # n = (name, index); G's and F's are odd

ZERO = (Q(0), Q(0), Q(0))

def vec(al=0, be=0, ga=0):
    return (Q(al), Q(be), Q(ga))

def vadd(u, v):
    return tuple(x + y for x, y in zip(u, v))

def vscal(c, u):
    return tuple(Q(c) * x for x in u)


def even_perm(p):
    """is the permutation p of (1,2,3,4) even?"""
    s = 0
    p = list(p)
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s += 1
    return s % 2 == 0


# ------------------------------------------------------- the printed table

def table_printed(nx, ny):
    """(70) of the text, literally as printed."""
    (n1, m), (n2, n) = nx, ny
    d0 = (m + n == 0)
    d1 = (m + n + 1 == 0)
    if n1 == "L" and n2 == "L":
        return vec(al=m * (m * m - 1)) if d0 else ZERO
    if n1 == n2 and n1.startswith("G"):
        return vec(al=m * (m + 1)) if d1 else ZERO
    if n1 == n2 and n1.startswith("T"):
        return vec(al=-m) if d0 else ZERO
    if n1 == n2 and n1.startswith("F"):
        return vec(al=-1) if d1 else ZERO
    if n1 == "S" and n2 == "S":
        return vec(al=Q(1, m)) if (d0 and m != 0) else ZERO
    if n1 == "L" and n2 == "S":
        return vec(be=m, ga=1) if d0 else ZERO
    if n1[0] == "G" and n2[0] == "F" and n1[1] == n2[1]:
        return vec(be=2 * m + 1, ga=2) if d1 else ZERO
    if n1.startswith("T") and n2.startswith("T") and n1 != n2:
        i, j = int(n1[1]), int(n1[2])
        k, l = int(n2[1]), int(n2[2])
        if {i, j, k, l} == {1, 2, 3, 4}:
            sg = 1 if even_perm((i, j, k, l)) else -1
            return vec(be=-sg * m) if d0 else ZERO
        return ZERO
    return None


def table_kvdl(nx, ny):
    """(formulasKvdL) of the text on spinor-oscillator representations - our machine-checked version."""
    (n1, m), (n2, n) = nx, ny
    d0 = (m + n == 0)
    d1 = (m + n + 1 == 0)
    if n1 == "L" and n2 == "L":
        return vec(al=m * (m * m - 1)) if d0 else ZERO
    if n1 == n2 and n1.startswith("G"):
        return vec(al=-m * (m + 1)) if d1 else ZERO
    if n1 == n2 and n1.startswith("T"):
        return vec(al=-m) if d0 else ZERO
    if n1 == n2 and n1.startswith("F"):
        return vec(al=1) if d1 else ZERO
    if n1 == "S" and n2 == "S":
        return vec(al=Q(1, m)) if (d0 and m != 0) else ZERO
    if n1 == "L" and n2 == "S":
        return vec(be=m, ga=1) if d0 else ZERO
    if n1[0] == "G" and n2[0] == "F" and n1[1] == n2[1]:
        return vec(be=Q(2 * m + 1, 2), ga=Q(1, 2)) if d1 else ZERO
    if n1.startswith("T") and n2.startswith("T") and n1 != n2:
        i, j = int(n1[1]), int(n1[2])
        k, l = int(n2[1]), int(n2[2])
        if {i, j, k, l} == {1, 2, 3, 4}:
            sg = 1 if even_perm((i, j, k, l)) else -1
            return vec(be=sg * m) if d0 else ZERO
        return ZERO
    return None


# ------------------------------------------------ extension to all of g

def c_mono(k1, k2, tab):
    nx, ny = name_of(k1), name_of(k2)
    s = nx[2] * ny[2]
    ax, ay = (nx[0], nx[1]), (ny[0], ny[1])
    v = tab(ax, ay)
    if v is None:
        w = tab(ay, ax)
        if w is None:
            return ZERO
        sg = -((-1) ** (int(ODD_NAMES(ax)) * int(ODD_NAMES(ay))))
        v = vscal(sg, w)
    return vscal(s, v)


def c_fun(f, g, tab):
    out = ZERO
    for k1, c1 in f.items():
        for k2, c2 in g.items():
            v = c_mono(k1, k2, tab)
            if v != ZERO:
                out = vadd(out, vscal(c1 * c2, v))
    return out


def monomials(arange):
    return [(a, S) for a in arange for r in range(5)
            for S in combinations(ODD, r)]


def weight(key):
    a, S = key
    return 2 * a - 2 + len(S)


def check_skew(keys, tab):
    bad = []
    for k1, k2 in product(keys, repeat=2):
        f, g = mono(*k1), mono(*k2)
        lhs = c_mono(k1, k2, tab)
        rhs = vscal(-((-1) ** (parity(f) * parity(g))), c_mono(k2, k1, tab))
        if lhs != rhs:
            bad.append((k1, k2, lhs, rhs))
    return bad


def triples(keys):
    W = {}
    for k in keys:
        W.setdefault(weight(k), []).append(k)
    for kf, kg in product(keys, repeat=2):
        for kh in W.get(-(weight(kf) + weight(kg)), ()):
            yield kf, kg, kh


def check_cocycle(keys, tab, kappa=Q(1), drop_S0=True):
    fails = {}
    for kf, kg, kh in triples(keys):
        if drop_S0 and any(name_of(k)[:2] == ("S", 0) for k in (kf, kg, kh)):
            continue
        f, g, h = mono(*kf), mono(*kg), mono(*kh)
        lhs = c_fun(f, bracket(g, h, kappa), tab)
        rhs = vadd(c_fun(bracket(f, g, kappa), h, tab),
                   vscal((-1) ** (parity(f) * parity(g)),
                         c_fun(g, bracket(f, h, kappa), tab)))
        d = tuple(x - y for x, y in zip(lhs, rhs))
        if d != ZERO:
            sig = tuple(sorted(name_of(k)[0][0] for k in (kf, kg, kh)))
            fails.setdefault(sig, []).append((kf, kg, kh, d))
    return fails


def report(fails, n):
    tot = sum(len(v) for v in fails.values())
    print("   failing triples: %d (monomials in window: %d)" % (tot, n))
    for sig in sorted(fails, key=lambda s: -len(fails[s])):
        ex = fails[sig][0]
        print("     %-10s %5d   e.g. %s %s %s -> defect %s"
              % ("".join(sig), len(fails[sig]),
                 name_of(ex[0])[:2], name_of(ex[1])[:2], name_of(ex[2])[:2], ex[3]))


if __name__ == "__main__":
    small = monomials(range(-2, 3))
    print("Jacobi for the theta contact bracket (kappa=1):",
          "OK" if jacobi_ok(monomials(range(-1, 2))) == 0 else "FAILS")
    for nm, tab in (("(70) as printed in aShdet", table_printed),
                    ("(formulasKvdL) of the text on spinor-oscillator representations", table_kvdl)):
        print("\n===== %s =====" % nm)
        bad = check_skew(small, tab)
        print("   super-skew-symmetry:", "OK" if not bad else "%d violations" % len(bad))
        for kappa in (Q(1), Q(-1)):
            print("   -- kappa = %s" % kappa)
            report(check_cocycle(small, tab, kappa), len(small))
