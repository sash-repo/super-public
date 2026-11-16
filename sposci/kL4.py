"""
k^L(1|4) in the xi-eta basis (the text, eq. equation 272, equation 273).

Realisation: generating functions f(t, xi_1, xi_2, eta_1, eta_2), Laurent in t,
with the contact bracket

  {f,g} = (2-E)(f) d_t g - d_t f (2-E)(g)
          - (-1)^{p(f)} kappa * sum_{i=1,2} (d_{xi_i}f d_{eta_i}g + d_{eta_i}f d_{xi_i}g),

E = the odd Euler operator.  kappa is the normalisation of the odd (Poisson)
part relative to the even one; it depends on how the contact form is written
(alpha = dt + sum_i (xi_i d eta_i + eta_i d xi_i) or with a factor), so we scan
over it instead of guessing.

Odd labels: xi_1 = 1, xi_2 = 2, eta_1 = 3, eta_2 = 4.
"""

from fractions import Fraction as Q
from itertools import combinations, product
import sys

from stringy import mono, add, mul, smul, d_t, d_odd, parity

XI = (1, 2)
ETA = (3, 4)
ODD = (1, 2, 3, 4)

# --------------------------------------------------------------- the bracket

def nabla(h):
    """(2 - E)(h)."""
    out = {}
    for (a, S), c in h.items():
        v = (2 - len(S)) * c
        if v:
            out[(a, S)] = v
    return out


def bracket(f, g, kappa=Q(1)):
    res = add(mul(nabla(f), d_t(g)), smul(-1, mul(d_t(f), nabla(g))))
    s = {}
    for i in (0, 1):
        x, y = XI[i], ETA[i]
        s = add(s, mul(d_odd(x, f), d_odd(y, g)), mul(d_odd(y, f), d_odd(x, g)))
    return add(res, smul(-((-1) ** parity(f)) * kappa, s))


def jacobi_ok(keys, kappa=Q(1)):
    """super Jacobi: [f,[g,h]] = [[f,g],h] + (-1)^{p(f)p(g)}[g,[f,h]]."""
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


# ------------------------------------------------- basis <-> monomial keys
# every basis vector of equation 272 is K_{t^a * M}; we store the monomial in
# the canonical order 1<2<3<4 together with the sign of the reordering.

def name_of(key):
    """(a,S) -> (name, index) as in equation 272, or None."""
    a, S = key
    if S == ():
        return ("L", a - 1)
    if S == (1, 2):
        return ("Txx", a)              # T^{xi_1 xi_2}
    if S == (3, 4):
        return ("Tee", a)              # T^{eta_1 eta_2}
    if S == (1, 3):
        return ("Tx1e1", a)
    if S == (1, 4):
        return ("Tx1e2", a)
    if S == (2, 3):
        return ("Tx2e1", a)
    if S == (2, 4):
        return ("Tx2e2", a)
    if S == (1, 2, 3, 4):
        return ("S", a + 1)            # S_n = K_{t^{n-1} eta1 eta2 xi1 xi2}
    if S == (1,):
        return ("G1", a - 1)
    if S == (2,):
        return ("G2", a - 1)
    if S == (3,):
        return ("Gb1", a - 1)          # \overline G^{eta_1}
    if S == (4,):
        return ("Gb2", a - 1)
    if S == (1, 3, 4):
        return ("F1", a)               # F^1 = K_{t^n eta1 eta2 xi1} = +K_{t^n xi1 eta1 eta2}
    if S == (2, 3, 4):
        return ("F2", a)
    if S == (1, 2, 3):
        return ("Fb1", a)              # \overline F^1 = K_{t^n xi1 xi2 eta1}
    if S == (1, 2, 4):
        return ("Fb2", a)
    raise ValueError(key)


# ------------------------------------------------------- the cocycle (table)
# values are vectors (alpha, beta, gamma) over Q.

ZERO = (Q(0), Q(0), Q(0))

def vec(al=0, be=0, ga=0):
    return (Q(al), Q(be), Q(ga))

def vadd(u, v):
    return tuple(x + y for x, y in zip(u, v))

def vscal(c, u):
    return tuple(Q(c) * x for x in u)


def table_raw(nx, ny):
    """The entries literally as printed in equation 273; nx, ny are
    (name, index) pairs.  Returns None if the pair is not listed."""
    (n1, m), (n2, n) = nx, ny
    d0 = (m + n == 0)
    d1 = (m + n + 1 == 0)
    if n1 == "L" and n2 == "L":
        return vec(al=m * (m * m - 1)) if d0 else ZERO
    if (n1, n2) in (("G1", "Gb1"), ("G2", "Gb2")):
        return vec(al=m * (m + 1)) if d1 else ZERO
    if n1 == "Tx2e2" and n2 == "Tx2e2":
        return vec(al=m) if d0 else ZERO
    if n1 == "Tx1e1" and n2 == "Tx1e1":
        return vec(al=m) if d0 else ZERO
    if n1 == "Tx1e1" and n2 == "Tx2e2":
        return vec(be=-m) if d0 else ZERO
    if (n1, n2) in (("F1", "Fb1"), ("F2", "Fb2")):
        return vec(al=1) if d1 else ZERO
    if n1 == "S" and n2 == "S":
        return vec(al=Q(1, m)) if (d0 and m != 0) else ZERO
    if n1 == "Txx" and n2 == "Tee":
        return vec(al=-m, be=m) if d0 else ZERO
    if n1 == "Tx1e2" and n2 == "Tx2e1":
        return vec(al=m, be=m) if d0 else ZERO
    if n1 == "L" and n2 == "S":
        return vec(be=m, ga=1) if d0 else ZERO
    if n1 == "G1" and n2 == "F2":
        return vec(be=Q(2 * m + 1, 2), ga=Q(1, 2)) if d1 else ZERO
    if n1 == "Gb1" and n2 == "Fb2":
        return vec(be=Q(2 * m + 1, 2), ga=Q(1, 2)) if d1 else ZERO
    if n1 == "G2" and n2 == "F1":
        return vec(be=-Q(2 * m + 1, 2), ga=-Q(1, 2)) if d1 else ZERO
    if n1 == "Gb2" and n2 == "Fb1":
        return vec(be=-Q(2 * m + 1, 2), ga=-Q(1, 2)) if d1 else ZERO
    return None


ODD_NAMES = {"G1", "G2", "Gb1", "Gb2", "F1", "F2", "Fb1", "Fb2"}

def par_of_name(n):
    return 1 if n in ODD_NAMES else 0


def c_mono(k1, k2, tab=table_raw):
    """cocycle on two monomials, extended by super-skew-symmetry
       c(x,y) = -(-1)^{p(x)p(y)} c(y,x)."""
    nx, ny = name_of(k1), name_of(k2)
    v = tab(nx, ny)
    if v is not None:
        return v
    w = tab(ny, nx)
    if w is not None:
        s = -((-1) ** (par_of_name(nx[0]) * par_of_name(ny[0])))
        return vscal(s, w)
    return ZERO


def c_fun(f, g, tab=table_raw):
    """bilinear extension to functions."""
    out = ZERO
    for k1, c1 in f.items():
        for k2, c2 in g.items():
            v = c_mono(k1, k2, tab)
            if v != ZERO:
                out = vadd(out, vscal(c1 * c2, v))
    return out


# ------------------------------------------------------------------- checks

def monomials(arange):
    out = []
    for a in arange:
        for r in range(5):
            for S in combinations(ODD, r):
                out.append((a, S))
    return out


def weight(key):
    """eigenvalue of ad_{K_t} : {t, t^a M} = (2a - 2 + |M|) t^a M."""
    a, S = key
    return 2 * a - 2 + len(S)


def by_weight(keys):
    d = {}
    for k in keys:
        d.setdefault(weight(k), []).append(k)
    return d


def check_skew(keys, tab=table_raw):
    bad = []
    for k1, k2 in product(keys, repeat=2):
        f, g = mono(*k1), mono(*k2)
        lhs = c_mono(k1, k2, tab)
        rhs = vscal(-((-1) ** (parity(f) * parity(g))), c_mono(k2, k1, tab))
        if lhs != rhs:
            bad.append((k1, k2, lhs, rhs))
    return bad


def triples(keys):
    """only triples of total weight 0 can give a nontrivial equation."""
    W = by_weight(keys)
    for kf, kg in product(keys, repeat=2):
        for kh in W.get(-(weight(kf) + weight(kg)), ()):
            yield kf, kg, kh


def check_cocycle(keys, kappa=Q(1), tab=table_raw, drop_S0=True, verbose=0):
    """c(f,[g,h]) = c([f,g],h) + (-1)^{p(f)p(g)} c(g,[f,h])"""
    fails = {}
    for kf, kg, kh in triples(keys):
        if drop_S0 and any(name_of(k) == ("S", 0) for k in (kf, kg, kh)):
            continue
        f, g, h = mono(*kf), mono(*kg), mono(*kh)
        lhs = c_fun(f, bracket(g, h, kappa), tab)
        rhs = vadd(c_fun(bracket(f, g, kappa), h, tab),
                   vscal((-1) ** (parity(f) * parity(g)),
                         c_fun(g, bracket(f, h, kappa), tab)))
        d = tuple(x - y for x, y in zip(lhs, rhs))
        if d != ZERO:
            sig = tuple(sorted(name_of(k)[0] for k in (kf, kg, kh)))
            fails.setdefault(sig, []).append((kf, kg, kh, d))
    return fails


def report(fails, keys_n):
    tot = sum(len(v) for v in fails.values())
    print("   failing triples: %d (out of triples over %d monomials)" % (tot, keys_n))
    for sig in sorted(fails, key=lambda s: -len(fails[s])):
        ex = fails[sig][0]
        print("     %-28s %5d   e.g. %s %s %s  ->  defect %s"
              % ("/".join(sig), len(fails[sig]),
                 name_of(ex[0]), name_of(ex[1]), name_of(ex[2]), ex[3]))


if __name__ == "__main__":
    small = monomials(range(-2, 3))
    print("Jacobi identity for the xi-eta contact bracket (kappa=1):",
          "OK" if jacobi_ok(monomials(range(-1, 2))) == 0 else "FAILS")
    bad = check_skew(small)
    print("super-skew-symmetry of the table:", "OK" if not bad else "%d violations" % len(bad))
    for kappa in (Q(1), Q(-1), Q(1, 2), Q(-1, 2), Q(2), Q(-2)):
        print("\n=== kappa = %s ===" % kappa)
        fails = check_cocycle(small, kappa)
        report(fails, len(small))
