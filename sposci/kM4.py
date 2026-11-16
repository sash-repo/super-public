"""
k^M(1|4) (Ramond twist) in the xi-eta-theta-zeta basis of the text,
eq. equation 274, and the cocycle equation 275.

Realisation (as in stringy.py): functions t^a * (monomial in
zeta, xi, eta, theta), zeta being the twisted odd coordinate, with

  D = d_t - (zeta / 2t) d_zeta,
  {f,g} = (2-E)(f) Dg - Df (2-E)(g)
          + sigma (-1)^{p(f)} [ kz (1/t) d_zeta f d_zeta g
                                + k ( d_xi f d_eta g + d_eta f d_xi g
                                      + d_theta f d_theta g ) ].

sigma, k, kz are convention constants fixed below by the Jacobi identity.

Odd labels: zeta = 0, xi = 1, eta = 2, theta = 3.
"""

from fractions import Fraction as Q
from itertools import combinations, product

from stringy import mono, add, mul, smul, d_t, d_odd, parity

Z, XI, ETA, TH = 0, 1, 2, 3
ODD = (0, 1, 2, 3)

def nabla(h):
    out = {}
    for (a, S), c in h.items():
        v = (2 - len(S)) * c
        if v:
            out[(a, S)] = v
    return out


def D(f):
    return add(d_t(f), smul(Q(-1, 2), mul(mono(-1, (Z,)), d_odd(Z, f))))


def bracket(f, g, sigma=1, k=Q(1), kz=Q(1)):
    res = add(mul(nabla(f), D(g)), smul(-1, mul(D(f), nabla(g))))
    s = smul(kz, mul(mono(-1), mul(d_odd(Z, f), d_odd(Z, g))))
    s = add(s, smul(k, add(mul(d_odd(XI, f), d_odd(ETA, g)),
                           mul(d_odd(ETA, f), d_odd(XI, g)),
                           mul(d_odd(TH, f), d_odd(TH, g)))))
    return add(res, smul(sigma * ((-1) ** parity(f)), s))


def monomials(arange):
    out = []
    for a in arange:
        for r in range(5):
            for S in combinations(ODD, r):
                out.append((a, S))
    return out


def weight(key):
    a, S = key
    return 2 * a - 2 + len(S) - (1 if Z in S else 0)


def by_weight(keys):
    d = {}
    for k in keys:
        d.setdefault(weight(k), []).append(k)
    return d


def jacobi_bad(keys, **kw):
    bad = 0
    for kf, kg, kh in product(keys, repeat=3):
        f, g, h = mono(*kf), mono(*kg), mono(*kh)
        lhs = bracket(f, bracket(g, h, **kw), **kw)
        rhs = add(bracket(bracket(f, g, **kw), h, **kw),
                  smul((-1) ** (parity(f) * parity(g)),
                       bracket(g, bracket(f, h, **kw), **kw)))
        if add(lhs, smul(-1, rhs)):
            bad += 1
    return bad


# --------------------------------------------- basis of equation 274 <-> keys
# name_of(key) = (name, index, sign):  K_{t^a M} = sign * <name>_index

def name_of(key):
    a, S = key
    if S == ():
        return ("L", a - 1, 1)
    if S == (XI, ETA):
        return ("Txe", a, 1)
    if S == (XI, TH):
        return ("Txt", a, 1)
    if S == (ETA, TH):
        return ("Tet", a, 1)
    if S == (Z, XI):
        return ("Txz", a, -1)          # T^{xi zeta} = - K_{t^a zeta xi}
    if S == (Z, ETA):
        return ("Tez", a, -1)
    if S == (Z, TH):
        return ("Ttz", a, -1)          # T^{theta zeta}
    if S == (Z, XI, ETA, TH):
        return ("S", a + 1, -1)        # S_n = K_{t^{n-1} xi eta theta zeta}
    if S == (XI,):
        return ("Gx", a - 1, 1)
    if S == (ETA,):
        return ("Ge", a - 1, 1)
    if S == (TH,):
        return ("Gt", a - 1, 1)
    if S == (Z,):
        return ("Gz", a - 1, 1)
    if S == (XI, ETA, TH):
        return ("Ft", a, 1)            # \widetilde F
    if S == (Z, XI, ETA):
        return ("Fxez", a, 1)
    if S == (Z, XI, TH):
        return ("Fxtz", a, 1)
    if S == (Z, ETA, TH):
        return ("Fetz", a, 1)
    raise ValueError(key)


ODD_NAMES = {"Gx", "Ge", "Gt", "Gz", "Ft", "Fxez", "Fxtz", "Fetz"}

def par_of_name(n):
    return 1 if n in ODD_NAMES else 0


# --------------------------------------------------------- the table equation 275

def table_raw(nx, ny):
    (n1, m), (n2, n) = nx, ny
    d0 = (m + n == 0)
    d1 = (m + n + 1 == 0)
    dm1 = (m + n - 1 == 0)
    if n1 == "L" and n2 == "L":
        return Q(m * (m * m - 1)) if d0 else Q(0)
    if (n1, n2) == ("Gx", "Ge"):
        return Q(m * (m + 1)) if d1 else Q(0)
    if (n1, n2) == ("Gz", "Gz"):
        return Q(m * m) - Q(1, 4) if d0 else Q(0)
    if (n1, n2) == ("Gt", "Gt"):
        return Q(m * (m + 1)) if d1 else Q(0)
    if (n1, n2) == ("Txe", "Txe"):
        return Q(m) if d0 else Q(0)
    if (n1, n2) == ("Txz", "Tez"):
        return -(Q(m) - Q(1, 2)) if dm1 else Q(0)
    if (n1, n2) == ("Txt", "Tet"):
        return Q(-m) if d0 else Q(0)
    if (n1, n2) == ("Ttz", "Ttz"):
        return -(Q(m) - Q(1, 2)) if dm1 else Q(0)
    if (n1, n2) == ("Fxez", "Fxez"):
        return Q(1) if d0 else Q(0)
    if (n1, n2) == ("Ft", "Ft"):
        return Q(1) if d1 else Q(0)
    if (n1, n2) == ("Fxtz", "Fetz"):
        return Q(-1) if d0 else Q(0)
    if (n1, n2) == ("S", "S"):
        return -Q(2, 2 * m - 1) if dm1 else Q(0)
    return None


def c_mono(k1, k2, tab=table_raw):
    nx, ny = name_of(k1), name_of(k2)
    sgn = nx[2] * ny[2]
    v = tab(nx[:2], ny[:2])
    if v is not None:
        return sgn * v
    w = tab(ny[:2], nx[:2])
    if w is not None:
        s = -((-1) ** (par_of_name(nx[0]) * par_of_name(ny[0])))
        return sgn * s * w
    return Q(0)


def c_fun(f, g, tab=table_raw):
    out = Q(0)
    for k1, c1 in f.items():
        for k2, c2 in g.items():
            out += c1 * c2 * c_mono(k1, k2, tab)
    return out


def check_skew(keys, tab=table_raw):
    bad = []
    for k1, k2 in product(keys, repeat=2):
        f, g = mono(*k1), mono(*k2)
        if c_mono(k1, k2, tab) != -((-1) ** (parity(f) * parity(g))) * c_mono(k2, k1, tab):
            bad.append((k1, k2))
    return bad


def triples(keys):
    W = by_weight(keys)
    for kf, kg in product(keys, repeat=2):
        for kh in W.get(-(weight(kf) + weight(kg)), ()):
            yield kf, kg, kh


def check_cocycle(keys, tab=table_raw, **kw):
    fails = {}
    for kf, kg, kh in triples(keys):
        f, g, h = mono(*kf), mono(*kg), mono(*kh)
        lhs = c_fun(f, bracket(g, h, **kw), tab)
        rhs = (c_fun(bracket(f, g, **kw), h, tab)
               + (-1) ** (parity(f) * parity(g)) * c_fun(g, bracket(f, h, **kw), tab))
        if lhs != rhs:
            sig = tuple(sorted(name_of(k)[0] for k in (kf, kg, kh)))
            fails.setdefault(sig, []).append((kf, kg, kh, lhs - rhs))
    return fails


if __name__ == "__main__":
    tiny = monomials(range(-1, 2))
    print("--- Jacobi identity, scanning the convention constants ---")
    good = []
    for sigma in (1, -1):
        for k in (Q(1), Q(-1), Q(1, 2), Q(-1, 2), Q(2), Q(-2)):
            for kz in (Q(1), Q(-1), Q(1, 2), Q(-1, 2), Q(2), Q(-2)):
                if jacobi_bad(tiny, sigma=sigma, k=k, kz=kz) == 0:
                    good.append((sigma, k, kz))
    print("  (sigma, k, kz) satisfying Jacobi:", good)

    small = monomials(range(-3, 4))
    print("\nsuper-skew-symmetry of equation 275:",
          "OK" if not check_skew(small) else check_skew(small)[:5])
    for (sigma, k, kz) in good:
        f = check_cocycle(small, sigma=sigma, k=k, kz=kz)
        tot = sum(len(v) for v in f.values())
        print("\n=== sigma=%s k=%s kz=%s : %d failing triples ===" % (sigma, k, kz, tot))
        for sig in sorted(f, key=lambda s: -len(f[s]))[:12]:
            ex = f[sig][0]
            print("   %-24s %5d  e.g. %s %s %s -> %s"
                  % ("/".join(sig), len(f[sig]),
                     name_of(ex[0])[:2], name_of(ex[1])[:2], name_of(ex[2])[:2], ex[3]))
