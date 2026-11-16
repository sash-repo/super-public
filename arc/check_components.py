"""
Component form of the cocycles of Table (61), compared with the printed
component formulas (48) for k^L and (50) for k^M.

For every algebra we
  * fix the sign convention (with or without the factor (-1)^{p(f)}) by
    REQUIRING the cocycle identity  c(f,{g,h}) = c({f,g},h) + (-1)^{p(f)p(g)} c(g,{f,h});
  * print the resulting values on the basis pairs of [KvdL], next to the values
    printed in the text.
"""

from fractions import Fraction as F
from stringy import mono, add, mul, smul, d_t, d_odd, parity, show
from check_cocycles import (L_bracket, L_res, L_sn, M_bracket, M_res, M_sn,
                            monomials)
from check_table import cocycle_test, L4_ops

RANGE = range(-3, 4)

def find_sign(n, bracket, cbase, labels, excl=None):
    """returns list of sign conventions ('plain'/'sign') for which c is a cocycle"""
    keys = [k for k in monomials(n, range(-2, 3), labels) if not (excl and excl(k))]
    good = []
    for tag, use in (("plain", False), ("sign", True)):
        c = (lambda f, g: cbase(f, g) * ((-1) ** parity(f) if use else 1))
        bad = cocycle_test(keys, bracket, c)
        if not bad:
            good.append(tag)
    return good

# ------------------------------------------------------------------ k^L(1|n)

def L_report(n):
    print("=" * 72)
    print("k^L(1|%d)" % n)
    sn = L_sn(n)
    base = lambda f, g: L_res(n, mul(f, sn(g)))
    labels = tuple(range(1, n + 1))
    good = find_sign(n, lambda f, g: L_bracket(n, f, g), base, labels)
    print("  Table (61) operator is a cocycle with convention(s): %s" % good)
    use = ("sign" in good)
    c = lambda f, g: base(f, g) * ((-1) ** parity(f) if use else 1)
    # component values
    print("  values (m ranges over %s):" % list(RANGE))
    rows = []
    rows.append(("c(L_m,L_n), m+n=0        ", "m(m^2-1)",
                 [(m, c(mono(m + 1), mono(-m + 1))) for m in RANGE]))
    if n >= 1:
        rows.append(("c(E^1_m,E^1_n), m+n+1=0  ", "m(m+1)",
                     [(m, c(mono(m + 1, (1,)), mono(-m, (1,)))) for m in RANGE]))
    if n >= 2:
        rows.append(("c(T^12_m,T^12_n), m+n=0  ", "-m",
                     [(m, c(mono(m, (1, 2)), mono(-m, (1, 2)))) for m in RANGE]))
    if n >= 3:
        rows.append(("c(t^m th1th2th3, ...)    ", "-1 (m+n+1=0)",
                     [(m, c(mono(m, (1, 2, 3)), mono(-m - 1, (1, 2, 3)))) for m in RANGE]))
    for name, printed, vals in rows:
        print("   %s printed: %-14s computed: %s" % (name, printed,
              ", ".join("m=%d:%s" % (m, v) for m, v in vals)))

# ---------------------------------------------------------------- k^L'(1|4)

def L4_report():
    print("=" * 72)
    print("k^L'(1|4): the three cocycles of Table (61)")
    Dth, Dinv = L4_ops()
    def c1b(f, g):
        h = g
        for i in (4, 3, 2, 1):
            h = Dth(i)(h)
        return L_res(4, mul(f, Dinv(h)))
    def c2b(f, g):
        h = {(a - 1, S): (2 - len(S)) * cc for (a, S), cc in g.items() if (2 - len(S))}
        return L_res(4, mul(f, h))
    def c3b(f, g):
        return L_res(4, mul(f, d_t(g)))
    excl = lambda k: (k[0] == -1 and len(k[1]) == 4)
    for nm, cb in (("(1)", c1b), ("(2)", c2b), ("(3)", c3b)):
        good = find_sign(4, lambda f, g: L_bracket(4, f, g), cb, (1, 2, 3, 4), excl)
        print("  cocycle %s : valid convention(s) %s" % (nm, good))
        if not good:
            continue
        use = ("sign" in good)
        c = lambda f, g: cb(f, g) * ((-1) ** parity(f) if use else 1)
        print("     c(L_m,L_-m)      [printed alpha: m(m^2-1)] :",
              ", ".join("%d:%s" % (m, c(mono(m + 1), mono(-m + 1))) for m in RANGE))
        print("     c(E^1_m,E^1_-m-1)[printed alpha: m(m+1)]   :",
              ", ".join("%d:%s" % (m, c(mono(m + 1, (1,)), mono(-m, (1,)))) for m in RANGE))
        print("     c(T^12_m,T^12_-m)[printed alpha: -m]       :",
              ", ".join("%d:%s" % (m, c(mono(m, (1, 2)), mono(-m, (1, 2)))) for m in RANGE))
        print("     c(F^1_m,F^1_-m-1)[printed alpha: -1]       :",
              ", ".join("%d:%s" % (m, c(mono(m, (2, 3, 4)), mono(-m - 1, (2, 3, 4)))) for m in RANGE))
        print("     c(S_m,S_-m)      [printed alpha: 1/m]      :",
              ", ".join("%d:%s" % (m, c(mono(m - 1, (1, 2, 3, 4)), mono(-m - 1, (1, 2, 3, 4))))
                        for m in RANGE if m != 0))
        print("     c(L_m,S_-m)      [printed: gamma+beta m]   :",
              ", ".join("%d:%s" % (m, c(mono(m + 1), mono(-m - 1, (1, 2, 3, 4)))) for m in RANGE))
        print("     c(E^1_m,F^1_-m-1)[printed: g/2+b(m+1/2)]   :",
              ", ".join("%d:%s" % (m, c(mono(m + 1, (1,)), mono(-m - 1, (2, 3, 4)))) for m in RANGE))
        print("     c(T^12_m,T^34_-m)[printed: -beta m]        :",
              ", ".join("%d:%s" % (m, c(mono(m, (1, 2)), mono(-m, (3, 4)))) for m in RANGE))

# ------------------------------------------------------------------ k^M(1|n)

def M_report(n):
    print("=" * 72)
    print("k^M(1|%d)" % n)
    sn = M_sn(n)
    base = lambda f, g: M_res(n, mul(f, sn(g)))
    labels = tuple(range(0, n))
    good = find_sign(n, lambda f, g: M_bracket(n, f, g), base, labels)
    print("  Table (61) operator is a cocycle with convention(s): %s" % good)
    use = ("sign" in good)
    c = lambda f, g: base(f, g) * ((-1) ** parity(f) if use else 1)
    rows = [("c(t^{m+1},t^{n+1}), m+n=0      ", "m(m^2-1)",
             [(m, c(mono(m + 1), mono(-m + 1))) for m in RANGE])]
    rows.append(("c(t^{m+1}z,t^{n+1}z), m+n=0    ", "m^2-1/4",
                 [(m, c(mono(m + 1, (0,)), mono(-m + 1, (0,)))) for m in RANGE]))
    if n >= 2:
        rows.append(("c(t^{m+1}th,t^{n+1}th),m+n+1=0", "m(m+1)",
                     [(m, c(mono(m + 1, (1,)), mono(-m, (1,)))) for m in RANGE]))
        rows.append(("c(t^m th z,t^n th z), m+n-1=0 ", "-(m-1/2)",
                     [(m, c(mono(m, (1, 0)), mono(1 - m, (1, 0)))) for m in RANGE]))
    if n >= 3:
        rows.append(("c(t^m th1th2,...), m+n=0      ", "-m",
                     [(m, c(mono(m, (1, 2)), mono(-m, (1, 2)))) for m in RANGE]))
        rows.append(("c(t^m th1th2 z,...), m+n=0    ", "-1",
                     [(m, c(mono(m, (1, 2, 0)), mono(-m, (1, 2, 0)))) for m in RANGE]))
    if n >= 4:
        rows.append(("c(t^m th1th2th3,..), m+n+1=0  ", "-1",
                     [(m, c(mono(m, (1, 2, 3)), mono(-m - 1, (1, 2, 3)))) for m in RANGE]))
        rows.append(("c(t^{m-1}th1th2th3 z,..)      ", "1/(m-1/2)",
                     [(m, c(mono(m - 1, (1, 2, 3, 0)), mono(-m, (1, 2, 3, 0)))) for m in RANGE]))
    for name, printed, vals in rows:
        print("   %s printed: %-11s computed: %s" % (name, printed,
              ", ".join("m=%d:%s" % (m, v) for m, v in vals)))

if __name__ == "__main__":
    for n in (0, 1, 2, 3):
        L_report(n)
    L4_report()
    for n in (1, 2, 3, 4):
        M_report(n)
