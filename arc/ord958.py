"""
The order of the factors in the operators cD_n of the k^L column.

Subsection 9.3 of the text writes

    cD_n := del_t^{2-n} D_{th_1}...D_{th_n},        del_t cD_n = the operator
                                                    under Res in Table (61),

with the power of del_t LEFTMOST.  The opposite order, read at n=4, is

    D_{th_1}D_{th_2}D_{th_3}D_{th_4} (D_1)^{-1},

in which (D_1)^{-1} acts first, and it is undefined on every t^{-1}phi(theta).
The accepted operator, eq. (57)/(58), has (D_1)^{-1} LEFTMOST, and the uniform
formula is written in the same order.

Checked here, exactly, over Q:

  1) [del_t, D_{th_i}] = 0 on every monomial t^a th_S -- so in the k^L case the
     two orders are ONE operator, for n <= 3, and nothing moves when the rows of
     Table (61) and the definition of cD_n are turned round;
  2) the two orders agree as operators, n = 0,...,3 (an independent check of 1
     on the composites actually printed);
  3) the cocycle identity holds for the REVERSED order, n = 0,...,3 (so the
     printed change is safe end-to-end, not merely at the level of symbols);
  4) at n = 4 the two orders are NOT one operator:
       (a) (D_1)^{-1} leftmost is defined on all of k^L'(1|4) after the
           simplification of eq. (58) -- and is a cocycle;
       (b) (D_1)^{-1} rightmost is undefined on the monomials t^{-1}phi(theta)
           that DO belong to k^L'(1|4);
     so at n = 4 only the leftmost order exists, and it is Delta_IS;
  5) the weight of cD_n is the sum of the weights of its factors, hence
     independent of their order: cF_{-1/2} --> cF_{(3-n)/2}.

Run:  python3 ord958.py
"""

from fractions import Fraction as F
from itertools import combinations

from stringy import mono, add, mul, smul, d_t, d_odd, parity, eq, show
from check_cocycles import (L_ops, L_bracket, L_res, L_sn, monomials,
                            check_cocycle_identity)
from check_bases import Delta_IS, Delta_IS_literal, d_t_inv

FAIL = 0
NCHECK = 0


def report(name, ok, detail=""):
    global FAIL, NCHECK
    NCHECK += 1
    if not ok:
        FAIL += 1
    print(("  OK   " if ok else "  FAIL ") + name + (("   " + detail) if detail else ""))


def compose(*ops):
    """ops as WRITTEN, left to right; the rightmost acts first."""
    def h(f):
        for op in reversed(ops):
            f = op(f)
        return f
    return h


def keys_of(n, arange=range(-4, 5)):
    return monomials(n, arange, tuple(range(1, n + 1)))


# ------------------------------------------------- 1) del_t commutes with D_th

print("1) [del_t, D_{th_i}] = 0")
for n in range(1, 5):
    D1, Dth, Kth = L_ops(n)
    ks = keys_of(n)
    for i in range(1, n + 1):
        bad = [k for k in ks
               if not eq(d_t(Dth(i)(mono(*k))), Dth(i)(d_t(mono(*k))))]
        report("n=%d, i=%d on %d monomials" % (n, i, len(ks)), not bad,
               "" if not bad else str(bad[:3]))
    # and with K_th, the other square root, for completeness
    bad = [k for k in ks
           if not eq(d_t(Kth(1)(mono(*k))), Kth(1)(d_t(mono(*k))))]
    report("n=%d, the same for K_{th_1}" % n, not bad)

# --------------------------------------- 2) the two orders are one operator

print("\n2) D_{th_1}...D_{th_n} del_t^{3-n}  =  del_t^{3-n} D_{th_1}...D_{th_n}")


def sn_reversed(n):
    """del_t^{3-n} D_{th_1}...D_{th_n}, n <= 3."""
    D1, Dth, Kth = L_ops(n)
    ops = [D1] * (3 - n) + [Dth(i) for i in range(1, n + 1)]
    return compose(*ops)


def cDn_printed(n):
    """D_{th_1}...D_{th_n} del_t^{2-n}, n <= 2 (a differential operator)."""
    D1, Dth, Kth = L_ops(n)
    return compose(*([Dth(i) for i in range(1, n + 1)] + [D1] * (2 - n)))


def cDn_reversed(n):
    D1, Dth, Kth = L_ops(n)
    return compose(*([D1] * (2 - n) + [Dth(i) for i in range(1, n + 1)]))


for n in range(0, 4):
    ks = keys_of(n)
    sp, sr = L_sn(n), sn_reversed(n)
    bad = [k for k in ks if not eq(sp(mono(*k)), sr(mono(*k)))]
    report("s_%d, the two orders, on %d monomials" % (n, len(ks)), not bad,
           "" if not bad else str(bad[:3]))
for n in range(0, 3):
    ks = keys_of(n)
    bad = [k for k in ks if not eq(cDn_printed(n)(mono(*k)), cDn_reversed(n)(mono(*k)))]
    report("cD_%d, the two orders, on %d monomials" % (n, len(ks)), not bad)

# the same at the level of the cocycle VALUE, which is what the table prints
print("\n   and hence Res(f s_n(g)) is unchanged:")
for n in range(0, 4):
    ks = keys_of(n, range(-3, 4))
    sp, sr = L_sn(n), sn_reversed(n)
    bad = 0
    for kf in ks:
        for kg in ks:
            f, g = mono(*kf), mono(*kg)
            if L_res(n, mul(f, sp(g))) != L_res(n, mul(f, sr(g))):
                bad += 1
    report("n=%d, %d pairs" % (n, len(ks) ** 2), bad == 0)

# ------------------------------------------- 3) the reversed order is a cocycle

print("\n3) the cocycle identity for the REVERSED order")
for n in range(0, 4):
    small = keys_of(n, range(-2, 3))
    sr = sn_reversed(n)
    sign = n in (2,)          # the sign factor printed in Table (61)

    def c(f, g, sr=sr, n=n, sign=sign):
        v = L_res(n, mul(f, sr(g)))
        return v * (-1) ** parity(f) if sign else v
    bad = check_cocycle_identity(n, lambda f, g: L_bracket(n, f, g), c, small)
    report("n=%d on %d^3 triples" % (n, len(small)), bad == 0,
           "" if bad == 0 else "%d failures" % bad)

# ------------------------------------------------------- 4) n = 4: one order only

print("\n4) n = 4: the two orders are NOT one operator")
D1_4, Dth4, Kth4 = L_ops(4)
P = compose(*[Dth4(i) for i in range(1, 5)])      # D_th1 D_th2 D_th3 D_th4

ks4 = keys_of(4)
excluded = (-1, (1, 2, 3, 4))                     # thrown away in k^L'(1|4)
ks4p = [k for k in ks4 if k != excluded]

# (a) the accepted order: (D_1)^{-1} leftmost, simplified before evaluation
ok, undef = True, []
for k in ks4p:
    try:
        Delta_IS(mono(*k))
    except Exception:
        ok = False
        undef.append(k)
report("(a) Delta_IS = (D_1)^{-1} D_th1..D_th4 defined on all of k^L'(1|4)"
       " (%d monomials)" % len(ks4p), ok, str(undef[:3]))

# (b) the rejected order: (D_1)^{-1} rightmost, i.e. applied FIRST
undef_right = []
for k in ks4:
    try:
        d_t_inv(mono(*k))
    except ValueError:
        undef_right.append(k)
report("(b) D_th1..D_th4 (D_1)^{-1} undefined on t^{-1}phi(theta)",
       len(undef_right) == 16 and all(a == -1 for (a, S) in undef_right),
       "%d monomials of k^L(1|4), among them %s"
       % (len(undef_right), ", ".join(sorted("t^{-1}" + "".join("th%d" % j for j in S)
                                             for (a, S) in undef_right)[:3])))
# ... and they are genuine elements of k^L'(1|4): only ONE of them is excluded
report("    of which only t^{-1}th1th2th3th4 is excluded from k^L'(1|4)",
       sum(1 for k in undef_right if k != excluded) == 15,
       "%d survive" % sum(1 for k in undef_right if k != excluded))

# (c) the literal leftmost composition, for the record: it differs from
#     Delta_IS exactly at the zeros of the falling factorial 
diff = []
for k in ks4p:
    try:
        lit = Delta_IS_literal(mono(*k))
    except ValueError:
        diff.append((k, "undefined"))
        continue
    if not eq(lit, Delta_IS(mono(*k))):
        diff.append((k, "differs"))
report("(c) the literal composition differs from Delta_IS at %d monomials"
       % len(diff), len(diff) > 0, str([d[0] for d in diff[:3]]))

# (d) Delta_IS is the n=4 member of the uniform formula: a cocycle
small4 = [k for k in keys_of(4, range(-2, 3)) if k != excluded]


def c4(f, g):
    return L_res(4, mul(f, Delta_IS(g))) * (-1) ** parity(f)


bad = check_cocycle_identity(4, lambda f, g: L_bracket(4, f, g), c4, small4)
report("(d) c(f,g) = (-1)^{p(f)} Res(f Delta_IS(g)) is a cocycle on %d^3 triples"
       % len(small4), bad == 0, "" if bad == 0 else "%d failures" % bad)

# ------------------------------------------------------------ 5) the weight

print("\n5) the weight of cD_n is the sum of the weights of the factors")
for n in range(0, 4):
    w = F(3 - n, 1) * 1 + n * F(1, 2)       # ord: del_t counts 1, D_th counts 1/2
    report("n=%d: ord(s_n) = %s, mu = %s" % (n, w, F(3 - n, 2)),
           w == F(6 - n, 2) and F(3 - n, 2) == F(-1, 2) + (w - 1),
           "cF_{-1/2} --> cF_{(3-n)/2}")

print("\n%d checks, %d failures" % (NCHECK, FAIL))
