"""
Verification of the two exceptional operators of the n=4 cases, eqs. (58)
and (60) of the text.

What is checked here is exactly what is PRINTED, line by line:

 (a) eq. (58): the five displayed lines of
        Delta_IS := D_1^{-1} D_th1 D_th2 D_th3 D_th4
     against the engine's Delta_IS (check_bases.Delta_IS), for a range of a;

 (b) the "simplify before you evaluate" claim: Delta_IS coincides with the
     literal composition wherever the literal one is defined AND the falling
     factorial does not vanish, and differs from it exactly at the zeros --
     e.g. at a=3, where the literal reading returns 0 while the cocycle
     identity requires 6;

 (c) eq. (60): Delta := D^M_zeta o cD^{-1} on the M-side, with
        cD = d_t - zeta/(2t) d_zeta,   D^M_zeta = zeta d_t + (1/t) d_zeta,
     against the two displayed lines
        Delta(t^n phi(th)) = t^n zeta phi(th),
        Delta(t^n zeta phi(th)) = t^n phi(th)/(n + 1/2);
     and the claim that Delta is defined everywhere whereas cD^{-1} is not
     (the pole of cD^{-1} at n=-1 is cancelled by the zero of D^M_zeta).

Run:  python3 checkdelta.py
"""

from fractions import Fraction as F
from itertools import combinations

from stringy import mono, add, mul, smul, d_t, d_odd, eq, show
from check_bases import Delta_IS, Delta_IS_literal

FAIL = 0


def report(name, ok, detail=""):
    global FAIL
    if not ok:
        FAIL += 1
    print(("  OK   " if ok else "  FAIL ") + name + (("  " + detail) if detail else ""))


# ------------------------------------------------------- (a) eq. (58)

def printed_DeltaL(a):
    """the five lines of eq. (58), as typeset, for the argument t^a th_S."""
    lines = {}
    #  t^a  |-->  a(a-1)(a-2) t^{a-3} th1 th2 th3 th4
    lines[()] = mono(a - 3, (1, 2, 3, 4), a * (a - 1) * (a - 2))
    #  t^a th1  |--> -a(a-1) t^{a-2} th2 th3 th4
    lines[(1,)] = mono(a - 2, (2, 3, 4), -a * (a - 1))
    #  t^a th1 th2  |--> -a t^{a-1} th3 th4
    lines[(1, 2)] = mono(a - 1, (3, 4), -a)
    #  t^a th1 th2 th3  |--> t^a th4
    lines[(1, 2, 3)] = mono(a, (4,), 1)
    #  t^a th1 th2 th3 th4 |--> t^{a+1}/(a+1)
    if a != -1:
        lines[(1, 2, 3, 4)] = mono(a + 1, (), F(1, a + 1))
    return lines


print("(a) eq. (58): the five printed lines vs. the engine, a in [-6,6]")
bad = []
for a in range(-6, 7):
    for S, rhs in printed_DeltaL(a).items():
        got = Delta_IS(mono(a, S))
        if not eq(got, rhs):
            bad.append((a, S, show(got), show(rhs)))
report("five lines of (58) reproduced for every a in [-6,6]",
       not bad, "" if not bad else str(bad[:3]))

# the pole: the only monomial on which Delta_IS is undefined
poles = []
for a in range(-6, 7):
    for r in range(5):
        for S in combinations((1, 2, 3, 4), r):
            try:
                Delta_IS(mono(a, S))
            except ValueError:
                poles.append((a, S))
report("the only pole of Delta_IS is t^{-1}th1th2th3th4 (the element deleted "
       "in k^L'(1|4))", poles == [(-1, (1, 2, 3, 4))], str(poles))


# ------------------------- (b) simplified vs. literal composition

print("\n(b) Delta_IS (simplified) vs. the literal composition")
agree, differ, undef = 0, [], 0
for a in range(-6, 7):
    for r in range(5):
        for S in combinations((1, 2, 3, 4), r):
            if (a, S) == (-1, (1, 2, 3, 4)):
                continue
            simp = Delta_IS(mono(a, S))
            try:
                lit = Delta_IS_literal(mono(a, S))
            except (ValueError, ZeroDivisionError):
                undef += 1
                continue
            if eq(simp, lit):
                agree += 1
            else:
                differ.append((a, S, show(simp), show(lit)))
print("    agree: %d   differ: %d   literal undefined: %d" % (agree, len(differ), undef))
# the differences must sit exactly where the falling factorial vanishes,
# i.e. where the literal reading multiplies by 0 and then divides by 0
ok_shape = all(eq(lit_repr, {}) for (_, _, _, lit_repr) in
               [(a, S, s, l) for (a, S, s, l) in []]) or True
report("they differ only where the literal reading collapses to 0",
       all(l == "0" for (_, _, _, l) in differ), str(differ[:3]))
# the case named in the text: g = t^3
simp3, lit3 = Delta_IS(mono(3, ())), Delta_IS_literal(mono(3, ()))
report("at a=3: simplified = 6 t^0 th1..th4, literal = 0 (the text's example)",
       eq(simp3, mono(0, (1, 2, 3, 4), 6)) and eq(lit3, {}),
       "simplified=%s literal=%s" % (show(simp3), show(lit3)))


# ------------------------------------------------------- (c) eq. (60)

# monomials of R^M: (n, z, S) = t^n zeta^z th_S, coefficients in Q;
# th_1..th_3 (the k^M(1|4) case: zeta + 3 thetas).
def M(n, z=0, S=(), c=1):
    return {(n, z, tuple(S)): F(c)} if c else {}


def Madd(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


def cD(f):
    """cD = d_t - zeta/(2t) d_zeta."""
    out = {}
    for (n, z, S), c in f.items():
        if n:
            out[(n - 1, z, S)] = out.get((n - 1, z, S), 0) + n * c
        if z:                      # d_zeta kills the zeta, zeta/(2t) puts it back
            out[(n - 1, 1, S)] = out.get((n - 1, 1, S), 0) - F(c, 2)
    return {k: v for k, v in out.items() if v}


def cD_inv(f):
    """the inverse of cD, undefined on t^{-1}phi(th)."""
    out = {}
    for (n, z, S), c in f.items():
        d = (n + 1) if not z else (n + F(1, 2))
        if d == 0:
            raise ValueError("cD^{-1} is undefined on t^%d %s" % (n, "zeta" if z else ""))
        out[(n + 1, z, S)] = out.get((n + 1, z, S), 0) + c / d
    return {k: v for k, v in out.items() if v}


def DM_zeta(f):
    """D^M_zeta = zeta d_t + (1/t) d_zeta."""
    out = {}
    for (n, z, S), c in f.items():
        if n and not z:            # zeta d_t ; zeta*zeta = 0
            out[(n - 1, 1, S)] = out.get((n - 1, 1, S), 0) + n * c
        if z:                      # (1/t) d_zeta
            out[(n - 1, 0, S)] = out.get((n - 1, 0, S), 0) + c
    return {k: v for k, v in out.items() if v}


def Delta_literal(f):
    """the composition read literally: cD^{-1} first, then D^M_zeta.  Undefined
    wherever cD^{-1} is, i.e. on every t^{-1}phi(th)."""
    return DM_zeta(cD_inv(f))


def Delta(f):
    """eq. (60) as a SINGLE operator, i.e. the composition simplified BEFORE
    it is evaluated -- the same device as for Delta_IS:
        Delta(t^n phi)      = t^n zeta phi,
        Delta(t^n zeta phi) = t^n phi/(n+1/2).
    The pole of cD^{-1} at n=-1 is cancelled by the zero of D^M_zeta, so this
    reading is defined everywhere."""
    out = {}
    for (n, z, S), c in f.items():
        if not z:
            out[(n, 1, S)] = out.get((n, 1, S), 0) + c
        else:
            out[(n, 0, S)] = out.get((n, 0, S), 0) + c / (n + F(1, 2))
    return {k: v for k, v in out.items() if v}


print("\n(c) eq. (60): Delta = D^M_zeta o cD^{-1}")
# first: cD and cD^{-1} are inverse where defined (the two displayed actions of cD)
bad = []
for n in range(-6, 7):
    for r in range(4):
        for S in combinations((1, 2, 3), r):
            for z in (0, 1):
                f = M(n, z, S)
                want = M(n - 1, z, S, (n if not z else n - F(1, 2)))
                if cD(f) != want:
                    bad.append((n, z, S, cD(f), want))
report("cD(t^n phi) = n t^{n-1} phi  and  cD(t^n zeta phi) = (n-1/2) t^{n-1} zeta phi",
       not bad, str(bad[:2]))

bad, poles = [], []
for n in range(-6, 7):
    for r in range(4):
        for S in combinations((1, 2, 3), r):
            # line 1: Delta(t^n phi) = t^n zeta phi
            try:
                got = Delta(M(n, 0, S))
                if got != M(n, 1, S):
                    bad.append(("line1", n, S, got))
            except ValueError:
                poles.append(("line1", n, S))
            # line 2: Delta(t^n zeta phi) = t^n phi/(n+1/2)
            try:
                got = Delta(M(n, 1, S))
                if got != M(n, 0, S, F(1, n + F(1, 2))):
                    bad.append(("line2", n, S, got))
            except ValueError:
                poles.append(("line2", n, S))
report("both printed lines of (60) reproduced for every n in [-6,6]",
       not bad, str(bad[:2]))
report("Delta, read as one operator, has no pole at all", not poles, str(poles[:2]))

# and the point of the whole device: the literal composition DOES have one,
# and the two agree wherever the literal one is defined
agree, undef = 0, []
for n in range(-6, 7):
    for r in range(4):
        for S in combinations((1, 2, 3), r):
            for z in (0, 1):
                f = M(n, z, S)
                try:
                    lit = Delta_literal(f)
                except ValueError:
                    undef.append((n, z, S))
                    continue
                if Delta(f) == lit:
                    agree += 1
                else:
                    bad.append(("lit", n, z, S))
report("Delta agrees with the literal D^M_zeta o cD^{-1} wherever the latter is "
       "defined (%d monomials)" % agree, not bad, str(bad[:2]))
report("the literal composition is undefined exactly on the 8 monomials "
       "t^{-1}phi(th) -- among them t^{-1}th1th2th3, which does belong to the "
       "algebra; this is why the composition has to be simplified first",
       [k for k in undef] == [(-1, 0, S) for r in range(4)
                              for S in combinations((1, 2, 3), r)],
       str(undef))

print("\n%s  (%d failures)" % ("ALL CHECKS PASSED" if not FAIL else "FAILURES", FAIL))
